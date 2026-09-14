"""
model.py -- гибридная GDN-2/MLA модель c Delta Attention Residuals (DAR).

Новый проект (не использует существующий atomic_ops/model.py).

Архитектура:
  - DAR-блок из cfg.layers_per_block слоёв, каждый либо "gdn2", либо "mla".
    Соотношение gdn2:mla = 4:1 (см. configs.ModelConfig.layer_types) --
    держится ровно на уровне каждого блока.
  - Вместо наивного `x = x + delta` каждый слой ПЕРЕД вычислением своей
    подсети делает retrieval-attention ("Delta Attention Residual",
    DeltaAttentionResidual ниже) по накопленным дельтам -- своим внутри
    текущего блока (local_deltas) и дельтам ВСЕХ прошлых блоков целиком
    (history_blocks). Softmax по оси "источник" позволяет каждому слою
    выбрать, какая предыдущая дельта сейчас релевантна на каждой позиции,
    вместо фиксированного равновесного сложения всех дельт подряд.
  - MLA использует TPU flash-attention (attention.mla_causal_attention).
  - GDN-2 использует attention.gdn2_chunked_delta_rule (чистый JAX scan).
  - Один SwiGLU FFN на блок (не per-layer, не MoE) -- держит бюджет
    параметров предсказуемым и убирает роутинг из первой версии проекта.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from configs import ModelConfig
from attention import gdn2_chunked_delta_rule, mla_causal_attention

_HIGHEST = jax.lax.Precision.HIGHEST

_model_mesh = None
_batch_axis = None


def set_model_mesh(mesh, batch_axis=None):
    global _model_mesh, _batch_axis
    _model_mesh = mesh
    _batch_axis = batch_axis


def get_model_mesh():
    return _model_mesh


def get_batch_axis():
    return _batch_axis


def _sanitize(x, clip=1e3):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def make_grad_sanitizer(tag: str, clip_val: float = 1e3):
    """custom_vjp: identity в forward, чинит (clip + nan_to_num) non-finite
    градиент в backward -- защищает узлы стыковки (embed, выход
    sub-слоя) от того, чтобы один плохой шаг/токен уронил весь
    оптимизатор через один параметр, используемый в нескольких местах
    (например, tied embedding)."""
    @jax.custom_vjp
    def _fn(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        g_safe = jnp.nan_to_num(
            jnp.clip(g, -clip_val, clip_val), nan=0.0, posinf=clip_val, neginf=-clip_val
        )
        return (g_safe,)

    _fn.defvjp(_fwd, _bwd)
    return _fn


# ==========================================================================
# RoPE (используется только MLA -- GDN-2 кодирует позицию через decay)
# ==========================================================================
class RoPE(nn.Module):
    dim: int
    theta: float = 10000.0

    @nn.compact
    def __call__(self, seq_len):
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2) / self.dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x, cos, sin):
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    rot = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


# ==========================================================================
# GDN-2 sublayer
# ==========================================================================
class GDN2Sublayer(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.cfg
        b, l, d = x.shape
        H, D = cfg.n_heads, cfg.d_head
        eps = 1e-6

        def short_conv(name, u):
            w = self.param(f"{name}_conv_w", nn.initializers.normal(0.02), (d, cfg.d_conv))
            bconv = self.param(f"{name}_conv_b", nn.initializers.zeros, (d,))
            rhs = w.T[:, None, :].astype(u.dtype)
            out = jax.lax.conv_general_dilated(
                u, rhs, window_strides=(1,), padding=[(cfg.d_conv - 1, 0)],
                feature_group_count=d, dimension_numbers=("NHC", "HIO", "NHC"),
            )
            return out + bconv[None, None, :].astype(u.dtype)

        q_lin = nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="q_proj")(x)
        k_lin = nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="k_proj")(x)
        v_lin = nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="v_proj")(x)

        q = jax.nn.silu(short_conv("q", q_lin)).reshape(b, l, H, D)
        k = jax.nn.silu(short_conv("k", k_lin)).reshape(b, l, H, D)
        v = jax.nn.silu(short_conv("v", v_lin)).reshape(b, l, H, D)
        v = jnp.clip(v, -50.0, 50.0)

        def l2norm(t):
            return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps ** 2)

        q = make_grad_sanitizer("gdn2_q_norm")(l2norm(q))
        k = make_grad_sanitizer("gdn2_k_norm")(l2norm(k))

        erase = jax.nn.sigmoid(
            nn.Dense(d, use_bias=True, dtype=jnp.bfloat16, name="erase_gate")(x)
        ).reshape(b, l, H, D)
        write = jax.nn.sigmoid(
            nn.Dense(d, use_bias=True, dtype=jnp.bfloat16, name="write_gate")(x)
        ).reshape(b, l, H, D)

        decay_a = self.param("decay_a", nn.initializers.zeros, (H,)).astype(jnp.float32)
        decay_f = nn.Dense(H, use_bias=True, dtype=jnp.bfloat16, name="decay_proj")(x).astype(jnp.float32)
        log_decay = -jnp.exp(jnp.clip(decay_a, -20.0, 20.0))[None, None, :] * jax.nn.softplus(decay_f)
        log_decay = jnp.nan_to_num(log_decay, nan=0.0, posinf=0.0, neginf=-20.0)

        out_gate = jnp.clip(
            nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="out_gate")(x), -1e2, 1e2
        )

        q, k, v, erase, write = map(_sanitize, (q, k, v, erase, write))

        mesh = get_model_mesh()
        batch_axis = get_batch_axis()
        _fwd = partial(gdn2_chunked_delta_rule, chunk_size=cfg.gdn2_chunk_size)

        if mesh is not None:
            spec4 = P(batch_axis, None, None, None)
            spec3 = P(batch_axis, None, None)
            sharded = jax.shard_map(
                _fwd, mesh=mesh,
                in_specs=(spec4, spec4, spec4, spec4, spec4, spec3),
                out_specs=(spec4, P(batch_axis, None, None, None)),
                check_vma=False,
            )
            out, _h_final = sharded(q, k, v, erase, write, log_decay)
        else:
            out, _h_final = _fwd(q, k, v, erase, write, log_decay)

        out = out.reshape(b, l, d)
        out = nn.RMSNorm(epsilon=1e-6, name="out_norm")(out).astype(x.dtype)
        return nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="out_proj")(
            out * jax.nn.silu(out_gate)
        )


# ==========================================================================
# MLA sublayer (flash-attention)
# ==========================================================================
class MLASublayer(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, cos, sin):
        cfg = self.cfg
        b, l, d = x.shape
        H, D = cfg.n_heads, cfg.d_head

        Q = nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="W_q")(x)
        Q = Q.reshape(b, l, H, D).transpose(0, 2, 1, 3)

        kv = nn.Dense(cfg.d_latent, use_bias=False, dtype=jnp.bfloat16, name="W_kv_down")(x)
        K = nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="W_k_up")(kv)
        V = nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="W_v_up")(kv)
        K = K.reshape(b, l, H, D).transpose(0, 2, 1, 3)
        V = V.reshape(b, l, H, D).transpose(0, 2, 1, 3)

        Q = apply_rope(Q, cos[None, None, :, :D], sin[None, None, :, :D])
        K = apply_rope(K, cos[None, None, :, :D], sin[None, None, :, :D])
        Q, K, V = map(_sanitize, (Q, K, V))

        sm_scale = 1.0 / math.sqrt(D)
        out = mla_causal_attention(Q, K, V, sm_scale, mesh=get_model_mesh(), batch_axis=get_batch_axis())
        out = out.transpose(0, 2, 1, 3).reshape(b, l, d).astype(x.dtype)
        out = nn.RMSNorm(epsilon=1e-6, name="attn_out_norm")(out).astype(x.dtype)
        out = make_grad_sanitizer("mla_attn_out")(out)
        return nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="W_o")(out)


# ==========================================================================
# Delta Attention Residual (DAR)
# ==========================================================================
class DeltaAttentionResidual(nn.Module):
    """Retrieval-attention поверх накопленных дельт (истории прошлых
    блоков + локальных дельт текущего блока). Заменяет наивное
    `current_x = current_x + sum(deltas)` на softmax-взвешенный выбор,
    какая дельта релевантна ТЕКУЩЕМУ слою на каждой позиции -- отсюда
    "Delta Attention Residual"."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, current_x, sources):
        n = len(sources)
        if n == 0:
            return jnp.zeros_like(current_x)
        if n == 1:
            # softmax по единственному источнику тождественно даёт вес 1 --
            # q_proj/k_proj получили бы структурно нулевой градиент, поэтому
            # прямой shortcut (экономит FLOPs, не портит диагностику весов).
            return sources[0].astype(current_x.dtype)

        b, l, d = current_x.shape
        stack = jnp.stack(sources, axis=0)  # (n, b, l, d)

        q = nn.Dense(self.cfg.d_latent, use_bias=False, dtype=jnp.bfloat16, name="q_proj")(current_x)
        flat = stack.reshape(n * b * l, d)
        k_flat = nn.Dense(self.cfg.d_latent, use_bias=False, dtype=jnp.bfloat16, name="k_proj")(flat)
        k = k_flat.reshape(n, b, l, self.cfg.d_latent)

        scale = 1.0 / math.sqrt(self.cfg.d_latent)
        scores = jnp.einsum("bld,nbld->nbl", q, k, precision=_HIGHEST) * scale
        weights = jax.nn.softmax(scores, axis=0)
        retrieved = jnp.einsum("nbl,nbld->bld", weights, stack, precision=_HIGHEST)
        return retrieved.astype(current_x.dtype)


# ==========================================================================
# Один слой DAR-блока
# ==========================================================================
class DARLayer(nn.Module):
    cfg: ModelConfig
    layer_type: str
    layer_idx: int

    @nn.compact
    def __call__(self, current_x, local_deltas, history_blocks, cos, sin):
        dar_sources = [history_blocks[j] for j in range(history_blocks.shape[0])] + list(local_deltas)
        retrieved = DeltaAttentionResidual(cfg=self.cfg, name="dar")(current_x, dar_sources)
        current_x = current_x + retrieved

        normed = nn.RMSNorm(epsilon=1e-6, name="pre_sublayer_norm")(current_x)

        if self.layer_type == "gdn2":
            delta = GDN2Sublayer(cfg=self.cfg, name="sublayer")(normed)
        elif self.layer_type == "mla":
            delta = MLASublayer(cfg=self.cfg, name="sublayer")(normed, cos, sin)
        else:
            raise ValueError(f"Unknown layer_type={self.layer_type!r}")

        delta = make_grad_sanitizer(f"delta_layer{self.layer_idx}_{self.layer_type}")(delta)
        delta = _sanitize(delta)
        return current_x, delta


# ==========================================================================
# SwiGLU FFN (один на блок)
# ==========================================================================
class BlockFFN(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.cfg.d_ff, use_bias=False, dtype=jnp.bfloat16, name="gate_proj")(x)
        up = nn.Dense(self.cfg.d_ff, use_bias=False, dtype=jnp.bfloat16, name="up_proj")(x)
        h = jax.nn.silu(gate) * up
        return nn.Dense(self.cfg.d_model, use_bias=False, dtype=jnp.bfloat16, name="down_proj")(h)


# ==========================================================================
# DAR-блок
# ==========================================================================
class DARBlock(nn.Module):
    cfg: ModelConfig
    block_idx: int
    layer_idx_start: int

    @nn.compact
    def __call__(self, current_x, history_blocks, cos, sin):
        local_deltas = []
        for i in range(self.cfg.layers_per_block):
            layer_idx = self.layer_idx_start + i
            layer_type = self.cfg.layer_types[layer_idx]
            current_x, delta = DARLayer(
                cfg=self.cfg, layer_type=layer_type, layer_idx=layer_idx,
                name=f"layer_{layer_idx}",
            )(current_x, local_deltas, history_blocks, cos, sin)
            local_deltas.append(delta)
            current_x = _sanitize(current_x)

        block_delta = sum(local_deltas)
        new_history = jnp.concatenate([history_blocks, block_delta[None, ...]], axis=0)

        normed = nn.RMSNorm(epsilon=1e-6, name="pre_ffn_norm")(current_x)
        ffn_out = BlockFFN(cfg=self.cfg, name="ffn")(normed)
        output = _sanitize(current_x + ffn_out)
        return output, new_history


# ==========================================================================
# Полная модель
# ==========================================================================
class HybridByteLM(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True, return_hidden: bool = False):
        cfg = self.cfg
        b, l = input_ids.shape

        embed = nn.Embed(
            num_embeddings=cfg.vocab_size, features=cfg.d_model, dtype=jnp.bfloat16, name="embed"
        )
        x = embed(input_ids)
        x = make_grad_sanitizer("embed_input", clip_val=1e3)(x)

        cos, sin = RoPE(dim=cfg.d_head, theta=cfg.rope_theta)(l)

        history_blocks = jnp.zeros((0, b, l, cfg.d_model), dtype=x.dtype)

        RematBlock = nn.remat(DARBlock)
        for block_idx in range(cfg.num_blocks):
            layer_idx_start = block_idx * cfg.layers_per_block
            x, history_blocks = RematBlock(
                cfg=cfg, block_idx=block_idx, layer_idx_start=layer_idx_start,
                name=f"block_{block_idx}",
            )(x, history_blocks, cos, sin)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x).astype(x.dtype)
        if return_hidden:
            return final

        if cfg.tie_embeddings:
            logits = embed.attend(final)
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False, dtype=jnp.bfloat16, name="lm_head")(final)
        return logits


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
