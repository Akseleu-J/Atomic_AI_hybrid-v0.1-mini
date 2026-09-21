"""
model.py -- гибридная GDN-2/MLA модель БЕЗ Delta Attention Residuals (DAR).

Упрощённая версия: классические residual-связки `x = x + sublayer(norm(x))`.
DAR механизм (retrieval-attention поверх накопленных дельт) удалён —
это снимает ~20% времени шага и значительный объём активаций в HBM.

Остальное:
  - GDN-2 через atomic_ops.fallback.gdn2_forward_trainable (shim подменяет
    на atomic_gdn2 если активен).
  - MLA через TPU flash-attention.
  - Один SwiGLU FFN на блок из cfg.layers_per_block слоёв.
  - Ремat применяется к целому блоку для контроля памяти.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from configs import ModelConfig
from attention import mla_causal_attention
from atomic_ops.fallback import gdn2_forward_trainable
from atomic_ops.configs import KernelConfig as AtomicKernelConfig

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


def make_grad_sanitizer(clip_val: float = 1e3):
    """custom_vjp: identity в forward, clip + nan_to_num в backward.
    Используется ТОЛЬКО на embed_input (tied embeddings)."""
    @jax.custom_vjp
    def _fn(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        g_safe = jnp.nan_to_num(
            jnp.clip(g, -clip_val, clip_val),
            nan=0.0, posinf=clip_val, neginf=-clip_val,
        )
        return (g_safe,)

    _fn.defvjp(_fwd, _bwd)
    return _fn


# ==========================================================================
# RoPE (только для MLA)
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
# GDN-2 sublayer (atomic_ops Pallas kernel + shim на atomic_gdn2)
# ==========================================================================
class GDN2Sublayer(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.cfg
        b, l, d = x.shape
        H = cfg.n_heads
        D = cfg.gdn2_d_head
        proj_dim = H * D
        eps = 1e-6

        assert l % cfg.gdn2_chunk_size == 0, (
            f"seq_len={l} must be divisible by gdn2_chunk_size={cfg.gdn2_chunk_size}"
        )

        def short_conv(name, u):
            dim = u.shape[-1]
            w = self.param(f"{name}_conv_w", nn.initializers.normal(0.02),
                           (dim, cfg.d_conv))
            bconv = self.param(f"{name}_conv_b", nn.initializers.zeros, (dim,))
            rhs = w.T[:, None, :].astype(u.dtype)
            out = jax.lax.conv_general_dilated(
                u, rhs, window_strides=(1,), padding=[(cfg.d_conv - 1, 0)],
                feature_group_count=dim, dimension_numbers=("NHC", "HIO", "NHC"),
            )
            return out + bconv[None, None, :].astype(u.dtype)

        q_lin = nn.Dense(proj_dim, use_bias=False, dtype=jnp.bfloat16, name="q_proj")(x)
        k_lin = nn.Dense(proj_dim, use_bias=False, dtype=jnp.bfloat16, name="k_proj")(x)
        v_lin = nn.Dense(proj_dim, use_bias=False, dtype=jnp.bfloat16, name="v_proj")(x)

        q = jax.nn.silu(short_conv("q", q_lin)).reshape(b, l, H, D).astype(jnp.float32)
        k = jax.nn.silu(short_conv("k", k_lin)).reshape(b, l, H, D).astype(jnp.float32)
        v = jax.nn.silu(short_conv("v", v_lin)).reshape(b, l, H, D).astype(jnp.float32)
        v = jnp.clip(v, -50.0, 50.0)

        def l2norm(t):
            return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps ** 2)

        q = l2norm(q)
        k = l2norm(k)

        erase_b = jax.nn.sigmoid(
            nn.Dense(proj_dim, use_bias=True, dtype=jnp.bfloat16, name="erase_gate")(x)
        ).reshape(b, l, H, D).astype(jnp.float32)
        write_w = jax.nn.sigmoid(
            nn.Dense(proj_dim, use_bias=True, dtype=jnp.bfloat16, name="write_gate")(x)
        ).reshape(b, l, H, D).astype(jnp.float32)

        decay_a = self.param("decay_a", nn.initializers.zeros, (H,)).astype(jnp.float32)
        decay_f = nn.Dense(H, use_bias=True, dtype=jnp.bfloat16,
                           name="decay_proj")(x).astype(jnp.float32)
        log_decay_h = -jnp.exp(jnp.clip(decay_a, -20.0, 20.0))[None, None, :] * jax.nn.softplus(decay_f)
        log_decay_h = jnp.nan_to_num(log_decay_h, nan=0.0, posinf=0.0, neginf=-20.0)

        log_decay = jnp.broadcast_to(log_decay_h[..., None], (b, l, H, D))

        out_gate = jnp.clip(
            nn.Dense(proj_dim, use_bias=False, dtype=jnp.bfloat16, name="out_gate")(x),
            -1e2, 1e2,
        )

        q, k, v, erase_b, write_w, log_decay = map(
            _sanitize, (q, k, v, erase_b, write_w, log_decay)
        )

        bc = cfg.gdn2_chunk_size // 2
        mb = 16 if bc % 16 == 0 else bc
        kernel_config = AtomicKernelConfig(
            bt=cfg.gdn2_chunk_size, bc=bc, mb=mb, wy_eps=1e-3,
        )

        mesh = get_model_mesh()
        batch_axis = get_batch_axis()
        _fwd = partial(gdn2_forward_trainable, scale=1.0, config=kernel_config)

        if mesh is not None:
            spec4 = P(batch_axis, None, None, None)
            sharded = jax.shard_map(
                _fwd, mesh=mesh,
                in_specs=(spec4, spec4, spec4, spec4, spec4, spec4),
                out_specs=(spec4, spec4),
                check_vma=False,
            )
            out, _h_final = sharded(q, k, v, write_w, erase_b, log_decay)
        else:
            out, _h_final = _fwd(q, k, v, write_w, erase_b, log_decay)

        out = out.reshape(b, l, proj_dim)
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
        out = mla_causal_attention(Q, K, V, sm_scale,
                                    mesh=get_model_mesh(),
                                    batch_axis=get_batch_axis())
        out = out.transpose(0, 2, 1, 3).reshape(b, l, d).astype(x.dtype)
        out = nn.RMSNorm(epsilon=1e-6, name="attn_out_norm")(out).astype(x.dtype)
        return nn.Dense(d, use_bias=False, dtype=jnp.bfloat16, name="W_o")(out)


# ==========================================================================
# Один слой (residual + sublayer)
# ==========================================================================
class Layer(nn.Module):
    cfg: ModelConfig
    layer_type: str
    layer_idx: int

    @nn.compact
    def __call__(self, x, cos, sin):
        normed = nn.RMSNorm(epsilon=1e-6, name="pre_sublayer_norm")(x)

        if self.layer_type == "gdn2":
            delta = GDN2Sublayer(cfg=self.cfg, name="sublayer")(normed)
        elif self.layer_type == "mla":
            delta = MLASublayer(cfg=self.cfg, name="sublayer")(normed, cos, sin)
        else:
            raise ValueError(f"Unknown layer_type={self.layer_type!r}")

        delta = _sanitize(delta)
        return _sanitize(x + delta)


# ==========================================================================
# SwiGLU FFN
# ==========================================================================
class BlockFFN(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.cfg.d_ff, use_bias=False, dtype=jnp.bfloat16,
                        name="gate_proj")(x)
        up = nn.Dense(self.cfg.d_ff, use_bias=False, dtype=jnp.bfloat16,
                      name="up_proj")(x)
        h = jax.nn.silu(gate) * up
        return nn.Dense(self.cfg.d_model, use_bias=False, dtype=jnp.bfloat16,
                        name="down_proj")(h)


# ==========================================================================
# Блок: несколько слоёв + один FFN в конце
# ==========================================================================
class Block(nn.Module):
    cfg: ModelConfig
    block_idx: int
    layer_idx_start: int

    @nn.compact
    def __call__(self, x, cos, sin):
        for i in range(self.cfg.layers_per_block):
            layer_idx = self.layer_idx_start + i
            layer_type = self.cfg.layer_types[layer_idx]
            x = Layer(
                cfg=self.cfg, layer_type=layer_type, layer_idx=layer_idx,
                name=f"layer_{layer_idx}",
            )(x, cos, sin)

        normed = nn.RMSNorm(epsilon=1e-6, name="pre_ffn_norm")(x)
        ffn_out = BlockFFN(cfg=self.cfg, name="ffn")(normed)
        return _sanitize(x + ffn_out)


# ==========================================================================
# Полная модель
# ==========================================================================
class HybridByteLM(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True,
                 return_hidden: bool = False):
        cfg = self.cfg
        b, l = input_ids.shape

        embed = nn.Embed(
            num_embeddings=cfg.vocab_size, features=cfg.d_model,
            dtype=jnp.bfloat16, name="embed",
        )
        x = embed(input_ids)
        x = make_grad_sanitizer(clip_val=1e3)(x)

        cos, sin = RoPE(dim=cfg.d_head, theta=cfg.rope_theta)(l)

        RematBlock = nn.remat(Block)
        for block_idx in range(cfg.num_blocks):
            layer_idx_start = block_idx * cfg.layers_per_block
            x = RematBlock(
                cfg=cfg, block_idx=block_idx, layer_idx_start=layer_idx_start,
                name=f"block_{block_idx}",
            )(x, cos, sin)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x).astype(x.dtype)
        if return_hidden:
            return final

        if cfg.tie_embeddings:
            logits = embed.attend(final)
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False,
                              dtype=jnp.bfloat16, name="lm_head")(final)
        return logits


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
