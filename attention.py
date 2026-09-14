"""
attention.py -- две ядровые механики гибридной модели, обе написаны с
нуля для этого проекта.

GDN-2 (Gated DeltaNet-2): собственная чистая JAX реализация gated
delta-rule как scan по chunk'ам (jax.checkpoint на границе каждого
chunk'а для контроля памяти), внутри chunk'а -- ещё один scan по токенам
(delta-rule по своей природе строго последовательна: state в шаге t
зависит от state в шаге t-1). Это НЕ Pallas-кернел -- сознательный выбор
для "с нуля" версии: корректность и простота отладки важнее пиковой
скорости на этом этапе. Сигнатура gdn2_chunked_delta_rule() не завязана
на реализацию -- при необходимости внутренности можно заменить на
Pallas-кернели позже, не трогая model.py.

MLA (Multi-head Latent Attention): использует TPU flash-attention ядро из
jax.experimental.pallas.ops.tpu.flash_attention (библиотечный Pallas
кернел из самого JAX, не собственный код) -- O(L) память вместо
O(L^2) материализации attention-матрицы. Graceful fallback на обычный
softmax-attention, если пакет недоступен (CPU/GPU отладка, малые L).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as _pallas_flash_attention,
        BlockSizes as _FlashBlockSizes,
    )
    _HAS_FLASH = True
    _FLASH_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - платформозависимо
    _pallas_flash_attention = None
    _FlashBlockSizes = None
    _HAS_FLASH = False
    _FLASH_IMPORT_ERROR = _e


_HIGHEST = jax.lax.Precision.HIGHEST


def _sanitize(x, clip=3e2):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)


def flash_available() -> bool:
    return _HAS_FLASH


# ==========================================================================
# GDN-2 -- gated delta rule, chunked-scan (чистый JAX)
# ==========================================================================
def _gdn2_token_step(state, inputs):
    """Один токен. state: (H, D, Dv) -- key->value ассоциативная память.
    k_t,q_t,erase_t: (H, D). v_t,write_t: (H, Dv). alpha_t: (H,) decay."""
    k_t, v_t, q_t, alpha_t, erase_t, write_t = inputs
    state = state * alpha_t[:, None, None]
    erase_signal = erase_t * k_t
    retrieved = jnp.einsum("hd,hdv->hv", erase_signal, state, precision=_HIGHEST)
    v_new = write_t * v_t - retrieved
    state = state + jnp.einsum("hd,hv->hdv", k_t, v_new, precision=_HIGHEST)
    out = jnp.einsum("hdv,hd->hv", state, q_t, precision=_HIGHEST)
    return state, out


def _gdn2_chunk_scan(state, chunk_inputs):
    def step(carry, inp):
        new_state, out = _gdn2_token_step(carry, inp)
        new_state = _sanitize(new_state, clip=1e4)
        return new_state, out
    new_state, outs = jax.lax.scan(step, state, chunk_inputs)
    return new_state, outs


def gdn2_chunked_delta_rule(q, k, v, erase_gate, write_gate, log_decay, chunk_size,
                             h0=None):
    """q,k: (B,L,H,D), L2-нормированные снаружи. v: (B,L,H,Dv).
    erase_gate/write_gate: (B,L,H,D)/(B,L,H,Dv) в [0,1]. log_decay:
    (B,L,H), уже <= 0. h0: (B,H,D,Dv) опционально.

    Возвращает (out (B,L,H,Dv), h_final (B,H,D,Dv)). Честная O(L)
    рекуррентность (scan по chunk'ам с remat, внутри -- scan по токенам),
    без низкоранговых WY-трюков chunk-parallel форм."""
    bsz, L, H, D = q.shape
    Dv = v.shape[-1]
    if L % chunk_size != 0:
        raise ValueError(f"seq_len={L} должен делиться на chunk_size={chunk_size}")
    n_chunks = L // chunk_size

    alpha = jnp.exp(jnp.clip(log_decay.astype(jnp.float32), -20.0, 0.0))

    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, Dv), dtype=jnp.float32)

    def to_chunks(t):
        shp = t.shape
        return t.reshape(bsz, n_chunks, chunk_size, *shp[2:])

    q_c, k_c, v_c, eg_c, wg_c = map(to_chunks, (q, k, v, erase_gate, write_gate))
    alpha_c = to_chunks(alpha)

    def per_example(h0_ex, q_ex, k_ex, v_ex, eg_ex, wg_ex, alpha_ex):
        def chunk_step(state, chunk_idx_inputs):
            q_ch, k_ch, v_ch, eg_ch, wg_ch, alpha_ch = chunk_idx_inputs
            new_state, out_ch = _gdn2_chunk_scan(
                state, (k_ch, v_ch, q_ch, alpha_ch, eg_ch, wg_ch)
            )
            return new_state, out_ch

        chunk_step = jax.checkpoint(chunk_step)

        scan_inputs = (q_ex, k_ex, v_ex, eg_ex, wg_ex, alpha_ex)
        h_final, out_chunks = jax.lax.scan(chunk_step, h0_ex, scan_inputs)
        return out_chunks, h_final

    out_chunks, h_final = jax.vmap(per_example)(h0, q_c, k_c, v_c, eg_c, wg_c, alpha_c)
    out = out_chunks.reshape(bsz, L, H, Dv)
    return _sanitize(out, clip=1e3).astype(q.dtype), h_final


# ==========================================================================
# MLA -- flash-attention (библиотечное Pallas-ядро) с graceful fallback
# ==========================================================================
def mla_causal_attention(q, k, v, sm_scale, mesh=None, batch_axis=None):
    """q,k,v: (B,H,L,D). Возвращает (B,H,L,Dv)."""
    if _HAS_FLASH:
        def _flash_call(q_l, k_l, v_l):
            local_b = q_l.shape[0]
            block_sizes = _FlashBlockSizes(
                block_q=512, block_k_major=512, block_k=512, block_b=local_b,
                block_q_major_dkv=512, block_k_major_dkv=512, block_k_dkv=256,
                block_q_dkv=512, block_k_major_dq=512, block_k_dq=256, block_q_dq=512,
            )
            return _pallas_flash_attention(
                q_l, k_l, v_l, causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
            )

        if mesh is not None:
            from jax.sharding import PartitionSpec as P
            spec = P(batch_axis, None, None, None)
            sharded = jax.shard_map(
                _flash_call, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False,
            )
            return sharded(q, k, v)
        return _flash_call(q, k, v)

    # -------- fallback: explicit softmax attention (CPU/GPU, малые L) --------
    L = q.shape[-2]
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision=_HIGHEST) * sm_scale
    causal_mask = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))[None, None, :, :]
    scores = jnp.where(causal_mask, scores, -1e9)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkv->bhqv", attn, v, precision=_HIGHEST)
