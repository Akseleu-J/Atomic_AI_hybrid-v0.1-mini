"""
optimizer.py -- гибридный оптимизатор (Muon для >=2D весов, AdamW для
эмбеддинга/норм/биасов/decay_a) + warmup->cosine LR schedule +
byte-level cross-entropy loss.

Новый проект с нуля -- упрощённая, но функционально аналогичная старому
hybrid-стеку схема: Muon-ортогонализация (Newton-Schulz) для больших
2D Dense-ядер (магистраль сети), AdamW -- для всего остального.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from utils import path_to_str


# ==========================================================================
# Muon (Newton-Schulz ортогонализация градиента)
# ==========================================================================
def _newton_schulz(X, steps: int = 5):
    a, b, c = 3.4445, -4.7750, 2.0315
    was_tall = X.shape[-2] > X.shape[-1]
    if was_tall:
        X = jnp.swapaxes(X, -2, -1)
    norm = jnp.linalg.norm(X, axis=(-2, -1), keepdims=True)
    X = X / (norm * 1.01 + 1e-7)
    for _ in range(steps):
        A = X @ jnp.swapaxes(X, -2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if was_tall:
        X = jnp.swapaxes(X, -2, -1)
    return X


def _muon_orthogonalize(g, ns_steps: int = 5):
    eps = 1e-7
    norm = jnp.linalg.norm(g)
    safe_norm = jnp.where(norm < eps, 1.0, norm)
    X = g / safe_norm
    X = _newton_schulz(X, ns_steps)
    return jnp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), norm


class MuonState(NamedTuple):
    count: jnp.ndarray


def muon_transform(base_lr_fn, ns_steps: int = 5, weight_decay: float = 0.02):
    def init_fn(params):
        return MuonState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        step_lr = base_lr_fn(state.count)

        def _upd(p, g):
            X, gnorm = _muon_orthogonalize(g, ns_steps)
            eff_lr = jnp.where(gnorm < 1e-7, 0.0, step_lr)
            return -(X * eff_lr) - eff_lr * weight_decay * p

        new_updates = jax.tree_util.tree_map(_upd, params, updates)
        return new_updates, MuonState(count=state.count + 1)

    return optax.GradientTransformation(init_fn, update_fn)


# ==========================================================================
# LR schedule -- линейный warmup -> cosine decay до min_lr_ratio*peak_lr
# ==========================================================================
def make_lr_schedule(total_steps: int, peak_lr: float, warmup_ratio: float, min_lr_ratio: float):
    warmup_steps = max(100, int(total_steps * warmup_ratio))
    warmup = optax.linear_schedule(0.0, peak_lr, warmup_steps)
    cosine = optax.cosine_decay_schedule(
        peak_lr, max(1, total_steps - warmup_steps), alpha=min_lr_ratio
    )
    return optax.join_schedules([warmup, cosine], boundaries=[warmup_steps])


# ==========================================================================
# Метки параметров (какой из двух оптимизаторов их ведёт)
# ==========================================================================
def _label_leaf(path, param):
    path_str = path_to_str(path)
    if "embed" in path_str or "lm_head" in path_str:
        return "adamw"
    if "norm" in path_str or path_str.endswith("_b") or "_conv_b" in path_str or "bias" in path_str:
        return "adamw"
    if "decay_a" in path_str:
        return "adamw"
    if param.ndim >= 2:
        return "muon"
    return "adamw"


def make_label_fn():
    def label_fn(params):
        return jax.tree_util.tree_map_with_path(_label_leaf, params)
    return label_fn


def make_hybrid_optimizer(total_steps: int, peak_lr: float, weight_decay: float,
                           warmup_ratio: float = 0.03, min_lr_ratio: float = 0.1,
                           grad_clip_norm: float = 1.0):
    """tx.update(grads, opt_state, params) -- стандартный optax-интерфейс.
    Возвращает (tx, lr_schedule); lr_schedule нужен train.py только для
    логирования текущего LR (сам расчёт уже встроен в tx)."""
    lr_schedule = make_lr_schedule(total_steps, peak_lr, warmup_ratio, min_lr_ratio)

    tx_adamw = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    # Muon традиционно использует более высокий LR, чем AdamW на тех же
    # шагах -- шкалируем ту же cosine-кривую, а не заводим отдельный
    # независимый schedule (иначе он рассинхронизируется с warmup AdamW).
    tx_muon = muon_transform(lambda step: lr_schedule(step) * 3.0, ns_steps=5, weight_decay=weight_decay)

    label_fn = make_label_fn()
    multi_tx = optax.multi_transform({"adamw": tx_adamw, "muon": tx_muon}, label_fn)

    tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm), multi_tx)
    return tx, lr_schedule


# ==========================================================================
# Byte-level cross-entropy
# ==========================================================================
def compute_loss(params, apply_fn, batch, label_smoothing: float = 0.0):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    logits = apply_fn({"params": params}, input_ids, deterministic=True)
    logits = jnp.clip(jnp.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4), -1e4, 1e4)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

    vocab_size = logits.shape[-1]
    labels_safe = jnp.clip(labels, 0, vocab_size - 1)
    nll = -jnp.take_along_axis(log_probs, labels_safe[..., None], axis=-1).squeeze(-1)

    if label_smoothing > 0:
        smooth_pos = 1.0 - label_smoothing
        smooth_neg = label_smoothing / (vocab_size - 1)
        sum_log_probs = jnp.sum(log_probs, axis=-1)
        loss_vec = nll * (smooth_pos - smooth_neg) - smooth_neg * sum_log_probs
    else:
        loss_vec = nll

    mask = (labels >= 0).astype(jnp.float32)
    mean_loss = jnp.sum(loss_vec * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    mean_loss = jnp.nan_to_num(mean_loss, nan=0.0, posinf=20.0, neginf=0.0)
    return mean_loss
