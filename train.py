"""
train.py -- основной цикл обучения гибридной GDN-2/MLA (4:1) модели с
Delta Attention Residuals на enwik8, метрика -- bits-per-byte (bpb).

Полностью новый проект (см. docstring'и configs.py/model.py) -- не
использует существующий atomic_ops/train.py/model.py/optimizer.py.

Запуск: python train.py
Конфигурация -- через configs.ModelConfig / configs.TrainConfig (правьте
поля дефолтов там же, либо создайте свои инстансы и подставьте в main()).
"""
from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import optax

from configs import ModelConfig, TrainConfig
from data_enwik8 import load_enwik8_splits, ByteSequenceLoader, bits_per_byte
from model import HybridByteLM, set_model_mesh, count_params
from optimizer import make_hybrid_optimizer, compute_loss
from sharding import build_mesh, data_sharding, make_param_sharding, make_opt_state_sharding, AXIS_NAME
from checkpoint_utils import make_manager, save as ckpt_save, restore as ckpt_restore, load_metadata
from utils import format_count


def build_train_and_apply(model, tx):
    """(train_micro_step, apply_step):
      - train_micro_step: loss/grad на ОДНОМ микробатче, аккумулирует в accum_grads.
      - apply_step: применяет усреднённый градиент через tx; при non-finite
        global_norm шаг оптимизатора ПРОПУСКАЕТСЯ (веса не меняются), но
        обучение продолжается -- вместо падения всей сессии."""

    def loss_fn(params, batch):
        return compute_loss(params, model.apply, batch)

    grad_fn = jax.value_and_grad(loss_fn)

    def train_micro_step(params, accum_grads, batch):
        loss, grads = grad_fn(params, batch)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1e3, neginf=-1e3), grads
        )
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        micro_grad_norm = optax.global_norm(grads)
        return new_accum, loss, micro_grad_norm

    def apply_step(params, opt_state, accum_grads, accum_steps: int):
        avg_grads = jax.tree_util.tree_map(lambda g: g / accum_steps, accum_grads)
        global_norm = optax.global_norm(avg_grads)
        is_finite = jnp.isfinite(global_norm)

        def _do_update(_):
            updates, new_opt_state = tx.update(avg_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        def _skip(_):
            return params, opt_state

        new_params, new_opt_state = jax.lax.cond(is_finite, _do_update, _skip, operand=None)
        return new_params, new_opt_state, is_finite, global_norm

    train_micro_step_jit = jax.jit(train_micro_step, donate_argnums=(1,))
    apply_step_jit = jax.jit(apply_step, donate_argnums=(0, 1, 2), static_argnums=(3,))
    return train_micro_step_jit, apply_step_jit


def build_eval_step(model):
    def eval_step(params, batch):
        return compute_loss(params, model.apply, batch)
    return jax.jit(eval_step)


def run_eval(eval_step, params, loader: ByteSequenceLoader, max_batches):
    total, n = 0.0, 0
    for batch in loader.eval_epoch(max_batches=max_batches):
        loss = eval_step(params, batch)
        total += float(jax.device_get(loss))
        n += 1
    mean_loss = total / max(n, 1)
    return mean_loss, bits_per_byte(mean_loss)


def main():
    mcfg = ModelConfig()
    tcfg = TrainConfig()

    types = mcfg.layer_types
    print(f"[CONFIG] {mcfg.num_layers} слоёв, gdn2:mla = "
          f"{types.count('gdn2')}:{types.count('mla')} "
          f"(паттерн на блок: {types[:mcfg.layers_per_block]})")

    mesh = build_mesh()
    set_model_mesh(mesh, batch_axis=AXIS_NAME)
    dsh = data_sharding(mesh)

    train_bytes, val_bytes, test_bytes = load_enwik8_splits(
        tcfg.data_dir, tcfg.train_bytes, tcfg.val_bytes, tcfg.test_bytes
    )
    train_loader = ByteSequenceLoader(
        train_bytes, tcfg.seq_len, tcfg.micro_batch_size, data_sharding=dsh,
        shuffle=True, seed=tcfg.seed,
    )
    val_loader = ByteSequenceLoader(
        val_bytes, tcfg.seq_len, tcfg.micro_batch_size, data_sharding=dsh,
        shuffle=False, drop_last=False,
    )

    steps_per_epoch = train_loader.steps_per_epoch // tcfg.accum_steps
    total_steps = max(1, steps_per_epoch * tcfg.max_epochs)
    print(f"[TRAIN] micro-шагов/эпоху={train_loader.steps_per_epoch}, "
          f"эффективных шагов/эпоху={steps_per_epoch}, "
          f"всего эффективных шагов={total_steps} (до {tcfg.max_epochs} эпох)")

    model = HybridByteLM(cfg=mcfg)
    rng = jax.random.PRNGKey(tcfg.seed)

    def _init(r):
        dummy = jnp.zeros((tcfg.micro_batch_size, tcfg.seq_len), dtype=jnp.int32)
        return model.init(r, dummy)["params"]

    abstract_params = jax.eval_shape(_init, rng)
    param_sharding = make_param_sharding(mesh, abstract_params)
    params = jax.jit(_init, out_shardings=param_sharding)(rng)

    n_params = count_params(params)
    print(f"[MODEL] Параметров: {n_params:,} (~{format_count(n_params)})")

    tx, lr_schedule = make_hybrid_optimizer(
        total_steps=total_steps, peak_lr=tcfg.peak_lr, weight_decay=tcfg.weight_decay,
        warmup_ratio=tcfg.warmup_ratio, min_lr_ratio=tcfg.min_lr_ratio,
        grad_clip_norm=tcfg.grad_clip_norm,
    )

    opt_state_abstract = jax.eval_shape(tx.init, abstract_params)
    opt_state_sharding = make_opt_state_sharding(mesh, opt_state_abstract)
    opt_state = jax.jit(tx.init, out_shardings=opt_state_sharding)(params)

    zero_accum = jax.jit(
        lambda p: jax.tree_util.tree_map(jnp.zeros_like, p), out_shardings=param_sharding
    )(params)

    train_micro_step, apply_step = build_train_and_apply(model, tx)
    eval_step = build_eval_step(model)

    ckpt_dir = os.path.join(tcfg.ckpt_dir, "latest")
    best_dir = os.path.join(tcfg.ckpt_dir, "best_val")
    mngr_latest = make_manager(ckpt_dir, max_to_keep=2)
    mngr_best = make_manager(best_dir, max_to_keep=1)

    global_step = 0
    best_val_bpb = float("inf")
    start_epoch = 0
    resume_step = mngr_latest.latest_step()
    if resume_step is not None:
        print(f"[RESUME] Найден чекпоинт на шаге {resume_step}, восстанавливаю...")
        params, opt_state = ckpt_restore(mngr_latest, resume_step, params, opt_state)
        meta = load_metadata(ckpt_dir, resume_step)
        global_step = meta.get("global_step", resume_step)
        best_val_bpb = meta.get("best_val_bpb", float("inf"))
        start_epoch = meta.get("epoch", 0)
        print(f"[RESUME] step={global_step}, epoch={start_epoch}, best_val_bpb={best_val_bpb:.4f}")
    else:
        print("[RESUME] Свежий старт.")

    accum_grads = zero_accum
    micro_in_accum = 0
    t_start = time.perf_counter()

    for epoch in range(start_epoch, tcfg.max_epochs):
        epoch_t0 = time.perf_counter()
        for batch in train_loader.epoch():
            accum_grads, loss, micro_grad_norm = train_micro_step(params, accum_grads, batch)
            micro_in_accum += 1

            if micro_in_accum == tcfg.accum_steps:
                params, opt_state, was_finite, global_norm = apply_step(
                    params, opt_state, accum_grads, tcfg.accum_steps
                )
                accum_grads = zero_accum
                micro_in_accum = 0
                global_step += 1

                if global_step % 50 == 0:
                    lr_now = float(lr_schedule(global_step))
                    loss_v = float(jax.device_get(loss))
                    print(f"[E{epoch}] step={global_step}/{total_steps} "
                          f"loss={loss_v:.4f} bpb~{bits_per_byte(loss_v):.4f} "
                          f"grad_norm={float(jax.device_get(global_norm)):.3f} "
                          f"finite={bool(jax.device_get(was_finite))} lr={lr_now:.2e}")

                if global_step % tcfg.eval_every_steps == 0:
                    val_loss, val_bpb = run_eval(eval_step, params, val_loader, tcfg.eval_max_batches)
                    print(f"[EVAL] step={global_step} val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} "
                          f"(частичный, {tcfg.eval_max_batches} батчей)")
                    if val_bpb < best_val_bpb:
                        best_val_bpb = val_bpb
                        ckpt_save(mngr_best, best_dir, global_step, params, opt_state,
                                  {"global_step": global_step, "epoch": epoch, "best_val_bpb": best_val_bpb})

                if global_step % tcfg.ckpt_every_steps == 0:
                    ckpt_save(mngr_latest, ckpt_dir, global_step, params, opt_state,
                              {"global_step": global_step, "epoch": epoch, "best_val_bpb": best_val_bpb})

        epoch_elapsed = time.perf_counter() - epoch_t0
        val_loss, val_bpb = run_eval(eval_step, params, val_loader, max_batches=None)
        print(f"===> Эпоха {epoch} завершена за {epoch_elapsed / 60:.1f} мин | "
              f"val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} (полный val) <===")

        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb
            ckpt_save(mngr_best, best_dir, global_step, params, opt_state,
                      {"global_step": global_step, "epoch": epoch + 1, "best_val_bpb": best_val_bpb})
        ckpt_save(mngr_latest, ckpt_dir, global_step, params, opt_state,
                  {"global_step": global_step, "epoch": epoch + 1, "best_val_bpb": best_val_bpb})

    total_elapsed = time.perf_counter() - t_start
    print(f"[DONE] Обучение завершено за {total_elapsed / 3600:.2f} ч. "
          f"Лучший val_bpb={best_val_bpb:.4f}")


if __name__ == "__main__":
    main()
