"""
checkpoint_utils.py -- orbax чекпоинтинг (params+opt_state+metadata JSON).

Написан заново для этого проекта -- без HF Hub relay / bucket-логики
старого checkpointing.py (проект тренируется локально/на одном
Kaggle-инстансе по умолчанию). Тот же принцип синхронного save() +
wait_until_finished(), что и в старом проекте: блокирует TPU на время
записи ценой отсутствия гонки "async write vs donate_argnums перезапись
буфера", см. документацию оригинала для контекста этого выбора.
"""
from __future__ import annotations

import os
import json
import time

import orbax.checkpoint as ocp


def make_manager(local_dir: str, max_to_keep: int = 2) -> ocp.CheckpointManager:
    os.makedirs(local_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(local_dir, ocp.StandardCheckpointer(), options)


def save(mngr: ocp.CheckpointManager, local_dir: str, step: int, params, opt_state, meta: dict):
    mngr.save(step, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
    mngr.wait_until_finished()
    step_dir = os.path.join(local_dir, str(step))
    os.makedirs(step_dir, exist_ok=True)
    meta = dict(meta)
    meta["timestamp"] = time.time()
    with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    print(f"[CKPT] Сохранён шаг {step} в {local_dir}")


def restore(mngr: ocp.CheckpointManager, step: int, target_params, target_opt_state):
    raw = mngr.restore(step, args=ocp.args.StandardRestore(
        {"params": target_params, "opt_state": target_opt_state}
    ))
    return raw["params"], raw["opt_state"]


def load_metadata(local_dir: str, step: int) -> dict:
    meta_path = os.path.join(local_dir, str(step), "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}
