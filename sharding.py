"""
sharding.py -- mesh + шардинг параметров/opt_state/батча.

Один mesh axis ("data"), используемый и для FSDP-шардинга параметров
(первая достаточно большая ось каждого веса), и для шардинга батча по
устройствам -- тот же паттерн, что в оригинальном проекте (один
'tpu_nodes' axis на всё), просто написан заново.
"""
from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

AXIS_NAME = "data"


def build_mesh() -> Mesh:
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=(AXIS_NAME,))
    print(f"[SHARD] Mesh: {mesh.shape[AXIS_NAME]} устройств на оси '{AXIS_NAME}'.")
    return mesh


def data_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P(AXIS_NAME, None))


def _param_spec_for_shape(shape, n_devices: int):
    """FSDP-эвристика: шардим первую ось параметра по mesh-оси 'data',
    если она делится на число устройств и достаточно велика, чтобы это
    было выгодно; иначе -- реплицируем (дёшево по памяти для мелких
    векторов вроде норм-гейнов/биасов/decay_a)."""
    if len(shape) == 0:
        return P()
    if len(shape) == 1:
        return P() if shape[0] < 4 * n_devices else P(AXIS_NAME)
    if shape[0] % n_devices == 0 and shape[0] >= n_devices:
        return P(AXIS_NAME, *([None] * (len(shape) - 1)))
    return P(*([None] * len(shape)))


def make_param_sharding(mesh: Mesh, params):
    n_devices = mesh.shape[AXIS_NAME]

    def _spec(x):
        return NamedSharding(mesh, _param_spec_for_shape(x.shape, n_devices))
    return jax.tree_util.tree_map(_spec, params)


def make_opt_state_sharding(mesh: Mesh, opt_state):
    n_devices = mesh.shape[AXIS_NAME]

    def _spec(x):
        if hasattr(x, "shape"):
            return NamedSharding(mesh, _param_spec_for_shape(x.shape, n_devices))
        return NamedSharding(mesh, P())

    return jax.tree_util.tree_map(
        _spec, opt_state, is_leaf=lambda x: hasattr(x, "shape")
    )
