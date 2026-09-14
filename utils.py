"""
utils.py -- маленькие независимые от остального проекта хелперы,
общие для optimizer.py / sharding.py / train.py.
"""
from __future__ import annotations


def path_to_str(path) -> str:
    """jax.tree_util key-path (кортеж DictKey/GetAttrKey/SequenceKey/...) ->
    lowercase '/'-строка для substring-меток (label_fn оптимизатора,
    sharding-эвристики). Обходит все варианты атрибутов, т.к. разные типы
    контейнеров pytree (dict/NamedTuple/list) дают разные типы key-объектов."""
    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "name"):
            parts.append(str(p.name))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return "/".join(parts).lower()


def format_count(n: int) -> str:
    """Человекочитаемое форматирование числа параметров: 1234567 -> '1.23M'."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)
