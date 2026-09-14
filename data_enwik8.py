"""
data_enwik8.py -- byte-level загрузчик enwik8 (Hutter Prize) для
GDN-2/MLA гибридной модели.

Модель работает на уровне байтов (vocab_size=256), поэтому bits-per-byte
(bpb) = средний CE-loss в натах / ln(2) -- БЕЗ дополнительной коррекции на
длину токена (в отличие от BPE-моделей, где нужно домножать на
tokens/bytes) -- один токен здесь == один байт.

Сплит: стандартный для enwik8 в литературе -- первые 90_000_000 байт
train, следующие 5_000_000 val, последние 5_000_000 test.
"""
from __future__ import annotations

import os
import math
import urllib.request
import zipfile

import numpy as np
import jax
import jax.numpy as jnp

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
LN2 = math.log(2.0)


def ensure_enwik8(data_dir: str) -> str:
    """Скачивает+распаковывает enwik8, если ещё не на диске. Возвращает
    путь к сырому файлу (100_000_000 байт)."""
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "enwik8")
    zip_path = os.path.join(data_dir, "enwik8.zip")
    if os.path.exists(raw_path):
        return raw_path
    if not os.path.exists(zip_path):
        print(f"[DATA] Скачиваю enwik8 из {ENWIK8_URL} ...")
        urllib.request.urlretrieve(ENWIK8_URL, zip_path)
    print("[DATA] Распаковываю enwik8.zip ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"После распаковки enwik8.zip не найден ожидаемый файл {raw_path} -- "
            f"проверьте содержимое {data_dir}."
        )
    return raw_path


def load_enwik8_splits(data_dir: str, train_bytes: int, val_bytes: int, test_bytes: int):
    """Возвращает (train, val, test) как np.uint8 массивы."""
    path = ensure_enwik8(data_dir)
    with open(path, "rb") as f:
        data = f.read()
    total = len(data)
    needed = train_bytes + val_bytes + test_bytes
    if needed > total:
        raise ValueError(
            f"Запрошено train+val+test={needed:,} байт, но enwik8 содержит только "
            f"{total:,} -- уменьшите train_bytes/val_bytes/test_bytes в TrainConfig."
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    train = arr[:train_bytes]
    val = arr[train_bytes:train_bytes + val_bytes]
    test = arr[train_bytes + val_bytes:train_bytes + val_bytes + test_bytes]
    print(f"[DATA] enwik8: train={len(train):,} val={len(val):,} test={len(test):,} байт")
    return train, val, test


def bits_per_byte(mean_ce_nats: float) -> float:
    """CE loss (наты, усреднённый по валидным байтам) -> bits per byte."""
    return mean_ce_nats / LN2


class ByteSequenceLoader:
    """Неперекрывающиеся окна длины seq_len+1 (вход = [:-1], метка = [1:]).
    Порядок чанков перемешивается между эпохами (не отдельные байты --
    сохраняет локальную структуру текста внутри одного окна)."""

    def __init__(self, byte_arr: np.ndarray, seq_len: int, batch_size: int,
                 data_sharding=None, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = True):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data_sharding = data_sharding
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.drop_last = drop_last

        n_chunks = (len(byte_arr) - 1) // seq_len
        if n_chunks < batch_size:
            raise ValueError(
                f"Датасет слишком мал для batch_size={batch_size} при seq_len={seq_len} "
                f"(доступно только {n_chunks} неперекрывающихся окон)."
            )
        usable = n_chunks * seq_len + 1
        self._flat = byte_arr[:usable]
        self.n_chunks = n_chunks
        self.steps_per_epoch = (
            n_chunks // batch_size if drop_last else math.ceil(n_chunks / batch_size)
        )

    def _chunk(self, idx: int):
        start = idx * self.seq_len
        window = self._flat[start:start + self.seq_len + 1].astype(np.int32)
        return window[:-1], window[1:]

    def _make_batch(self, idxs):
        ids_batch = np.empty((self.batch_size, self.seq_len), dtype=np.int32)
        lbl_batch = np.empty((self.batch_size, self.seq_len), dtype=np.int32)
        for i, idx in enumerate(idxs):
            ids_batch[i], lbl_batch[i] = self._chunk(int(idx))
        batch = {
            "input_ids": jnp.asarray(ids_batch),
            "labels": jnp.asarray(lbl_batch),
        }
        if self.data_sharding is not None:
            batch = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), batch
            )
        return batch

    def epoch(self):
        """Один проход по train-данным, с перемешиванием порядка чанков."""
        order = np.arange(self.n_chunks)
        if self.shuffle:
            self.rng.shuffle(order)
        for b in range(self.steps_per_epoch):
            idxs = order[b * self.batch_size:(b + 1) * self.batch_size]
            if len(idxs) < self.batch_size:
                if self.drop_last:
                    break
                pad = self.batch_size - len(idxs)
                idxs = np.concatenate([idxs, order[:pad]])
            yield self._make_batch(idxs)

    def eval_epoch(self, max_batches=None):
        """Детерминированный (без shuffle) проход -- для val/test."""
        n_batches = self.steps_per_epoch if max_batches is None else min(self.steps_per_epoch, max_batches)
        order = np.arange(self.n_chunks)
        for b in range(n_batches):
            idxs = order[b * self.batch_size:(b + 1) * self.batch_size]
            if len(idxs) < self.batch_size:
                break
            yield self._make_batch(idxs)
