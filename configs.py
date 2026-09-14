"""
configs.py -- центральная конфигурация модели/обучения/шардинга.

Новый проект с нуля (не использует существующий atomic_ops/train.py/...).
Архитектура: DAR-блоки (Delta Attention Residual) из 5 слоёв
(4x GDN-2 + 1x MLA, соотношение 4:1). MLA использует TPU flash-attention
ядро (jax.experimental.pallas.ops.tpu.flash_attention, библиотечное, не
собственное). GDN-2 использует собственную Pallas WY-chunked реализацию
из atomic_ops (gdn2_forward_trainable) -- production kernel library,
а не reference chunked-scan.

Параметры по умолчанию подобраны под ~250M параметров (см. README.md,
раздел "как пересчитать под свой бюджет параметров"; count_params()
в model.py печатает точное число при старте train.py).
"""
from __future__ import annotations

import dataclasses as dc
from typing import Tuple


@dc.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256          # byte-level (enwik8) -- один байт == один токен
    d_model: int = 1024
    n_heads: int = 16              # используется и GDN-2, и MLA (d_head = d_model // n_heads)
    d_latent: int = 512            # латент MLA + латент DAR retrieval-attention
    d_ff: int = 4096               # SwiGLU FFN hidden (один FFN на блок, не per-layer)
    d_conv: int = 4                # ширина short causal conv (только GDN-2, q/k/v)

    layers_per_block: int = 5      # 4x gdn2 + 1x mla -- см. layer_types ниже
    num_blocks: int = 5            # -> 25 слоёв: 20 gdn2 + 5 mla (ratio ровно 4:1)

    dropout_rate: float = 0.0      # не используется в этой версии (детерминированный forward)
    label_smoothing: float = 0.0

    gdn2_chunk_size: int = 128     # config.bt для atomic_ops Pallas-кернеля (chunk/scan-граница)
    gdn2_d_head: int = 128         # ЖЁСТКОЕ требование atomic_ops.configs.validate_inputs:
                                    # d_head должен быть ровно 128 (MXU tile). Развязан от
                                    # cfg.d_head (у MLA d_head = d_model // n_heads = 64),
                                    # иначе gdn2_forward_trainable упадёт с ValueError на TPU.
                                    # Из-за этого GDN2-проекции (q/k/v/erase/write/out_gate)
                                    # используют H*gdn2_d_head, а не H*d_head -- параметров
                                    # у GDN2-слоя станет больше, чем при чистом-JAX варианте.
    rope_theta: float = 10000.0    # только для MLA (GDN-2 не использует RoPE)

    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model={self.d_model} должен делиться на n_heads={self.n_heads}"
        return self.d_model // self.n_heads

    @property
    def num_layers(self) -> int:
        return self.layers_per_block * self.num_blocks

    @property
    def layer_types(self) -> Tuple[str, ...]:
        """4 gdn2 : 1 mla, повторяется num_blocks раз -- соотношение 4:1
        держится РОВНО на уровне каждого блока, не только в среднем."""
        assert self.layers_per_block >= 2, "нужен хотя бы 1 gdn2 + 1 mla на блок"
        base = ("gdn2",) * (self.layers_per_block - 1) + ("mla",)
        return base * self.num_blocks


@dc.dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 1024
    micro_batch_size: int = 16
    accum_steps: int = 2           # эффективный batch = micro_batch_size * accum_steps
    max_epochs: int = 5

    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1      # доля от peak_lr в конце cosine-декея
    warmup_ratio: float = 0.03     # доля total_steps на линейный warmup
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0

    eval_every_steps: int = 500    # частичная val-оценка (eval_max_batches батчей)
    eval_max_batches: int = 100
    ckpt_every_steps: int = 1000

    seed: int = 0

    data_dir: str = "./data"
    ckpt_dir: str = "./checkpoints"

    # стандартный enwik8-сплит (байты): 90M train / 5M val / 5M test --
    # общепринятое разбиение в литературе по LM на enwik8 (Hutter Prize),
    # позволяет сравнивать bpb с опубликованными результатами.
    train_bytes: int = 90_000_000
    val_bytes: int = 5_000_000
    test_bytes: int = 5_000_000
