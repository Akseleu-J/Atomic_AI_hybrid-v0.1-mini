# gdn2-mla-enwik8

Гибридная GDN-2 / MLA языковая модель (byte-level, ~250M параметров),
обучение на enwik8 до 5 эпох, метрика — **bits-per-byte (bpb)**.

Написано **с нуля** для этого проекта — не переиспользует существующий
`atomic_ops`/`train.py`/`model.py`/`optimizer.py` стек. Общая архитектура
подхода (шардинг, чекпоинтинг, hybrid-оптимизатор, gradient accumulation,
skip-on-nonfinite) сохранена в духе прежнего проекта, но реализована
заново и проще.

## Архитектура

- **Соотношение gdn2:mla = 4:1**, держится ровно на уровне каждого блока
  (`layers_per_block=5` → 4×GDN-2 + 1×MLA), не только "в среднем" по сети.
  См. `configs.ModelConfig.layer_types`.
- **Delta Attention Residual (DAR)**: вместо наивного `x = x + delta`
  каждый слой перед своей подсетью делает softmax-retrieval attention по
  накопленным дельтам — своим внутри блока и дельтам всех прошлых блоков
  целиком (`model.DeltaAttentionResidual`). Слой сам "решает", какая
  прошлая дельта сейчас релевантна на каждой позиции.
- **GDN-2**: собственная чистая-JAX реализация gated delta-rule —
  `jax.lax.scan` по chunk'ам (remat-граница) → `jax.lax.scan` по токенам
  внутри chunk'а (`attention.gdn2_chunked_delta_rule`). Не Pallas-кернел —
  сознательный выбор для версии "с нуля": корректность важнее пиковой
  скорости на этом этапе; сигнатура функции не завязана на реализацию,
  внутренности можно заменить на Pallas позже.
- **MLA**: `jax.experimental.pallas.ops.tpu.flash_attention` (библиотечное
  TPU flash-attention ядро из самого JAX) с graceful fallback на обычный
  softmax-attention при недоступности (CPU/GPU отладка).
- Один SwiGLU FFN на блок (не per-layer, не MoE) — предсказуемый бюджет
  параметров без роутинга.

## Файлы

| Файл | Назначение |
|---|---|
| `configs.py` | `ModelConfig` (архитектура), `TrainConfig` (гиперпараметры обучения) |
| `data_enwik8.py` | скачивание/сплит enwik8, byte-level `ByteSequenceLoader`, `bits_per_byte()` |
| `attention.py` | GDN-2 (chunked scan) и MLA (flash-attention) ядра |
| `model.py` | DAR-блоки, полная модель `HybridByteLM`, `count_params()` |
| `optimizer.py` | Muon+AdamW гибрид, LR schedule, byte-level CE loss |
| `sharding.py` | mesh + FSDP-подобный шардинг параметров/opt_state/батча |
| `checkpoint_utils.py` | orbax save/restore + JSON-metadata |
| `utils.py` | `path_to_str` (метки параметров), `format_count` |
| `train.py` | основной цикл: до 5 эпох, periodic/epoch-end eval, checkpointing |

## Запуск

```bash
pip install -r requirements.txt
python train.py
```

enwik8 скачается автоматически при первом запуске (`data_enwik8.ensure_enwik8`,
~36MB zip) в `TrainConfig.data_dir`. Чекпоинты пишутся в `TrainConfig.ckpt_dir`
(`latest/` — resume, `best_val/` — лучший `val_bpb`); `train.py` подхватывает
`latest/` автоматически при повторном запуске.

## Как пересчитать под свой бюджет параметров

`train.py` печатает точное число параметров при старте
(`count_params(params)`). Основные рычаги в `ModelConfig`:

- `d_model` / `n_heads` — магистраль (GDN-2 даёт ~8 Dense d×d на слой,
  MLA — компактнее за счёт `d_latent`)
- `d_ff` — размер SwiGLU FFN (один на блок)
- `num_blocks` (при фиксированном `layers_per_block=5`) — количество
  DAR-блоков, шаг в 5 слоёв (4 gdn2 + 1 mla)

Дефолты (`d_model=1024, n_heads=16, d_latent=512, d_ff=4096, num_blocks=5`)
дают архитектуру в районе 250M параметров — уточняйте по фактическому
принту при старте и подстраивайте `d_model`/`num_blocks`.

## Известные упрощения этой версии

- GDN-2 — честный `O(L)` scan, не Pallas chunked-WY кернел (см. выше) —
  на TPU будет медленнее, чем оптимизированные Pallas-реализации.
- Нет MoE/роутинга, нет W&B/HF Hub релея, нет диагностики уровня
  "какой физический слой/оптимизаторская группа стали non-finite" —
  есть только глобальный `global_norm`/`is_finite` skip-on-nonfinite.
- Dropout не подключён (`dropout_rate` в конфиге зарезервирован на
  будущее, `deterministic=True` жёстко зашит в `train.py`/`optimizer.py`).
