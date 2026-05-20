<!--
  © 2025-2026 MSC Star Team (Esraderey, Raul Cruz Acosta).
  Todos los derechos reservados.
  MZTrain se distribuye bajo licencia MSL-R 1.0 (ver LICENSE).
  Lea AUTHORSHIP.md, NOTICE.md y PRIOR_ART.md antes de cualquier uso.
-->

# MZTrain — Motor de Entrenamiento en Espacio Comprimido v1.0

> Entrenamiento de modelos de IA sin miles de GPUs mediante
> factorización SVD entrenable, rango progresivo, rango bidireccional
> (ElasticRank) y gobernador de memoria (VRAM Governor).

[![License: MSL-R 1.0](https://img.shields.io/badge/License-MSL--R%201.0-red.svg)](LICENSE)
[![Authorship](https://img.shields.io/badge/Authorship-MSC%20Star%20Team-blueviolet.svg)](AUTHORSHIP.md)
[![Prior Art](https://img.shields.io/badge/Prior%20Art-Published-orange.svg)](PRIOR_ART.md)
[![Sealed](https://img.shields.io/badge/SHA--256%20%2B%20SHA--512-Sealed-success.svg)](SEAL.json)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-272%20passing-brightgreen.svg)]()
[![GPU validated](https://img.shields.io/badge/GPU-RTX%204060%20validated-success.svg)](bench/governor_gpu_check.py)

---

## ⚠️ AVISO LEGAL — LEER ANTES DE USAR

MZTrain es **software propietario** distribuido bajo la licencia
**MSL-R 1.0** (MSC Star Team Restricted License), una licencia
**restringida y no de código abierto**. La descarga o el uso
implican la **aceptación íntegra** de los términos del archivo
[`LICENSE`](LICENSE).

| Documento | Qué contiene |
|-----------|--------------|
| [`LICENSE`](LICENSE) | Licencia propietaria restringida MSL-R 1.0. |
| [`AUTHORSHIP.md`](AUTHORSHIP.md) | Declaración formal de autoría y co-titularidad. |
| [`NOTICE.md`](NOTICE.md) | Avisos legales, reserva de derechos, terceros, marcas. |
| [`PRIOR_ART.md`](PRIOR_ART.md) | Publicación defensiva de las invenciones reivindicadas. |
| [`CLA.md`](CLA.md) | Acuerdo de cesión para contribuidores externos. |
| [`TRADEMARK.md`](TRADEMARK.md) | Política de uso de marcas (MZTrain®, ElasticRank™, etc.). |
| [`SECURITY.md`](SECURITY.md) | Política de seguridad y divulgación responsable. |
| [`SEAL.json`](SEAL.json) + [`MANIFEST.sha256`](MANIFEST.sha256) | Sello criptográfico (SHA-256 + SHA-512 + Merkle). |

### Resumen de lo que NO está permitido (sin licencia escrita)

- Uso comercial directo o indirecto, incluida la prestación de
  servicios de entrenamiento a terceros.
- Redistribución, fork público o privado, mirror, paquetes binarios
  o imágenes de contenedor.
- Inclusión del código o documentación en corpus de entrenamiento /
  fine-tuning / destilación / RLHF de modelos de IA de terceros.
- Patentamiento, por sí o por terceros, de las invenciones descritas
  en [`PRIOR_ART.md`](PRIOR_ART.md) o equivalentes obvios.
- Eliminación o modificación de cabeceras de copyright, avisos
  legales o de los archivos de sellado.

Para licencia comercial: **msc.framework@gmail.com**.

### Verificación de integridad

Cualquier copia legítima de MZTrain debe verificarse íntegra:

```bash
python scripts/seal.py verify
```

Si la verificación falla, **la copia no es íntegra y no está
autorizada** para uso.

---

## Principio Fundamental

Si [MNEME](https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica)
puede comprimir un modelo entrenado 93%+, entonces podemos
**ENTRENAR directamente en ese espacio comprimido**.

En vez de entrenar tensores completos `W (m × n)`, MZTrain entrena
sus factores descompuestos `U (m × r)`, `S (r)`, `V (r × n)` con
`r << min(m, n)` — como **parámetros de primer orden**, no como una
compresión a posteriori.

## Ahorro de Memoria

| Componente | Tradicional | MZTrain | Ahorro |
|-----------|------------|---------|--------|
| Pesos | 100% | 20-30% | 70-80% |
| Gradientes | 100% | 20-30% | 70-80% |
| Optimizer (Adam m, v) | 100% | 20-30% | 70-80% |
| Activaciones | 100% | 40-60% | 40-60% |
| **TOTAL (1B params)** | **32 GB VRAM** | **6-10 GB VRAM** | **3-5×** |

> Demostrado en hardware real: ZCodeBERT ~410M params (dense-equiv),
> RTX 4060, pico **3.83 GB**, todos los subsistemas opt-in activos.

## Innovaciones reivindicadas

Las técnicas siguientes son contribuciones originales de MSC Star Team
y están publicadas como prior art en [`PRIOR_ART.md`](PRIOR_ART.md):

1. **Entrenamiento directo en espacio factorizado SVD** (2025-01-15)
2. **Scheduler de rango progresivo** con 5 políticas (2025-01-15)
3. **ZCompressedAdam** — Adam con estados INT8 (2025-01-15)
4. **ZGradientCompressor** con error feedback (2025-01-15)
5. **ZActivationCheckpoint** con MNEME/INT8 (2025-01-15)
6. **ElasticRank v1** — rango bidireccional con sleep bank
   (sleep / revive / prune) y señal híbrida con **AND** (2026-05-15)
7. **ElasticRank v2** — señal de redundancia funcional por
   leverage estadístico del Gram (2026-05-15)
8. **ElasticRank v3 / Loss-Guard** — el delta de pérdida medido es
   el árbitro de la compactación, no los proxies (2026-05-15)
9. **Probe de sensibilidad de pérdida** como diagnóstico permanente
   (2026-05-15)
10. **VRAM Governor MVP** — gating de crecimiento de rango bajo
    presión EMA y recuperación de OOM (GPU-validado, 2026-05-15)
11. **Error feedback topology-aware** para compresión de
    gradientes (2026-05-16)
12. **Reset coordinado de ElasticRank tras refactorización**
    (2026-05-15)

## Instalación

### Desde fuente (única vía autorizada)

```bash
git clone <repositorio-oficial>
cd mztrain
python scripts/seal.py verify   # verifica integridad criptografica
pip install -e .
```

### Con MNEME (recomendado)

```bash
pip install -e ".[mneme]"
```

### Desarrollo

```bash
pip install -e ".[dev]"
```

## Inicio Rápido

### 1. Factorizar un modelo existente

```python
import torch.nn as nn
from mztrain import factorize_existing_model, ZTrainConfig

model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10),
)

z_model, stats = factorize_existing_model(model, rank=32)
print(f"Ahorro total: {stats['total_savings_pct']:.1f}%")
```

### 2. Entrenamiento completo con ZTrainEngine

```python
from mztrain import ZTrainEngine, ZTrainConfig, RankSchedule
import torch.nn.functional as F

config = ZTrainConfig(
    initial_rank=32,
    max_rank=256,
    rank_schedule=RankSchedule.EXPONENTIAL,
    compress_optimizer_states=True,
    use_amp=True,
)

engine = ZTrainEngine(model, config)

def loss_fn(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)

summary = engine.train(
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    epochs=50,
)
```

### 3. Con ElasticRank + Loss-Guard + VRAM Governor (todo opt-in)

```python
config = ZTrainConfig(
    initial_rank=32, max_rank=256, rank_schedule=RankSchedule.COSINE,
    use_elastic_rank=True,                       # rango bidireccional
    elastic_rank_use_redundancy_signal=True,     # v2: redundancia funcional
    elastic_rank_loss_guard_enabled=True,        # v3: loss-guard
    use_vram_governor=True,                      # gobernador de VRAM
    compress_optimizer_states=True,
    use_amp=True,
)
```

### 4. Estimar ahorro de memoria

```python
from mztrain import estimate_memory_savings
savings = estimate_memory_savings(model, rank=64)
print(f"Factor: {savings['savings']['factor']:.1f}x")
```

### 5. Exportar para deployment

```python
full_model = engine.export_full_model()
torch.save(full_model.state_dict(), "model_final.pt")
```

## Arquitectura

```
mztrain/
├── src/mztrain/
│   ├── __init__.py
│   ├── config.py            # ZTrainConfig + enumeraciones
│   ├── layers.py            # ZFactorizedLinear / Attention / TransformerBlock
│   ├── engine.py            # ZTrainEngine (orquestador)
│   ├── optimizer.py         # ZCompressedAdam (INT8)
│   ├── gradient.py          # ZGradientCompressor (TOP_K / 1-bit / INT8 / SVD)
│   ├── scheduler.py         # ZRankScheduler (5 políticas)
│   ├── checkpoint.py        # ZActivationCheckpoint (MNEME/INT8)
│   ├── refactorize.py       # Refactorización SVD periódica
│   ├── elastic_rank.py      # ElasticRank v1/v2/v3 + LossGuard
│   ├── vram_governor.py     # VRAM Governor MVP (GPU-validated)
│   ├── precision.py         # Gestión de precisión
│   └── projector.py         # Proyectores
├── tests/                   # 272 tests
├── bench/                   # Benchmarks (incluye GPU validation)
├── docs/                    # Documentación técnica
├── examples/                # Ejemplos
├── scripts/
│   ├── seal.py              # Sellado criptográfico
│   └── test_zcoder_410m_full.py  # Smoke "todo prendido"
├── LICENSE                  # MSL-R 1.0 (restringida)
├── AUTHORSHIP.md            # Declaración de autoría
├── PRIOR_ART.md             # Publicación defensiva
├── NOTICE.md                # Avisos legales
├── CLA.md                   # Acuerdo de contribución
├── TRADEMARK.md             # Política de marcas
├── SECURITY.md              # Política de seguridad
├── CODEOWNERS               # Revisión obligatoria
├── MANIFEST.sha256          # Sello (formato sha256sum)
└── SEAL.json                # Sello JSON con Merkle root
```

## Componentes principales

### ZFactorizedLinear

```python
from mztrain import ZFactorizedLinear
# W(768,768) = 589.824 params  →  U(768,64)+S(64)+V(64,768) = 98.432 (-83%)
layer = ZFactorizedLinear(768, 768, rank=64)
y = layer(x)  # forward sin reconstruir W: y = ((x·V) * S) · Uᵀ
```

### ZFactorizedAttention / ZFactorizedTransformerBlock

Multi-Head Attention y bloque Transformer completos con
proyecciones Q, K, V, O factorizadas.

### ZCompressedAdam

Adam con estados (m, v) cuantizados a INT8 con escala por tensor y
recompresión periódica.

### ZGradientCompressor

`TOP_K`, `ONE_BIT` (SignSGD escalado), `INT8`, `SVD`, todos con
**error feedback topology-aware** (los buffers se invalidan al
cambiar la forma del parámetro).

### ElasticRank (v1 + v2 + v3 / Loss-Guard)

```python
config = ZTrainConfig(
    use_elastic_rank=True,
    elastic_rank_use_redundancy_signal=True,
    elastic_rank_loss_guard_enabled=True,
)
```

- **v1**: señal híbrida (espectral ∧ update) con AND, sleep bank en
  CPU + baja precisión, revive con momentum intacto, podado por
  edad de sueño.
- **v2**: redundancia funcional por leverage estadístico del Gram
  unitario (`coherence_i ∈ [0, 1)`).
- **v3 / Loss-Guard**: el árbitro de la decisión es el delta de
  pérdida medido en un probe batch; rollback estructural completo
  si el delta supera umbral.

> Honestidad técnica: ElasticRank rinde donde **existe** redundancia
> explotable (fine-tune sobre checkpoints aproximadamente
> low-rank). En pretraining desde cero, los proxies no predicen el
> impacto real (Spearman ≈ 0.13), por lo que ElasticRank queda
> deliberadamente **inerte** y el loss-guard garantiza
> safe-by-construction. Detalle en [`PRIOR_ART.md`](PRIOR_ART.md).

### VRAM Governor

```python
config = ZTrainConfig(use_vram_governor=True)
```

Sensor de memoria CUDA-guarded → presión EMA → 3 modos con
histéresis → veto del crecimiento de rango bajo presión →
recuperación de OOM transitorio con un único reintento.
**GPU-validado en RTX 4060 (15/15 PASS)**.

### ZRankScheduler

5 políticas: `CONSTANT`, `LINEAR`, `EXPONENTIAL`, `COSINE`, `ADAPTIVE`.

### ZActivationCheckpoint

```python
from mztrain import z_checkpoint
out = z_checkpoint(lambda x: model.layer(x), input_tensor)
```

## Benchmarks publicados (reproducibles)

### MLP (784→512→256→10)

| Métrica | Tradicional | MZTrain r=32 | MZTrain r=64 |
|---------|------------|--------------|--------------|
| Params | 535.818 | 117.514 | 198.666 |
| Memoria | 6,1 MB | 1,8 MB | 2,7 MB |
| Compresión | 1,0× | 4,6× | 2,7× |

### Transformer (embed=512, heads=8, layers=6)

| Métrica | Tradicional | MZTrain r=32 | MZTrain r=64 |
|---------|------------|--------------|--------------|
| Params/bloque | ~3,1 M | ~0,6 M | ~1,0 M |
| VRAM (batch=32) | ~1,2 GB | ~0,3 GB | ~0,5 GB |
| Compresión | 1,0× | 5,2× | 3,1× |

### Smoke full-stack — ZCodeBERT ~410M

`scripts/test_zcoder_410m_full.py`, RTX 4060: pico **3,83 GB**,
~3,7 min, exit 0. ElasticRank inerte (consistente con la teoría),
VRAM Governor en `normal` (0 bloqueos, 2 grows aprobados), grad
compression con `error_buffer_resets` correcto tras rank growth.

## Desarrollo

```bash
pytest tests/ -v --no-cov          # juzgar por pass/fail (272 tests)
black src/ tests/                  # formato
isort src/ tests/
ruff check src/mztrain/
mypy src/mztrain/                  # type checking
python scripts/seal.py verify      # verificar sello criptográfico
```

> El gate de cobertura `--cov-fail-under=85` está pre-existentemente
> por debajo (módulos sin test específico): juzgar regresiones por
> conteo de tests con `--no-cov`.

## Hoja de ruta (no implementado, propiedad reservada)

Las siguientes propuestas están descritas en [`PRIOR_ART.md`](PRIOR_ART.md)
únicamente como ideas reservadas — **NO** están implementadas y por
tanto **NO** son aval para reivindicación funcional, pero sí cuentan
como publicación defensiva:

- Soporte distribuido (DDP / FSDP), gradient accumulation, BF16,
  WandB/TensorBoard.
- Kernel CUDA para forward factorizado.
- LoRA/QLoRA, QAT, auto-tuning de rango óptimo.
- Extensión de ElasticRank a capas sparse (`ZSparseFactorizedLinear`).
- VRAM Governor v2: precision autopilot, byte-level rank budget,
  modo QUALITY / DEFENSIVE, acoplamiento con ElasticRank.

## Contribuir

Las contribuciones externas se aceptan **solo** bajo los términos del
[`CLA.md`](CLA.md). Ver también [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licencia

Este proyecto se licencia bajo **MSL-R 1.0** — ver [`LICENSE`](LICENSE)
para el texto íntegro y la lista de usos prohibidos.

## Autoría

- **Esraderey** — co-titular, arquitecto principal y co-inventor.
- **Raúl Cruz Acosta** — co-titular y co-inventor.

Detalle formal en [`AUTHORSHIP.md`](AUTHORSHIP.md).

## Relacionados

- [MNEME — Motor de Memoria Neural Mórfica](https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica)
  (compresión de modelos entrenados; premisa que originó MZTrain).

---

© 2025-2026 MSC Star Team. **Todos los derechos reservados.**
Distribuido bajo **MSL-R 1.0**. Verifique siempre la integridad con
`python scripts/seal.py verify` antes de usar.
