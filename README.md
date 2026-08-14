<!--
  © 2025-2026 MSC Star Team (Esraderey, Raul Cruz Acosta).
  Distribuido bajo licencia MIT (ver LICENSE).
-->

# MZTrain — Motor de Entrenamiento en Espacio Comprimido v1.3

> Entrena modelos que **no caben densos** en tu GPU, mediante factorización SVD
> entrenable, rango progresivo/bidireccional (ElasticRank) y gobernador de VRAM —
> con su costo de calidad **medido, preregistrado y publicado** en este mismo repo.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Authorship](https://img.shields.io/badge/Authorship-MSC%20Star%20Team-blueviolet.svg)](AUTHORSHIP.md)
[![Prior Art](https://img.shields.io/badge/Prior%20Art-Published-orange.svg)](PRIOR_ART.md)
[![Sealed](https://img.shields.io/badge/SHA--256%2B512%2BEd25519-Sealed-success.svg)](SEAL.json)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-347%20passing-brightgreen.svg)]()
[![Evidencia](https://img.shields.io/badge/benchmarks-preregistrados-informational.svg)](docs/evidencia/)

---

## La verdad en tres líneas (léela antes de nada)

1. **Si tu modelo CABE denso en tu GPU: entrena denso.** A igualdad de parámetros, un
   denso más pequeño gana al factorizado en calidad (~10% de BPC) y a menudo también en
   VRAM y velocidad. Medido a 3 escalas, robusto a LR, schedule de rango y ubicación de
   la factorización ([T4](docs/evidencia/T4-VEREDICTO.md), [T6](docs/evidencia/T6-VEREDICTO.md)).
2. **Si NO cabe: MZTrain es la puerta.** El ahorro de memoria es real y crece con la
   escala (pesos+gradientes+Adam ∝ params); medido: un equivalente denso de **1.06B
   params entrena a 6 361 tok/s con 7.2 GB de pico** en una RTX 4060 (bf16, batch 8).
3. El framework es **matemáticamente correcto y estable**: el forward factorizado iguala
   la SVD ideal a 4 cifras sobre un modelo real de 97 capas, y la serie empírica completa
   corrió sin un solo NaN. Auditado (peritaje + peritaje matemático + criba), reparado y
   sellado.

## Licencia y avisos

MZTrain es **software libre** bajo licencia **MIT** (ver [`LICENSE`](LICENSE)):
puedes usarlo, modificarlo, redistribuirlo e incorporarlo en proyectos comerciales,
conservando el aviso de copyright y el texto de la licencia.

| Documento | Qué contiene |
|-----------|--------------|
| [`LICENSE`](LICENSE) | Licencia MIT. |
| [`AUTHORSHIP.md`](AUTHORSHIP.md) | Declaración de autoría y co-titularidad. |
| [`NOTICE.md`](NOTICE.md) | Atribuciones, terceros y marcas. |
| [`PRIOR_ART.md`](PRIOR_ART.md) | Publicación defensiva de las invenciones (prior art). |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) + [`CLA.md`](CLA.md) | Cómo contribuir (inbound = outbound, DCO). |
| [`TRADEMARK.md`](TRADEMARK.md) | Política de marcas. |
| [`SECURITY.md`](SECURITY.md) | Política de seguridad y divulgación responsable. |
| [`SEAL.json`](SEAL.json) + [`MANIFEST.sha256`](MANIFEST.sha256) | Sello de integridad (SHA-256 + SHA-512 + Merkle + firma Ed25519). |

**El objeto canónico de verificación es el repositorio git** (o cualquier clon — los bytes
viajan exactos gracias a `.gitattributes`):

```bash
git clone https://github.com/esraderey/Mztrain.git && cd Mztrain && python scripts/seal.py verify
```

Debe reportar `OK` con firma Ed25519 válida. **Nota honesta:** el sdist de PyPI empaqueta un
*subconjunto* de instalación (sin `.github/`, `bench/`, JSONs crudos de evidencia…); `verify`
dentro del sdist reportará faltantes — no es manipulación; verifica contra el repo.

## Principio fundamental

En vez de entrenar tensores completos `W (m × n)`, MZTrain entrena sus factores
`U (m × r)`, `S (r)`, `V (r × n)` con `r << min(m, n)` — como **parámetros de primer
orden**, no como compresión a posteriori. Los gradientes y los estados de Adam viven
en el espacio factorizado: la memoria de entrenamiento entera se encoge con `r`.

## Qué está medido (serie empírica T0–T7, preregistrada)

Todos los experimentos de esta sección tienen **preregistro previo a los datos**
(hipótesis, métrica y condición de falsación fijadas antes de correr), artefactos JSON
crudos y veredicto formal, en [`docs/evidencia/`](docs/evidencia/). Hardware: RTX 4060
(8.6 GB), Windows 11, torch 2.11+cu128.

### Memoria y capacidad (T0, T5)

| Config | Estado | Dato |
|---|---|---|
| Pythia-410M fine-tune denso | al límite | 8.3 GB |
| Pythia-410M-equiv factorizado (engine completo) | holgado | **2.2 GB** (3.9×) |
| Techo práctico denso (fp32, batch 8) | ~280M | 5.9 GB, 3 452 tok/s |
| Techo práctico factorizado (fp32) | ~570M-equiv | 6.0 GB, **5 333 tok/s** |
| **Hallazgo Windows/WDDM**: no hay OOM — hay un **acantilado silencioso** ~7 GB donde la paginación a RAM colapsa el throughput ~30×. Diseña para quedarte debajo. | | [t5_capacity.json](docs/evidencia/t5_capacity.json) |

### Velocidad con bf16 (T6) — el acantilado se mueve

| Config (autocast bf16, batch 8) | tok/s | VRAM pico |
|---|---|---|
| Denso 277M-equiv | 7 780 | 5.7 GB |
| Factorizado 850M-equiv | 6 303 | 6.0 GB |
| **Factorizado 1.06B-equiv** | **6 361** | **7.2 GB** |

Con fp32 ese 1.06B-equiv rendía 159 tok/s (zona muerta). bf16 reduce activaciones →
el pico cae bajo el acantilado → **40×**. 1B de tokens ≈ 1.8 días en una 4060.
Paridad de calidad bf16 verificada en denso (dos escalas) y factorizado pequeño;
**pendiente** en factorizado grande (ver Limitaciones).

### El peaje de calidad (T4, T6, T7) — léelo antes de reivindicar nada

- Gap iso-parámetro (factorizado vs denso de iguales params, mismo presupuesto):
  **~10% de BPC, estable de 11M a 152M** (H1 "se cierra con la escala" falsada por regla
  preregistrada).
- El gap **no** es un artefacto: tunear el LR lo ensancha (el denso aprovecha LRs altos
  mejor), el annealing de rango (192→64 con cirugía completa) no aporta nada detectable,
  y factorizar solo el FFN conserva el 85% del gap. Es el costo de capacidad del cuello
  de rango.
- La dinámica fue interrogada (T7): sin muerte de direcciones (rango efectivo ≈ nominal),
  deriva de gauge real pero benigna, y el weight decay **estabiliza** al factorizado
  (nunca lo pongas en 0).

### Compresión de un modelo YA entrenado: usa bits, no rango

Sobre Pythia-410M real: la truncación SVD *matemáticamente óptima* conservando el 50%
de los params destruye el modelo (PPL ×~8 000), mientras que la cuantización calibrada
(GPTQ INT4 de [MNEME](https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica))
cuesta +35% de PPL con 4× menos memoria. **Entrena por rango, despliega por bits.**
Para factorizar un preentrenado con fines de análisis usa
`factorize_existing_model(preserve_map=True)` — la variante fiel a la SVD (verificada a
4 cifras); el default con EPSI es para inicializar desde cero, no para transferir.

## ElasticShape (nuevo en 1.3): entrena chico, crece a mitad de camino

La medición que abrió la puerta: a presupuesto corto, un denso pequeño aprende más calidad
por segundo que cualquier modelo grande ([T4/T6](docs/evidencia/)). ElasticShape lo convierte
en mecanismo: **entrena un denso pequeño, conviértelo EXACTO a factorizado (SVD completa),
ensancha/profundiza con cirugía que preserva la función, y sigue entrenando grande** — con la
migración completa del estado Adam, corrección de escala de atención y compensación de
varianza de LayerNorm.

**Resultado T8 (preregistrado, [veredicto](docs/evidencia/T8-VEREDICTO.md)):** el morph
alcanza la calidad del from-scratch en el **54% del reloj** (3/3 seeds, ±0.4 s) y, a tiempo
igual, rinde **2.34-2.37 BPC vs 2.90** del from-scratch. Deriva de la cirugía sobre modelos
entrenados: 0.24-1.4%. *Alcance: escala 11M-equiv, un schedule, fp32 — sin probar aún a
escala mayor.*

```python
from mztrain import GrowthEvent, LrWarmup, apply_event
from mztrain.elastic_shape import GPT, dense_lin

model = GPT(vocab, seq, d=192, layers=6, heads=6, lin=dense_lin)
# ... entrenar pequeño ...
ev = GrowthEvent(step=2000, factorize=True, new_d=384)
opt, reporte = apply_event(model, opt, ev, probe_x=probe)  # cirugia completa
warmup = LrWarmup(opt, steps=200)                          # y a seguir entrenando
```

Primitivas de bajo nivel en `mztrain.shape_ops` (ensanchar capas/embeddings/LN con mapas de
índices, denso→factorizado exacto, bloques identidad, pad de estado Adam). Diseño y
matemática de preservación: [docs/SPEC-elasticshape-v1.md](docs/SPEC-elasticshape-v1.md).

## Cuándo usar MZTrain (y cuándo no)

| Tu caso | Recomendación |
|---|---|
| El modelo cabe denso en tu GPU | Denso (mejor calidad por parámetro; con bf16 además rápido) |
| El modelo NO cabe denso | MZTrain factorizado — asumiendo el ~10% de peaje medido |
| Comprimir un checkpoint entrenado | Cuantización (MNEME/GPTQ), no factorización |
| Investigación de rank scheduling | ZTrainEngine + ElasticRank (maquinaria auditada) |

## Ahorro de memoria (por construcción; el total emerge con la escala)

| Componente | Tradicional | MZTrain | Ahorro |
|-----------|------------|---------|--------|
| Pesos | 100% | 20-30% | 70-80% |
| Gradientes | 100% | 20-30% | 70-80% |
| Optimizer (Adam m, v) | 100% | 20-30% | 70-80% |
| Activaciones* | 100% | 40-100% | 0-60% |

\* La factorización **no** reduce activaciones (conserva el ancho); el ahorro de esa fila
proviene de `ZActivationCheckpoint` (INT8/MNEME), opcional e independiente. Por eso el
ahorro TOTAL depende de la escala: nulo bajo ~50M (dominan activaciones), 3.9× medido a
410M, y creciente de ahí en adelante.

## Limitaciones conocidas y configuraciones a evitar

- **No esperes ganar a un denso iso-parámetro en calidad.** Está falsado a 3 escalas con
  preregistro. El valor del framework es habilitación, no eficiencia por parámetro.
- **`RANDOM_K` + error-feedback: NO usar** (diverge geométricamente; hallazgo del
  peritaje matemático, sin reparar). Métodos seguros: `TOP_K`, `ONE_BIT`, `INT8`, `SVD`.
- **`ZGaLoreOptimizer`: NO usar** (la cuantización INT8 lineal colapsa `v→0`; sin reparar).
- **Params en fp16 puro con optimizer comprimido: NO** (underflow del log-quant). Usa
  fp32 + autocast (bf16), el patrón validado.
- **bf16 + factorizado GRANDE: valida con seeds antes de confiar.** Un run a 57M mostró
  +0.44 BPC vs fp32 (1 seed, sin resolver); en denso y factorizado pequeño la paridad
  bf16 está verificada.
- **fp8: experimental.** El path corre en SM89 pero multiplica S en bf16 (pendiente de
  arreglo) y `forward_context` usa fp16 en su rama fp8. No validado end-to-end.
- **weight_decay nunca en 0 con capas factorizadas** (triplica el ruido entre seeds y
  empeora la media; T7).
- **Reporta ≥3 seeds** en cualquier comparación con factorizado: es la condición más
  ruidosa entre seeds (~×10 vs denso; causa abierta, sospecha de paisaje de pérdida).
- El sello detecta manipulación **del árbol sellado**; binarios y rutas excluidas quedan
  fuera (ver `scripts/seal.py`).

## Instalación

```bash
pip install mztrain            # desde PyPI
```

```bash
git clone https://github.com/esraderey/Mztrain.git
cd Mztrain
python scripts/seal.py verify  # verifica integridad criptografica
pip install -e .
```

### MNEME (backend opcional de almacenamiento/cuantización)

MNEME **no se distribuye por PyPI**; instálalo manualmente desde
[su repositorio](https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica)
si quieres el store de activaciones ZSpace o el despliegue GPTQ INT4. Sin MNEME,
MZTrain usa su fallback INT8 integrado (acotado y verificado).

## Inicio rápido

### 1. Entrenamiento completo con ZTrainEngine

```python
from mztrain import ZTrainEngine, ZTrainConfig, RankSchedule
import torch.nn.functional as F

config = ZTrainConfig(
    initial_rank=32,
    max_rank=256,
    rank_schedule=RankSchedule.EXPONENTIAL,
    compress_optimizer_states=True,
    use_amp=True,                      # bf16 en hardware Ampere+
)
engine = ZTrainEngine(model, config)

def loss_fn(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)

summary = engine.train(train_loader=train_loader, val_loader=val_loader,
                       loss_fn=loss_fn, epochs=50)
```

### 2. ElasticRank + Loss-Guard + VRAM Governor (opt-in)

```python
config = ZTrainConfig(
    initial_rank=32, max_rank=256, rank_schedule=RankSchedule.COSINE,
    use_elastic_rank=True,
    elastic_rank_use_redundancy_signal=True,
    elastic_rank_loss_guard_enabled=True,
    use_vram_governor=True,
    compress_optimizer_states=True,
    use_amp=True,
)
```

### 3. Factorizar un modelo existente (análisis / punto de partida)

```python
from mztrain import factorize_existing_model
z_model, stats = factorize_existing_model(model, rank=32, preserve_map=True)
```

### 4. Exportar para deployment

```python
full_model = engine.export_full_model()
torch.save(full_model.state_dict(), "model_final.pt")
# despliegue comprimido: cuantiza con MNEME/GPTQ (bits), no truncando rango
```

## Componentes

| Módulo | Qué hace | Estado de auditoría |
|---|---|---|
| `ZFactorizedLinear` / Attention / TransformerBlock | forward factorizado `((x·Vᵀ)·S)·Uᵀ` | Exacto vs SVD ideal (4 cifras, modelo real) |
| `ZCompressedAdam` | Adam con estados INT8 (log-quant en v) | Verificado = AdamW canónico; seguro en bf16 |
| `ZGradientCompressor` | TOP_K / 1-bit / INT8 / SVD + error feedback topology-aware | Contractivos verificados (evitar RANDOM_K) |
| `ZRankScheduler` | 5 políticas de crecimiento | Correctas (solo crecen; no hay schedules descendentes) |
| `ElasticRank` v1-v3 | rango bidireccional, sleep bank, loss-guard | Cirugía auditada; S siempre fp32 en el bank |
| `VRAM Governor` | presión EMA, histéresis, `oom_guarded` en el train loop | GPU-validado |
| `ZActivationCheckpoint` | activaciones INT8/MNEME | Evicción simétrica verificada |
| `scripts/seal.py` | sello sha256+sha512+Merkle + firma Ed25519 obligatoria con lista de confianza | Endurecido tras peritaje |
| `shape_ops` + `elastic_shape` | ElasticShape v1: crecimiento de forma con cirugía exacta | Doble G4 ciego por módulo; claim T8 confirmado (ratio 0.54) |

> **Honestidad técnica sobre ElasticRank:** rinde donde existe redundancia explotable.
> En pretraining desde cero los proxies no predicen el impacto real (Spearman ≈ 0.13) y
> queda deliberadamente inerte con loss-guard. El annealing descendente de rango
> (nacer alto → comprimir durante el entrenamiento) fue probado con cirugía completa y
> **no aporta beneficio detectable** ([T6](docs/evidencia/T6-VEREDICTO.md)); las
> direcciones aprendidas a rango alto no caben en rango bajo.

## Innovaciones reivindicadas (prior art)

Contribuciones originales de MSC Star Team publicadas en [`PRIOR_ART.md`](PRIOR_ART.md):
entrenamiento directo en espacio factorizado SVD; scheduler de rango progresivo;
ZCompressedAdam INT8; ZGradientCompressor con error feedback topology-aware;
ZActivationCheckpoint MNEME/INT8; ElasticRank v1 (sleep/revive/prune con señal AND),
v2 (redundancia por leverage del Gram), v3 (loss-guard como árbitro); probe de
sensibilidad de pérdida; VRAM Governor con recuperación de OOM; reset coordinado
tras refactorización. Fechas y detalle en el documento.

## Desarrollo

```bash
pytest tests/ -v --no-cov          # 347 tests
ruff check src/mztrain/
python scripts/seal.py verify
```

> El gate de cobertura `--cov-fail-under=85` está pre-existentemente por debajo (79.7%);
> juzgar regresiones por conteo de tests con `--no-cov`.

## Hoja de ruta (no implementado, propiedad reservada)

Descritas en [`PRIOR_ART.md`](PRIOR_ART.md) como publicación defensiva; **no** son
reivindicación funcional:

- **fp8 sobre factores** ("la factorización como acondicionador numérico": U,V
  casi-ortonormales → aptos para fp8; S fp32). Hipótesis falsable, hardware SM89 listo.
- **ElasticShape a escala** — el claim T8 (54% del reloj) está probado a 11M-equiv; extenderlo
  a 57M+ (T9), growths encadenados y bf16 es la continuación natural de lo ya publicado en 1.3.
- **Governor anti-acantilado WDDM** — mantener el pico bajo el umbral de paginación
  silenciosa de Windows (medido: vale 10-40× de throughput).
- Soporte distribuido (DDP/FSDP), kernel CUDA del forward factorizado, auto-tuning de
  rango, extensión de ElasticRank a capas sparse.

## Contribuir

Bajo la misma licencia MIT (inbound = outbound); ver [`CONTRIBUTING.md`](CONTRIBUTING.md)
y [`CLA.md`](CLA.md).

## Autoría

- **Esraderey** — co-titular, arquitecto principal y co-inventor.
- **Raúl Cruz Acosta** — co-titular y co-inventor.

## Relacionados

- [MNEME — Motor de Memoria Neural Mórfica](https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica):
  la otra mitad del pipeline — MZTrain entrena por rango, MNEME comprime por bits
  (GPTQ INT4 medido: +35% PPL a 4× menos memoria, sobre el mismo Pythia-410M de la
  serie empírica).

---

© 2025-2026 MSC Star Team. Distribuido bajo licencia **MIT**.
Integridad verificable con `python scripts/seal.py verify`.
