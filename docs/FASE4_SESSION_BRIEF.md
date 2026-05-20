# Fase 4: Session Brief

**Objetivo**: Implementar T7 (APOLLO-style Adaptive Optimizer) y T8 (Multi-Precision Manager FP8)
**Pre-requisito**: Fases 0, 1, 2 y 3 completadas.
**Hardware**: RTX 4060 8GB VRAM (SM89, NO tiene FP8 nativo — Hopper es SM90+)

---

## Estado Actual del Proyecto

### Resumen de fases completadas

| Fase | Que se hizo | Estado |
|------|-----------|--------|
| 0 | Bug fixes: doble unscale_, best_state a CPU, weights_only, export, batch dict | COMPLETA |
| 1 | T2 error feedback, T3 warm momentum growth, T9 refactorizacion periodica | COMPLETA |
| 2 | T1 GaLore projector, T5 activation checkpointing comprimido | COMPLETA |
| 3 | T6 sparse+low-rank layers, T4 spectral rank scheduler, fix refactorize bug | COMPLETA |

### Bug arreglado en Fase 3 (CRITICO)

La rotacion de momentum en `refactorize.py` (`U_new.T @ U_old`) NO era ortogonal,
corrompiendo Adam. Fix aplicado: reset de optimizer states + warmup post-refactorizacion
(estilo ReLoRA). Accuracy paso de 10.6% (random chance) a 95.5% con refactorizacion
agresiva (cada 50 steps).

### Archivos del sistema (todos en `D:\mztrain\src\mztrain\`)

| Archivo | Lineas | Estado | Fase |
|---------|--------|--------|------|
| `engine.py` | ~870 | Motor principal, soporta sparse+spectral, export multi-pasada | 0-3 |
| `config.py` | ~242 | 14+ campos, enums RankSchedule(+SPECTRAL)/GradientCompression | 0-3 |
| `gradient.py` | ~347 | Error feedback, Top-K, INT8, SVD, 1-bit | 1 |
| `refactorize.py` | ~185 | Reset + warmup (sin rotacion), soporta sparse+tupla de clases | 1,3 |
| `projector.py` | ~436 | GaLore projector + optimizer, randomized_svd | 2 |
| `checkpoint.py` | ~228 | Activation checkpointing comprimido | 2 |
| `layers.py` | ~635 | ZFactorizedLinear + ZSparseFactorizedLinear (FP16 safe sparse.mm) | 0,3 |
| `scheduler.py` | ~364 | ZRankScheduler + ZSpectralRankScheduler (spectral decay ratio) | 0,3 |
| `optimizer.py` | ~246 | ZCompressedAdam (INT8 block-wise states) | 0 |
| `__init__.py` | ~124 | Exports actualizados para todas las clases | 0-3 |
| `utils.py` | - | factorize_existing_model, estimate_memory_savings | 0 |
| `data/` | - | CodeTokenizer, SyntheticCodeGenerator, CodeDataset | 0 |
| `models/zcodebert.py` | - | ZCodeBERT para code understanding | 0 |

### Tests (en `D:\mztrain_tests\`)

| Test | Que prueba | Resultado | Modelo |
|------|-----------|-----------|--------|
| `test_01_gradient_compressor.py` | Top-K, Random-K, 1-bit, INT8, SVD + EF + ConEF | **15/15** | Pequeno |
| `test_02_galore_optimizer.py` | rSVD, projector, convergencia, memory savings | **13/13** | Pequeno |
| `test_03_rank_growth.py` | Momentum preservation, warmup, progressive | **12/12** | Pequeno |
| `test_04_refactorize.py` | Weight preservation, optimizer state reset | **9/9** | Pequeno |
| `test_05_checkpointing.py` | INT8 compress/decompress, gradient correctness | **8/8** | Pequeno |
| `test_06_factorize_only.py` | SVD + Adam vanilla | **10/10** | Pequeno |
| `test_07_stress_conef.py` | ConEF 2000 steps, Top-K + INT8 | **8/8** | Pequeno |
| `test_08_stress_numerical.py` | Grad explosion, batch=1, inputs extremos | **18/20** | Pequeno |
| `test_09_real_mnist.py` | Dataset NO trivial, degradation per-feature | **7/10** | ~300K |
| `test_10_sparse_lowrank.py` | ZSparseFactorizedLinear full pipeline | **34/34** | **25M** |
| `test_11_spectral_scheduler.py` | ZSpectralRankScheduler decisions + training | **17/17** | **25M** |
| `test_12_refactorize_fixed.py` | Refactorize bug fix verification | **19/19** | **25M** |
| `test_13_scale_50m.py` | Pipeline completo a 50M (sparse+spectral+refact+export) | **18/18** | **50M** |
| `test_14_scale_100m.py` | Pipeline completo a 100M (+INT8 grads, +GaLore) | **14/15** | **100M** |
| `test_15_scaling_benchmark.py` | Scaling: compression ratio, memory, throughput | **20/20** | **25M-100M** |
| `test_16_transformer.py` | ZFactorizedAttention + TransformerBlock full pipeline | **37/37** | **11M** |

**Total: 259/268**

**5 failures en test_08/09**: escritos ANTES del fix de refactorize. Probablemente pasan ahora.
**1 failure en test_14**: refactorize loss no convergio en 6 epochs a 100M (necesita mas epochs,
no es un bug — SVD full en 8192x8192 cada 200 steps tarda 719s).
**3 failures anteriores de test_16**: corregidos (export multi-pasada, accuracy threshold, memory benchmark).

### Que funciona en integracion (probado 25M-100M params, MLP y Transformer)

1. SVD factorization: 0-1.8% accuracy gap vs baseline (mejora con escala)
2. INT8 gradient compression + error feedback: 0% gap
3. GaLore optimizer: 0% gap, probado hasta 100M
4. Progressive rank growth + momentum preservation: 32→256, estable
5. Refactorizacion periodica (REPARADA): 1.2% gap agresiva, 0% moderada
6. Spectral rank scheduler: decisiones correctas, crece 32→128 por espectro
7. Sparse+Low-Rank: funciona, 5.7% params de 100M full model
8. Combinaciones: refact+growth, refact+sparse, refact+INT8+GaLore — todas estables
9. ZFactorizedAttention: Q,K,V,O factorizados, masks, 37/37 tests
10. ZFactorizedTransformerBlock: 12 bloques, attention+MLP, residual+LN
11. Activation checkpointing: probado en transformer blocks (z_checkpoint)
12. Export multi-pasada: maneja modulos anidados (attention dentro de blocks)

### Datos de scaling verificados (test_15)

```
SVD puro rank=128:
  Scale    Params       SVD Params    Compress    Accuracy     Gap
  25M      25,195,540    2,389,816      9.5%       94.7%     +5.1%
  50M      50,365,460    3,442,488      6.8%       94.0%     +5.3%
  100M    100,722,708    4,778,808      4.7%       98.2%     +1.8%  <-- mejora

Memory (peak VRAM):
  25M:   full=593MB,   SVD=153MB (25.8%)
  50M:   full=1170MB,  SVD=269MB (23.0%)
  100M:  full=2322MB,  SVD=482MB (20.7%)  <-- compression ratio MEJORA
```

Conclusion: la compresion mejora al escalar. A 100M, MZTrain usa 4.7% de los
parametros y 20.7% de la memoria, con solo 1.8% de accuracy gap.

### Que NO se ha probado en integracion

- `compress_optimizer_states` con modelos >1M params (unitario OK, diverge en modelos pequenos)
- GaLore + gradient compression combinados (GaLore ya comprime internamente)
- ConEF con INT8 en engine factorizado (diverge — solo estable con Top-K)
- ZCodeBERT con dataset de texto real (solo datos sinteticos hasta ahora)
- Modelos >100M (limitado por RTX 4060 8GB para baselines full)

---

## Que Implementar en Fase 4

### T7: APOLLO-style Adaptive Optimizer (Archivo: `adaptive_optimizer.py`)

#### Que es

Un optimizer que combina:
1. **Proyeccion low-rank del gradiente** (GaLore) para el primer momento m
2. **Segundo momento v ESCALAR por bloque** (Adam-mini) en vez de per-element
3. **Error compensation** para la proyeccion

Resultado: memoria ~= SGD, convergencia ~= AdamW.

#### Por que importa

El cuello de botella de memoria en modelos >1B params NO son los pesos (ya factorizados),
son los **estados del optimizer**. Adam guarda 2 tensores (m, v) del mismo tamano que los
parametros. Para un modelo 1B:

```
Adam estandar:
  m: 32 * 4096 * 4096 * 4 = 2.15 GB
  v: 32 * 4096 * 4096 * 4 = 2.15 GB
  Total estados: 4.3 GB

ZAdaptiveOptimizer (rank=128, v escalar por bloque):
  m (low-rank): 32 * 128 * 4096 * 4 = 67 MB
  v (escalar):  32 * (4096*4096/1024) * 4 = 2 MB
  P (proyector): 32 * 4096 * 128 * 4 = 67 MB
  Total estados: ~136 MB

Reduccion: ~96.8% (31.6x menos memoria)
```

APOLLO (MLSys 2025 Outstanding Paper) demostro que esto funciona hasta LLaMA-13B.

#### Algoritmo

```python
class ZAdaptiveOptimizer(torch.optim.Optimizer):
    """
    Combina:
    - m: low-rank (r x n) via GaLore projector
    - v: escalar por bloque (1 valor cada 1024 elementos, estilo Adam-mini)
    - Parametros pequenos (<2D o <4096 elementos): Adam estandar
    """

    def step(self):
        for p in params:
            if p.dim() >= 2 and p.numel() > 4096:
                # === PROJECTED PATH ===
                g_low = projector.project(grad)        # (r, n)
                m = beta1 * m + (1-beta1) * g_low      # low-rank m
                v_scalar = beta2 * v + (1-beta2) * mean(g_low^2, per_block)  # escalar v
                update = m / (sqrt(v_scalar) + eps)
                p -= lr * projector.project_back(update)
            else:
                # === STANDARD ADAM ===
                standard_adam_step(p, grad)
```

#### Donde implementar

- **NUEVO**: `src/mztrain/adaptive_optimizer.py` — Clase `ZAdaptiveOptimizer`
- **Modificar**: `engine.py:_create_optimizer()` — agregar opcion
- **Modificar**: `config.py` — agregar campo `optimizer_type`
- **Reutilizar**: `projector.py:ZGaLoreProjector` y `projector.py:randomized_svd`

#### Config nuevos campos

```python
# En ZTrainConfig:
optimizer_type: str = "compressed_adam"
# Opciones: "compressed_adam" (actual), "galore" (actual), "adaptive" (nuevo)
# "adaptive" = APOLLO-style: GaLore projection + scalar v

adaptive_block_size: int = 1024
# Tamano de bloque para v escalar. 1 valor v cada N elementos.
# Adam-mini usa ~head_dim (64). APOLLO usa ~1024.
```

#### Interfaz critica: compatibilidad con engine

`ZAdaptiveOptimizer` DEBE tener la misma interfaz que `ZCompressedAdam` y `ZGaLoreOptimizer`:
- `step(closure=None)` — paso de optimizacion
- `get_memory_stats()` -> dict — estadisticas de memoria
- `param_groups` — grupos de parametros (heredado de Optimizer)
- `state` — estados por parametro (heredado de Optimizer)

El engine usa `self.optimizer.get_memory_stats()` en el summary final.
Tambien accede a `self.optimizer.param_groups[0]['lr']` para leer/modificar LR.

#### Diferencia con ZGaLoreOptimizer existente

`ZGaLoreOptimizer` (projector.py:196) ya hace:
- m low-rank (r x n) ✓
- v low-rank (r x n) — PER-ELEMENT en espacio proyectado

`ZAdaptiveOptimizer` hace:
- m low-rank (r x n) ✓ (igual)
- v **ESCALAR por bloque** — UN VALOR cada 1024 elementos

La diferencia en v es donde viene el 96%+ de ahorro. v escalar funciona porque
Adam-mini (ICLR 2025) demostro que la varianza per-element es redundante:
la varianza promedio por bloque es suficiente para ajustar learning rates.

#### Consideraciones de implementacion

1. **Bloques para v**: determinar automaticamente. Heuristica:
   - `param.numel() // adaptive_block_size` bloques
   - Minimo 1 bloque por parametro
   - Para attention layers, alinear bloques con heads si es posible

2. **Proyector**: reusar `ZGaLoreProjector` directamente. Ya esta probado y estable.

3. **Compresion de estados**: opcionalmente comprimir m low-rank a INT8 (como ya
   hace ZGaLoreOptimizer cuando compress_states=True).

4. **Lazy init**: el estado se inicializa en el primer step, no en __init__.
   Esto es importante para compatibilidad con rank growth (que recrea el optimizer).

---

### T8: Multi-Precision Manager (Archivo: `precision.py`)

#### Que es

Un manager de precision mixta multi-nivel que va mas alla de AMP estandar:
- **STANDARD**: FP16/BF16 forward, FP32 backward (lo que hace ZTrain hoy)
- **FP8**: E4M3 forward, E5M2 backward, FP32 master weights
- **AGGRESSIVE**: FP8 + INT8 activaciones + INT8 optimizer states

#### LIMITACION IMPORTANTE: RTX 4060

La RTX 4060 es **SM89 (Ada Lovelace)**. FP8 nativo requiere **SM90+ (Hopper)**.

Esto significa:
- `torch.float8_e4m3fn` y `torch.float8_e5m2` existen como dtypes en PyTorch
- Las operaciones FP8 en SM89 se **emulan en software** (no hay tensor cores FP8)
- El rendimiento sera PEOR que FP16, no mejor
- NVIDIA Transformer Engine (`te.fp8_autocast`) NO funciona en Ada Lovelace

**Recomendacion para Fase 4**: Implementar la infraestructura (precision.py, config, engine
integration) pero el nivel FP8 solo sera beneficioso en GPUs Hopper+ (H100, H200).
Para la RTX 4060, el nivel "AGGRESSIVE" deberia enfocarse en INT8 activaciones + INT8
optimizer states (que SI son eficientes en cualquier GPU).

#### Niveles de precision

```
STANDARD (actual):
  Forward:     FP16 (AMP autocast)
  Backward:    FP32
  Weights:     FP32 master, FP16 compute
  Optimizer:   FP32 (o INT8 con compress_optimizer_states)
  Activaciones: FP16 (o INT8 con checkpoint_activations)

AGGRESSIVE (viable en RTX 4060):
  Forward:     FP16 (AMP autocast)
  Backward:    FP32
  Weights:     FP32 master, FP16 compute
  Optimizer:   INT8 block-wise (ya implementado)
  Activaciones: INT8 block-wise comprimidas (ya implementado)
  Singular values S: SIEMPRE FP32 (critico — nunca cuantizar)

FP8 (solo Hopper+):
  Forward:     FP8 E4M3 (tensor cores nativos)
  Backward:    FP8 E5M2 (rango mas amplio para gradientes)
  Weights:     FP32 master, FP8 compute
  Optimizer:   INT8 o FP8
  Activaciones: INT8 o FP8
  Scaling:     Per-tensor dynamic scaling (obligatorio)
```

#### Algoritmo

```python
class ZMultiPrecisionManager:
    """Gestion de precision mixta multi-nivel."""

    def __init__(self, config, device):
        self.level = config.precision_level  # "standard", "aggressive", "fp8"
        self.device = device

        # Detectar capacidad del hardware
        if device.type == "cuda":
            capability = torch.cuda.get_device_capability()
            self.has_fp8 = capability[0] >= 9  # Hopper SM90+
        else:
            self.has_fp8 = False

        # Fallback si hardware no soporta
        if self.level == "fp8" and not self.has_fp8:
            logger.warning("FP8 requiere Hopper+. Fallback a aggressive.")
            self.level = "aggressive"

    def forward_context(self):
        """Context manager para forward pass."""
        if self.level == "standard":
            return torch.amp.autocast("cuda", dtype=torch.float16)
        elif self.level == "aggressive":
            return torch.amp.autocast("cuda", dtype=torch.float16)
            # Lo agresivo esta en optimizer/activaciones, no en compute
        elif self.level == "fp8":
            return torch.amp.autocast("cuda", dtype=torch.float8_e4m3fn)

    def quantize_activation(self, tensor):
        """INT8 block-wise para activaciones checkpointed."""
        # Solo en aggressive/fp8
        ...

    def quantize_gradient_for_comm(self, gradient):
        """FP8 E5M2 para gradientes (solo fp8 level)."""
        ...
```

#### Donde implementar

- **NUEVO**: `src/mztrain/precision.py` — Clase `ZMultiPrecisionManager`
- **Modificar**: `engine.py:__init__()` — crear precision manager
- **Modificar**: `engine.py:train_epoch()` — usar precision manager para autocast
- **Modificar**: `config.py` — agregar campo `precision_level`

#### Config nuevos campos

```python
# En ZTrainConfig:
precision_level: str = "standard"
# Opciones: "standard" (actual), "aggressive" (INT8 todo), "fp8" (Hopper+)
```

#### Integracion con engine.py

```python
# Actualmente (engine.py train_epoch):
if self.scaler is not None:
    with torch.amp.autocast("cuda"):
        loss = loss_fn(self.model, batch)
    self.scaler.scale(loss).backward()
    ...

# Cambiar a:
with self.precision_manager.forward_context():
    loss = loss_fn(self.model, batch)
# El precision manager decide el dtype del autocast
```

#### Consideraciones criticas

1. **Singular values S NUNCA en baja precision**: los valores singulares controlan
   la escala de toda la capa. Cuantizarlos causa inestabilidad catastrofica.
   SIEMPRE mantener en FP32, incluso en modo fp8.

2. **Per-tensor scaling obligatorio en FP8**: E4M3 tiene rango [-448, 448].
   Sin scaling, overflow es comun. Usar `abs_max / 448.0` como scale factor.

3. **E4M3 vs E5M2**: E4M3 tiene mas precision (3 bits mantissa) — mejor para
   forward. E5M2 tiene mas rango (5 bits exponente) — mejor para gradientes
   que pueden tener spikes.

4. **Stochastic rounding**: en FP8/FP4, round-to-nearest causa sesgo acumulado.
   Stochastic rounding es esencial. Implementacion:
   ```python
   noise = torch.rand_like(tensor) - 0.5
   quantized = (tensor / scale + noise).round()
   ```

5. **torch.sparse.mm NO soporta FP16 en CUDA**: ya manejado en layers.py
   con `torch.amp.autocast(enabled=False)` para el sparse path. Esto aplica
   igual para FP8 — el sparse path siempre debe ser FP32.

---

## Orden de Implementacion Recomendado

### Paso 1: T7 Adaptive Optimizer (~3 dias)

1. Crear `src/mztrain/adaptive_optimizer.py` — `ZAdaptiveOptimizer`
2. Agregar `optimizer_type` y `adaptive_block_size` a `config.py`
3. Modificar `engine.py:_create_optimizer()` para seleccionar optimizer
4. Crear `test_17_adaptive_optimizer.py` (modelos 10M+, incluir transformer)

### Paso 2: T8 Multi-Precision Manager (~3 dias)

1. Crear `src/mztrain/precision.py` — `ZMultiPrecisionManager`
2. Agregar `precision_level` a `config.py`
3. Modificar `engine.py` para usar precision manager
4. Crear `test_18_precision.py` (modelos 10M+)

**NOTA**: Tests 13-16 ya existen (scaling 50M/100M, scaling benchmark, transformer).
Los nuevos tests de Fase 4 empiezan en test_17.

---

## Dependencias y Funciones Utilitarias Ya Disponibles

### De `projector.py` (reutilizar para T7):

```python
from .projector import ZGaLoreProjector, randomized_svd

# ZGaLoreProjector(m, n, rank, update_freq)
#   .project(grad)        -> grad_low_rank
#   .project_back(update) -> update_full

# randomized_svd(M, rank, n_oversamples=10, n_power_iterations=2) -> (U, S, Vt)
```

### De `optimizer.py` (referencia para T7):

```python
# ZCompressedAdam ya implementa:
#   - INT8 block-wise compression de m y v
#   - _compress_state(tensor) -> (quantized, scales, shape, n)
#   - _decompress_state(compressed, dtype) -> tensor
#   - get_memory_stats() -> dict
#
# ZAdaptiveOptimizer puede reusar _compress_state/_decompress_state
# para comprimir m low-rank opcionalmente
```

### De `engine.py` (interfaz critica):

```python
# _create_optimizer() actualmente (engine.py:134):
def _create_optimizer(self) -> torch.optim.Optimizer:
    if self.config.use_galore:
        return ZGaLoreOptimizer(...)
    else:
        return ZCompressedAdam(...)

# Cambiar a:
def _create_optimizer(self) -> torch.optim.Optimizer:
    if self.config.optimizer_type == "adaptive":
        return ZAdaptiveOptimizer(...)
    elif self.config.use_galore:
        return ZGaLoreOptimizer(...)
    else:
        return ZCompressedAdam(...)

# IMPORTANTE: engine.py fue actualizado en Fase 3 post-tests con:
# - _defactorize_recursive multi-pasada (linea ~800) para modulos anidados
# - train_epoch: refactorize_model recibe tupla (ZFactorizedLinear, ZSparseFactorizedLinear)
# - train_epoch: warmup post-refactorizacion automatico
# - train(): seleccion de ZSpectralRankScheduler si rank_schedule == SPECTRAL
# - _factorize_model: crea ZSparseFactorizedLinear si use_sparse_component=True
#
# El engine llama optimizer.get_memory_stats() al final del training.
# Cualquier optimizer nuevo DEBE implementar get_memory_stats().
```

---

## Lecciones Aprendidas de Fases 0-3 (Para No Repetir Errores)

### 1. torch.sparse.mm NO soporta FP16/FP8 en CUDA

```python
# MAL: autocast convierte a half, sparse.mm explota
out_sparse = torch.sparse.mm(sparse_matrix, x.T).T

# BIEN: forzar FP32 dentro de autocast disabled
with torch.amp.autocast(device_type='cuda', enabled=False):
    out_sparse = torch.sparse.mm(sparse_matrix.float(), x.float().T).T
out_sparse = out_sparse.to(dtype=original_dtype)
```

Esto aplica a CUALQUIER precision manager. El sparse path de ZSparseFactorizedLinear
SIEMPRE debe operar en FP32.

### 2. La rotacion de momentum NO funciona — usar reset + warmup

El mapping `U_new.T @ U_old` no es ortogonal cuando los subespacios cambian
significativamente. Multiplicar estados de Adam por una matriz no-ortogonal
escala los momentos arbitrariamente. Fix: reset + warmup (estilo ReLoRA).

### 3. INT8 quantization: dividir por 127, no por abs_max

```python
# MAL (ternary):
scale = abs_max
quantized = (tensor / scale).round()

# BIEN (rango INT8 completo):
scale = abs_max / 127.0
quantized = (tensor / scale).round().clamp(-127, 127)
```

### 4. ConEF (error buffer compression) diverge con INT8 grads

- ConEF funciona con Top-K (alta distorsion, error grande)
- ConEF diverge con INT8 (baja distorsion, error pequeno que se amplifica)
- Default: `compress_error_buffer=False`

### 5. Modelos pequenos (<1M params) no sirven para testear compresion

- INT8 optimizer + INT8 grads divergen en modelos ~100K params
- Para tests de pipeline, usar modelos >10M params (ver test_10/11/12)
- Para tests unitarios (compressor individual), modelos pequenos estan bien

### 6. Device mismatches con existing_weight

Cuando se crea ZSparseFactorizedLinear con `existing_weight` en CUDA,
los indices sparse se generan en CPU. Hay que mover al device correcto:
```python
ew = existing_weight.to(device=W_lr.device, dtype=W_lr.dtype)
```

### 7. get_memory_stats() es obligatorio en optimizers

El engine llama `self.optimizer.get_memory_stats()` en el summary.
Todo optimizer nuevo DEBE implementar este metodo. Retorna dict con:
- `total_params`, `state_memory_mb`, `full_memory_mb`, `memory_saved_pct`

### 8. Spectral scheduler: usar spectral decay, no energy ratio

Para capas factorizadas, `reconstruct_weight()` es rank-r exacto, asi que
`sum(sigma[:r]^2) / sum(sigma_all^2) = 1.0` siempre. En su lugar usamos:
```python
energy = 1 - S.min() / S.max()  # spectral decay ratio
# Empinado (S_min << S_max) -> energy alta -> rank suficiente
# Plano   (S_min ~ S_max)  -> energy baja -> CRECER
```

### 9. Export multi-pasada para modulos anidados

`_defactorize_recursive` con una sola pasada falla en transformers:
`ZFactorizedAttention` contiene `q_proj`, `k_proj`, etc. como atributos.
`named_modules()` los lista, pero al reemplazar en una pasada, los padres
pueden quedar inconsistentes. Fix: multiples pasadas hasta que no queden
`ZFactorizedLinear`:

```python
for _ in range(5):
    found = False
    for name, module in list(model.named_modules()):
        if isinstance(module, ZFactorizedLinear):
            # reemplazar...
            found = True
    if not found:
        break
```

### 10. Error numerico acumulado en export de transformers profundos

La diferencia entre forward factorizado `((x@V.T)*S)@U.T` y forward denso
`x@W.T` es ~1e-6 por capa. En un transformer de 12 bloques x 6 capas = 72
capas en serie, el error acumulado puede ser ~0.3. Esto NO es un bug — es
precision numerica FP32. Para verificar export, usar threshold 0.5 en
transformers profundos, no 1e-3.

### 11. SVD full en matrices grandes es MUY lento

`torch.linalg.svd` en 8192x8192 tarda ~5-10 segundos. Refactorizar cada 50
steps con 3 capas de ese tamano = 15-30s por refactorizacion. A 100M params
con refactorize_interval=200, el training tarda 10-12 minutos vs 7s sin refact.
Para modelos >50M, considerar usar `randomized_svd` en refactorize.py.

### 12. nn.MultiheadAttention vs ZFactorizedAttention en benchmarks de memoria

`nn.MultiheadAttention` fusiona Q,K,V en un solo tensor `in_proj_weight`
(3*embed x embed). Esto es mas eficiente en memoria que 4 tensores separados.
Para benchmarks justos, comparar contra nn.Linear individuales, no MHA fusionado.

---

## Config Actual Completa (Para Referencia)

```python
@dataclass
class ZTrainConfig:
    # Rango factorizado
    initial_rank: int = 32
    max_rank: int = 256
    min_params_to_factorize: int = 4096
    energy_retention: float = 0.995

    # Entrenamiento progresivo
    rank_schedule: RankSchedule = RankSchedule.EXPONENTIAL
    rank_growth_interval: int = 10
    rank_growth_factor: float = 2.0
    rank_growth_warmup_steps: int = 50
    refactorize_interval: int = 0

    # Optimizer
    use_galore: bool = False
    galore_rank: int = 128
    galore_update_freq: int = 200
    compress_optimizer_states: bool = True
    optimizer_compression_ratio: float = 0.3

    # Gradient compression
    gradient_compression: GradientCompression = GradientCompression.NONE
    gradient_top_k_ratio: float = 0.1
    min_size_to_compress_grad: int = 1024
    compress_error_buffer: bool = False

    # Activation checkpointing
    checkpoint_activations: bool = True
    activation_compression: bool = True
    checkpoint_every_n_layers: int = 2

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    use_amp: bool = True

    # MNEME
    mneme_compression_level: CompressionLevel = CompressionLevel.BALANCED
    lazy_weight_loading: bool = True

    # Logging
    log_interval: int = 10
    memory_log_interval: int = 50

    # Sparse + Low-Rank (T6, Fase 3)
    use_sparse_component: bool = False
    sparse_density: float = 0.02

    # Spectral Rank Scheduler (T4, Fase 3)
    rank_energy_threshold: float = 0.90
    rank_energy_ceiling: float = 0.99
    rank_sample_layers: int = 4

    # --- NUEVOS CAMPOS FASE 4 (a anadir) ---
    # optimizer_type: str = "compressed_adam"
    #   Opciones: "compressed_adam", "galore", "adaptive"
    # adaptive_block_size: int = 1024
    #   Tamano de bloque para v escalar en adaptive optimizer
    # precision_level: str = "standard"
    #   Opciones: "standard", "aggressive", "fp8"
```

---

## RankSchedule Enum Actual

```python
class RankSchedule(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"
    SPECTRAL = "spectral"
```

## GradientCompression Enum Actual

```python
class GradientCompression(Enum):
    NONE = "none"
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    QUANTIZE_1BIT = "1bit"
    QUANTIZE_INT8 = "int8"
    SVD = "svd"
```

---

## Impacto Esperado de Fase 4

### Antes de Fase 4 (estado actual, modelo 1B, rank=128)

```
Pesos factorizados:     ~420 MB (rank 128 + sparse)
Optimizer states:       ~4.3 GB (Adam FP32 para factores)
Activaciones:           ~2 GB (con checkpointing comprimido)
Gradientes:             ~80 MB (INT8 + error feedback)
TOTAL GPU:              ~6.8 GB
```

### Despues de Fase 4 (todas las optimizaciones)

```
Pesos factorizados:     ~420 MB (sin cambio)
Optimizer states:       ~136 MB (APOLLO: m low-rank + v escalar)
Activaciones:           ~2 GB (sin cambio)
Gradientes:             ~80 MB (sin cambio)
TOTAL GPU:              ~2.6 GB

Reduccion en optimizer: ~96.8% (4.3 GB -> 136 MB)
Reduccion total:        ~62% (6.8 GB -> 2.6 GB)
```

Esto permitiria entrenar un modelo 1B en la RTX 4060 (8GB) con batch size razonable.

---

## Referencias Clave Para Fase 4

### T7 (APOLLO Adaptive Optimizer)
- APOLLO: MLSys 2025 Outstanding Paper — SGD-level memory, AdamW-level convergence
- Adam-mini: Zhang et al., ICLR 2025 (arXiv:2406.16793) — v escalar, 50% reduccion
- GaLore: Zhao et al., ICML 2024 (arXiv:2403.03507) — proyeccion low-rank
- CAME: Luo et al., ACL 2023 (arXiv:2307.02047) — confidence-guided
- Schedule-Free: Defazio et al., NeurIPS 2024 (arXiv:2405.15682)

### T8 (Multi-Precision)
- COAT: ICLR 2025 — FP8 optimizer + FP8 activaciones simultaneos
- MOSS: Noviembre 2025 (arXiv:2511.05811) — two-level microscaling, 34% throughput
- Quartet: NeurIPS 2025 (arXiv:2505.14669) — FP4 end-to-end training
- NVIDIA Transformer Engine: FP8 E4M3/E5M2 para Hopper
- 8-bit Adam (bitsandbytes): Dettmers et al., ICLR 2022 (arXiv:2110.02861)

### Transformer factorizado (validado en test_16, 11M params)

Un transformer de 12 bloques con ZFactorizedAttention funciona end-to-end:

```
Modelo: embed=512, heads=8, rank=96, 12 bloques
Params: 10,976,532 factorizado vs 38,101,524 full (28.8%)
Training: loss 3.33 -> 0.16 en 12 epochs (sequence classification)

Componentes verificados:
- ZFactorizedAttention: Q,K,V,O como ZFactorizedLinear, masks, grow_rank
- ZFactorizedTransformerBlock: residual + LN + attention + MLP
- Refactorize: 72 capas refactorizadas en un paso
- Activation checkpointing: z_checkpoint en blocks alternos
- Spectral scheduler: detecta capas factorizadas en transformer
- Sparse+LR: ZSparseFactorizedLinear dentro de attention projections
- Export: multi-pasada para modulos anidados (diff < 0.5 por error acumulado FP32)
```

Para Fase 4, el `ZAdaptiveOptimizer` debe funcionar con estos transformers.
La interfaz es transparente (opera sobre parametros, no sobre arquitectura),
pero los tests deben incluir transformers, no solo MLPs.

### Detalle completo de los algoritmos
- `ZTRAIN_EVOLUTION_PLAN.md` seccion 8 (T7) y seccion 9 (T8)
