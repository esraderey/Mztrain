# Fase 3: Session Brief

**Objetivo**: Implementar T4 (Spectral Rank Scheduler) y T6 (Sparse + Low-Rank layers)
**Pre-requisito**: Fases 0, 1 y 2 completadas.
**CRITICO**: Antes de Fase 3, arreglar bug en `refactorize.py` (ver seccion BUGS ABIERTOS).

---

## BUGS ABIERTOS (arreglar ANTES de Fase 3)

### BUG CRITICO: refactorize_model destruye el entrenamiento

**Evidencia** (test_09_real_mnist.py — degradation summary):

```
Config                               Accuracy   Gap vs Base
Baseline (Adam, full model)            100.0%          ---
A: SVD factorization only              100.0%       +0.0%
B: + INT8 gradient compression         100.0%       +0.0%
C: + Periodic refactorization           10.6%      +89.4%   <-- DESTRUIDO
D: + GaLore optimizer                  100.0%       +0.0%
E: All features combined                10.6%      +89.4%   <-- por culpa de C
```

- Factorizacion SVD: **0% degradacion** — funciona perfecto
- INT8 gradient compression: **0% degradacion** — funciona perfecto
- GaLore optimizer: **0% degradacion** — funciona perfecto
- Refactorizacion periodica: **89.4% degradacion** — accuracy cae a random chance

**Causa raiz**: La rotacion de momentum en `refactorize.py:_rotate_optimizer_states()`
corrompe los estados exp_avg y exp_avg_sq del optimizer. Especificamente:

1. `rotation_U = U_new[:, :r].T @ U_old[:, :r]` NO es ortogonal cuando
   los subespacios cambian significativamente entre refactorizaciones
2. Multiplicar `m_new = m_old @ rotation.T` con una "rotacion" no-ortogonal
   escala los momentos arbitrariamente, corrompiendo la adaptividad de Adam
3. El decay `v * 0.5` para las regiones nuevas reduce la varianza acumulada,
   causando learning rates efectivos 1.4x mas grandes que los esperados

**Confirmado en**: test_08 (8e: refactorizacion agresiva produce NaN) y
test_09 (C: accuracy 10.6% vs 100% baseline)

**Posibles fixes** (evaluar en Fase 3):

1. **No rotar — resetear estados**: Despues de refactorizar, resetear los estados
   del optimizer para las capas afectadas. Mas simple, pero pierde momentum.
   Combinar con jagged warmup (como en rank growth).

2. **Solo refactorizar sin modificar optimizer**: Actualizar U, S, V pero
   dejar los estados del optimizer intactos. El optimizer se adapta solo
   en los siguientes steps. Riesgo: los estados old son para parametros old.

3. **Refactorizar con merge (estilo ReLoRA)**: En vez de rotar, hacer:
   a. Reconstruir W = U@S@V
   b. Re-SVD -> U_new, S_new, V_new
   c. Crear optimizer NUEVO para esa capa con lazy init
   d. Aplicar warmup de 50-100 steps
   Mas robusto, pero pierde toda la memoria adaptativa.

4. **Usar Frobenius orthogonal procrustes**: Calcular la rotacion ortogonal
   optima R = argmin_R ||U_old - U_new @ R||_F sujeto a R^T R = I.
   Solucion cerrada: R = V @ U^T donde U_old^T @ U_new = U @ S @ V^T.
   Esto GARANTIZA que la rotacion preserva normas.

**Recomendacion**: Fix 3 (resetear + warmup) es el mas seguro. Fix 4 es el mas elegante
pero necesita validacion. NO usar la rotacion actual — esta probada rota.

---

## Estado Actual del Proyecto

### Archivos del sistema (todos en `D:\mztrain\src\mztrain\`)

| Archivo | Lineas | Estado | Fase |
|---------|--------|--------|------|
| `engine.py` | ~640 | Extensivamente modificado | 0-2 |
| `config.py` | ~210 | 9 campos nuevos | 0-2 |
| `gradient.py` | ~270 | Reescrito completo | 1 |
| `refactorize.py` | ~190 | **NUEVO** | 1 |
| `projector.py` | ~320 | **NUEVO** | 2 |
| `checkpoint.py` | ~210 | Mejorado (block-wise INT8) | 2 |
| `layers.py` | ~437 | SIN CAMBIOS (se modifica en Fase 3) | - |
| `scheduler.py` | ~140 | SIN CAMBIOS (se modifica en Fase 3) | - |
| `optimizer.py` | ~247 | SIN CAMBIOS | - |
| `__init__.py` | ~120 | Exports actualizados | 1-2 |

### Tests (en `D:\mztrain_tests\`)

| Test | Que prueba | Resultado | Notas |
|------|-----------|-----------|-------|
| `test_01_gradient_compressor.py` | Top-K, Random-K, 1-bit, INT8, SVD + EF + ConEF | **15/15** | |
| `test_02_galore_optimizer.py` | rSVD, projector, convergencia, memory savings | **13/13** | |
| `test_03_rank_growth.py` | Momentum preservation, warmup, progressive | **12/12** | |
| `test_04_refactorize.py` | Weight preservation, optimizer state survival | **9/9** | Unitario OK, integracion falla |
| `test_05_checkpointing.py` | INT8 compress/decompress, gradient correctness | **8/8** | |
| `test_06_factorize_only.py` | SVD + Adam vanilla (renombrado) | **10/10** | Sin compresiones |
| `test_07_stress_conef.py` | ConEF 2000 steps, Top-K + INT8 | **8/8** | ConEF estable standalone |
| `test_08_stress_numerical.py` | Grad explosion, batch=1, inputs extremos, refact x20 | **18/20** | **2 FAIL: refactorize NaN** |
| `test_09_real_mnist.py` | Dataset NO trivial, degradation per-feature | **7/10** | **3 FAIL: refactorize 89% gap** |

**Total: 92/97 — 5 failures reales, todos en refactorize_model**

**Que se probo realmente en test_09 (ser preciso):**
- A: SVD factorization only (0% gap) — FUNCIONA
- B: SVD + INT8 gradient compression (0% gap) — FUNCIONA
- C: SVD + refactorizacion periodica (89% gap) — ROTO
- D: SVD + GaLore optimizer (0% gap) — FUNCIONA
- E: SVD + INT8 grads + refactorizacion (89% gap) — ROTO por refactorize

**Que NO se probo en test_09 (features desactivadas):**
- compress_optimizer_states (OFF — inestable en modelos <1M params)
- GaLore combinado con gradient compression (OFF — params demasiado pequenos)
- checkpoint_activations (OFF — modelo no tiene TransformerBlocks)
- compress_error_buffer / ConEF (OFF — inestable en engine factorizado)

**Conclusion honesta**: De las 9 features del pipeline, 3 estan probadas
funcionando en integracion (SVD, INT8 grads, GaLore), 1 esta probada ROTA
(refactorize), y 5 no se han probado en integracion (solo unitariamente).

### Documentacion

- `D:\mztrain\docs\ZTRAIN_EVOLUTION_PLAN.md` — Plan maestro con 9 teorias
- `D:\mztrain\docs\FASE3_SESSION_BRIEF.md` — Este documento

---

## Que Implementar en Fase 3

### T4: Spectral Rank Scheduler (Archivo: `scheduler.py`)

#### Que es

Un scheduler de rango que decide CUANDO crecer basandose en el analisis espectral de los
gradientes/pesos, en vez de seguir un schedule fijo (linear, exponential, cosine).

#### Por que importa

Los schedules fijos no consideran el estado real del entrenamiento. El paper de Dynamic Rank
Adjustment (2025) demostro que:
- Low-rank es mas fiel cuando el LR es bajo (low noise regime)
- Full-rank es mas necesario cuando el LR es alto (high noise regime)
- El rango efectivo de los pesos DECAE durante el entrenamiento
- Crecer rango sin verificar que el actual se usa es desperdiciar parametros

#### Algoritmo (del ZTRAIN_EVOLUTION_PLAN.md, seccion 5)

```
CADA N epochs:
  1. Muestrear K capas factorizadas (default K=4, NO todas)
  2. Para cada capa, computar SVD truncada del peso reconstruido
  3. Calcular energy ratio: E(r) = sum(sigma_i^2, i=1..r) / sum(sigma_j^2, all j)
  4. SI E(r_actual) < energy_threshold (default 0.90):
       -> Rango insuficiente, CRECER
  5. SI E(r_actual) > energy_ceiling (default 0.99) Y loss estancado:
       -> Rango suficiente, ajustar LR en su lugar, NO crecer
  6. Zona intermedia: usar schedule base como fallback
```

#### Donde implementar

- **Extender** `src/mztrain/scheduler.py` — anadir `ZSpectralRankScheduler` como nueva clase
- `RankSchedule` enum en `config.py` ya tiene `ADAPTIVE` — reusar o anadir `SPECTRAL`
- **Modificar** `engine.py:train()` — pasar el `model` al scheduler para que pueda analizar espectro
- **Config nuevos campos**:
  - `rank_energy_threshold: float = 0.90`
  - `rank_energy_ceiling: float = 0.99`
  - `rank_sample_layers: int = 4`

#### Interfaz actual del scheduler (NO romper)

```python
# engine.py usa el scheduler asi:
new_rank = self.rank_scheduler.get_rank(
    epoch,
    current_loss=(self._metrics["train_losses"][-1] if self._metrics["train_losses"] else None),
)
if new_rank > self.rank_scheduler.current_rank:
    self._grow_model_rank(new_rank)
    self.rank_scheduler.current_rank = new_rank
```

El nuevo `ZSpectralRankScheduler` debe tener la MISMA interfaz `get_rank(epoch, current_loss)`
pero ademas necesita acceso al modelo para el analisis espectral. Opciones:
- **Opcion A**: Pasar `model` en el constructor (preferida — simple)
- **Opcion B**: Pasar `model` como argumento adicional a `get_rank(epoch, loss, model=None)`
- **Opcion C**: Hacer que engine setee `scheduler.model = self.model` despues de crear el scheduler

**Recomendacion**: Opcion A. Cambiar la creacion en `engine.py:train()`:

```python
# Actualmente (engine.py linea ~480):
self.rank_scheduler = ZRankScheduler(
    initial_rank=self.config.initial_rank,
    max_rank=self.config.max_rank,
    total_epochs=epochs,
    schedule=self.config.rank_schedule,
    growth_interval=self.config.rank_growth_interval,
    growth_factor=self.config.rank_growth_factor,
)

# Cambiar a:
if self.config.rank_schedule == RankSchedule.SPECTRAL:
    self.rank_scheduler = ZSpectralRankScheduler(
        model=self.model,
        initial_rank=..., max_rank=..., total_epochs=...,
        energy_threshold=self.config.rank_energy_threshold,
        energy_ceiling=self.config.rank_energy_ceiling,
        sample_layers=self.config.rank_sample_layers,
        growth_interval=..., growth_factor=...,
    )
else:
    self.rank_scheduler = ZRankScheduler(...)  # existente
```

#### Consideraciones de rendimiento

- SVD completa es O(m * n * min(m,n)). Para una capa 4096x4096 tarda ~50ms.
- Solo muestrear 2-4 capas, no todas.
- Usar SVD aleatorizada de `projector.py:randomized_svd()` para eficiencia.
- El analisis espectral se hace SOLO cada `rank_growth_interval` epochs, no cada step.

---

### T6: Sparse + Low-Rank Layer (Archivo: `layers.py`)

#### Que es

Una capa que parametriza `W = U @ diag(S) @ V + S_sparse` donde:
- `U @ diag(S) @ V` captura el espectro dominante (low-rank, como ahora)
- `S_sparse` captura el espectro de cola con un componente sparse

#### Por que importa

Investigacion (Mayo 2025 benchmarking) demostro que:
- Low-rank puro tiene brecha de **4.25 puntos de perplexity** vs full-rank en modelos 1B
- La razon: no puede capturar singular values pequenos pero importantes (cola del espectro)
- SLTrain (NeurIPS 2024): sparse + low-rank cierra la brecha a ~0.5 puntos
- LOST (2025): SVD-guided sparse allocation mejora aun mas

#### Algoritmo (del ZTRAIN_EVOLUTION_PLAN.md, seccion 7)

```python
class ZSparseFactorizedLinear(nn.Module):
    """W_approx = U @ diag(S) @ V + S_sparse"""

    # Componente Low-Rank: U (m x r), S (r), V (r x n)  — IGUAL que ZFactorizedLinear
    # Componente Sparse: indices fijos (aleatorios), valores entrenables
    #   - sparse_row_idx, sparse_col_idx: buffers NO entrenables
    #   - sparse_values: nn.Parameter (SI entrenable)
    #   - density: tipicamente 1-5% de m*n

    # Forward eficiente:
    #   out_lr = ((x @ V^T) * S) @ U^T    # low-rank path (como antes)
    #   out_sparse = torch.sparse.mm(...)   # sparse path
    #   output = out_lr + out_sparse + bias

    # Inicializacion desde peso existente:
    #   1. SVD truncada -> U, S, V (top-r)
    #   2. Residual = W - U@diag(S)@V
    #   3. sparse_values = residual[sparse_row_idx, sparse_col_idx]

    # grow_rank:
    #   1. Reconstruir W = U@diag(S)@V + sparse
    #   2. Re-SVD con nuevo rango
    #   3. Re-calcular sparse desde nuevo residual
```

#### Donde implementar

- **Extender** `src/mztrain/layers.py` — anadir `ZSparseFactorizedLinear` DESPUES de `ZFactorizedLinear`
- **Modificar** `engine.py:_factorize_model()` — opcion de crear sparse+low-rank
- **Modificar** `engine.py:export_full_model()` / `_defactorize_recursive()` — manejar nueva clase
- **Modificar** `engine.py:_grow_model_rank()` — manejar nueva clase
- **Modificar** `refactorize.py:refactorize_model()` — manejar nueva clase
- **Config nuevos campos**:
  - `use_sparse_component: bool = False`
  - `sparse_density: float = 0.02`

#### Interfaz critica: compatibilidad con ZFactorizedLinear

`ZSparseFactorizedLinear` DEBE tener la misma interfaz publica que `ZFactorizedLinear`:
- `forward(x)` — mismo signature
- `reconstruct_weight()` — ahora retorna `U@diag(S)@V + sparse`
- `grow_rank(new_rank)` — re-SVD + re-sparse
- `num_parameters` — incluye sparse_values
- `full_parameters` — sin cambio
- `compression_ratio` — sin cambio
- `get_stats()` — anadir sparse stats
- `bias` — igual

#### Forward eficiente del sparse path

**NO usar loop Python** — es inaceptablemente lento. Opciones:

```python
# Opcion 1: torch.sparse_coo_tensor (recomendada)
sparse_matrix = torch.sparse_coo_tensor(
    indices=torch.stack([self.sparse_row_idx, self.sparse_col_idx]),
    values=self.sparse_values,
    size=(self.out_features, self.in_features),
)
# out_sparse = sparse_matrix @ x^T  ->  transponer resultado
out_sparse = torch.sparse.mm(sparse_matrix, x.T).T

# Opcion 2: scatter_add (mas rapido para densidades muy bajas)
# Precomputar indices lineales
linear_idx = self.sparse_row_idx * self.in_features + self.sparse_col_idx
# ... usar index_add o scatter

# Opcion 3: si density > 5%, materializar la matriz densa es mas rapido
sparse_dense = torch.zeros(out, in, device=...)
sparse_dense[row_idx, col_idx] = sparse_values
out_sparse = x @ sparse_dense.T
```

**Recomendacion**: Opcion 1 (`torch.sparse_coo_tensor`) para la implementacion base. Si el
profiling muestra que es bottleneck, considerar Triton kernel.

#### Impacto en parametros

```
Ejemplo: capa 4096 x 4096, rank=128, density=2%

Low-rank: 128 * (4096 + 4096) + 128 = ~1.05M params
Sparse:   4096 * 4096 * 0.02 = ~335K params (+ indices como buffers, no entrenables)
Total:    ~1.39M params vs full 16.8M = 91.7% ahorro

Comparado con low-rank puro (~1.05M, 93.7% ahorro):
  - 0.34M params extra (33% mas que low-rank puro)
  - Pero cierra brecha de 4.25 a ~0.5 puntos de perplexity
```

---

## Orden de Implementacion Recomendado

### Paso 1: T6 Sparse+Low-Rank (mas impacto, ~2 dias)

1. Implementar `ZSparseFactorizedLinear` en `layers.py`
2. Anadir `use_sparse_component` y `sparse_density` a `config.py`
3. Modificar `engine.py:_factorize_model()` para opcionalmente crear sparse layers
4. Modificar `engine.py:_defactorize_recursive()` para exportar sparse layers
5. Modificar `engine.py:_grow_model_rank()` para manejar sparse layers
6. Modificar `refactorize.py` para manejar sparse layers
7. Crear `test_07_sparse_lowrank.py`

### Paso 2: T4 Spectral Rank Scheduler (~1 dia)

1. Anadir `SPECTRAL` a `RankSchedule` enum en `config.py`
2. Implementar `ZSpectralRankScheduler` en `scheduler.py`
3. Anadir config fields (`rank_energy_threshold`, etc.)
4. Modificar `engine.py:train()` para seleccionar scheduler
5. Crear `test_08_spectral_scheduler.py`

---

## Dependencias y Funciones Utilitarias Ya Disponibles

### De `projector.py` (reutilizar):

```python
from .projector import randomized_svd
# randomized_svd(M, rank, n_oversamples=10, n_power_iterations=2) -> (U, S, Vt)
# Complejidad O(m*n*rank) — usar para el analisis espectral del scheduler
```

### De `refactorize.py` (extender):

```python
# refactorize_model() actualmente filtra por `factorized_linear_cls`
# Cuando se implemente ZSparseFactorizedLinear, pasar ambas clases o
# hacer que refactorize acepte una tupla de clases
```

### De `layers.py` (base para herencia):

`ZSparseFactorizedLinear` puede **heredar** de `ZFactorizedLinear` y anadir el componente sparse:
```python
class ZSparseFactorizedLinear(ZFactorizedLinear):
    def __init__(self, ..., sparse_density=0.02):
        super().__init__(...)  # Hereda U, S, V, bias, init
        # Anadir sparse component
        self.sparse_density = sparse_density
        num_nonzero = int(out_features * in_features * sparse_density)
        ...
```
Esto evita duplicar codigo y mantiene compatibilidad.

---

## Lecciones Aprendidas de Fases 0-2 (Para No Repetir Errores)

### 1. INT8 quantization: dividir por 127, no por abs_max

```python
# MAL (ternary — solo produce {-1, 0, 1}):
scale = abs_max
quantized = (tensor / scale).round()

# BIEN (usa rango completo INT8):
scale = abs_max / 127.0
quantized = (tensor / scale).round().clamp(-127, 127)
```

### 2. ConEF (error buffer compression) es inestable con compresores de baja distorsion

- ConEF funciona bien con Top-K (alta distorsion, error grande)
- ConEF diverge con INT8 (baja distorsion, error pequeno que se amplifica)
- Default: `compress_error_buffer=False`

### 3. Rank growth es la fuente principal de inestabilidad

- `grow_rank()` en `layers.py` inicializa nuevos componentes con `noise_scale = S.min() * 0.01`
- En modelos pequenos, esto puede ser demasiado grande relativo a los outputs
- El warmup post-growth (50 steps, 10%→100% LR) es CRITICO
- Testeado y estable en test_03

### 4. refactorize_model debe proteger contra NaN

Ya implementado: `if not torch.isfinite(W).all(): continue`

### 5. Modelos pequenos (<100K params) son malos para testear compresion

- INT8 optimizer states + INT8 gradient compression divergen en modelos ~100K
- Estos compresores funcionan bien en modelos >1M params
- Para tests de pipeline, usar solo factorizacion + rank growth sin compresiones
- Para tests de compresion, testear unitariamente (test_01, test_02)

### 6. torch.sparse.mm existe y es eficiente

Para el forward del componente sparse, NO usar loops Python.
`torch.sparse.mm` es la forma correcta. Funciona con autograd.

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

    # --- NUEVOS CAMPOS FASE 3 (a anadir) ---
    # use_sparse_component: bool = False
    # sparse_density: float = 0.02
    # rank_energy_threshold: float = 0.90
    # rank_energy_ceiling: float = 0.99
    # rank_sample_layers: int = 4
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
    # SPECTRAL = "spectral"  # <-- Anadir en Fase 3
```

---

## Referencias Clave Para Fase 3

### T4 (Spectral Scheduler)
- Dynamic Rank Adjustment: Octubre 2025 (arXiv:2508.08625v3) — el paper principal
- AdaLoRA: ICLR 2023 (arXiv:2303.10512) — rank adaptativo por importancia
- DR-LoRA: Enero 2026 (arXiv:2601.04823) — growing > pruning

### T6 (Sparse + Low-Rank)
- SLTrain: NeurIPS 2024 (arXiv:2406.02214) — sparse + low-rank, cierra brecha 4.25→0.5
- LOST: Agosto 2025 (arXiv:2508.02668) — SVD-guided sparse allocation
- Lottery Ticket en LoRA: Diciembre 2025 (arXiv:2512.22495) — sparse subnetworks en low-rank

### Detalle completo de los algoritmos
- `ZTRAIN_EVOLUTION_PLAN.md` seccion 5 (T4) y seccion 7 (T6)
