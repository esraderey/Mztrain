# PUBLICACIÓN DEFENSIVA — PRIOR ART

**Obra:** MZTrain — Motor de Entrenamiento en Espacio Comprimido
**Autores / inventores:** MSC Star Team (Esraderey, Raúl Cruz Acosta)
**Fecha de publicación de este documento:** 2026-05-20
**Sellado criptográfico:** ver `SEAL.json` y `MANIFEST.sha256`

---

## Finalidad del documento

Este documento se publica con efectos de **publicación defensiva** y
**establecimiento de fecha cierta de invención** sobre las técnicas y
diseños abajo descritos. Su propósito es:

1. Constituir **estado de la técnica (prior art)** oponible frente a
   cualquier solicitud de patente posterior, propia o de terceros,
   que reivindique las mismas invenciones o equivalentes obvios.
2. Documentar la **paternidad intelectual** de los autores sobre
   estas contribuciones.
3. Servir como **anclaje técnico** para acciones de defensa en caso
   de plagio, infracción o reivindicación indebida.

Las descripciones siguientes son **suficientemente específicas** para
permitir a un experto en la materia reproducir cada técnica a partir
del código fuente acompañante (`src/mztrain/`), sin necesidad de
información adicional.

---

## Invención 1 — Entrenamiento directo en espacio factorizado SVD

**Fecha de primera publicación:** 2025-01-15
**Archivos clave:** `src/mztrain/layers.py`, `src/mztrain/engine.py`

**Planteamiento técnico:** mientras la práctica habitual factoriza
SVD-mente un modelo **ya entrenado** para reducir su huella, los
autores reivindican el procedimiento de **entrenar de origen los
factores `U ∈ ℝ^{m×r}`, `S ∈ ℝ^r`, `V ∈ ℝ^{r×n}` como parámetros
optimizables de primer orden**, sin reconstruir `W = U·diag(S)·Vᵀ`
en ningún paso de forward. El forward eficiente
`y = ((x·V) · S) · Uᵀ` evita materializar `W` y mantiene la huella
de gradientes, optimizador y activaciones a una fracción
proporcional a `r/min(m,n)`.

**Elementos distintivos:**
- `S` se mantiene como vector entrenable independiente (no se absorbe
  en `U` o `V`), lo que permite (a) inspección espectral durante
  el entrenamiento, (b) gating por dirección (`wake_gate`), y
  (c) ordenación por magnitud sin recomputar SVDs costosas.
- Inicialización dual: SVD desde pesos pre-existentes o desde cero
  con `Kaiming` sobre `U` y `V` y `S=1`.
- Rango progresivo: `r` puede crecer durante el entrenamiento sin
  cold-start (preservación de los `r` antiguos al añadir nuevas
  direcciones).

## Invención 2 — Scheduler de rango progresivo con 5 políticas

**Fecha de primera publicación:** 2025-01-15
**Archivos clave:** `src/mztrain/scheduler.py`, `src/mztrain/engine.py`

Crecimiento de `r` a lo largo de las épocas según una de:
`CONSTANT`, `LINEAR`, `EXPONENTIAL`, `COSINE`, `ADAPTIVE` (basada en
convergencia del loss). La política `ADAPTIVE` aumenta `r` sólo si
la pendiente del loss en una ventana móvil cae por debajo de un
umbral, lo que evita gasto de capacidad sobre un modelo aún en
fase rápida de aprendizaje.

## Invención 3 — Optimizador Adam con estados comprimidos INT8

**Fecha de primera publicación:** 2025-01-15
**Archivos clave:** `src/mztrain/optimizer.py`

`ZCompressedAdam` cuantiza los momentos `m` y `v` a INT8 con escala
por tensor, descuantiza para el paso y vuelve a cuantizar
periódicamente (`compression_interval`), evitando el coste de
cuantizar/descuantizar en cada step.

## Invención 4 — Compresor de gradientes con error feedback

**Fecha de primera publicación:** 2025-01-15
**Archivos clave:** `src/mztrain/gradient.py`

Estrategias: `TOP_K`, `ONE_BIT` (SignSGD escalado), `INT8`, `SVD`.
Cada nombre de parámetro mantiene un **buffer de error feedback** que
acumula la diferencia entre el gradiente real y el comprimido para
ser sumada en pasos posteriores, garantizando convergencia.

## Invención 5 — Error feedback topology-aware

**Fecha de primera publicación:** 2026-05-16
**Archivos clave:** `src/mztrain/gradient.py`, regression test en
`tests/test_gradient.py`

Cuando un parámetro **cambia de forma** durante el entrenamiento
(rank growth, ElasticRank sleep/compactación, refactorización), su
buffer de error feedback queda desfasado. La invención introduce
**invalidación automática** del buffer por desajuste de forma y
contabilización del evento (`error_buffer_resets`). Esta
interacción cruzada se documenta como prior art porque resuelve un
bug latente entre dos features que mutan forma y un compresor con
estado por nombre.

## Invención 6 — ElasticRank v1: rango bidireccional con sleep bank

**Fecha de primera publicación:** 2026-05-15
**Archivos clave:** `src/mztrain/elastic_rank.py`

Procedimiento por el cual una dirección singular `i` puede:

- **DORMIR**: sus factores `U_:,i`, `S_i`, `V_i,:` y sus estados
  Adam (`exp_avg`, `exp_avg_sq`) se mueven a un *sleep bank* en CPU
  y baja precisión.
- **REVIVIR**: se reactivan en GPU recuperando los estados anteriores
  (continuidad de momentum) en lugar de un cold-start.
- **PODARSE**: tras un tiempo dormido se elimina definitivamente.

**Señal de sleep híbrida con AND (no producto):**

```
dead_i ⇔ (spectral_ema_i < τ_s)
       ∧ (update_ema_i   < τ_u)
       ∧ (low_counter_i ≥ patience)
       ∧ (age_i ≥ min_age)
       ∧ (revive_cd_i == 0)
       ∧ ¬grace ∧ ¬warmup
```

con:
- `spectral_score_i = |S_i|·‖U_:,i‖·‖V_i,:‖`, normalizada por un
  cuantil alto (p90/p95), no por el máximo;
- `update_score_i` = energía del paso efectivo de Adam por dirección
  `(m / (√v + ε))` propagada por la regla del producto sobre
  `U, S, V`, normalizada por la mediana.

El **AND de umbrales separados** (no el producto) impide matar
direcciones con score espectral bajo pero todavía actualizándose.

## Invención 7 — ElasticRank v2: señal de redundancia funcional

**Fecha de primera publicación:** 2026-05-15
**Archivos clave:** `src/mztrain/elastic_rank.py` (camino redundancia)

Camino **independiente** y opt-on por defecto que detecta direcciones
**redundantes en el span del resto**, no sólo las decadentes. Se
calcula el Gram de los términos rank-1 unitarios:

```
R = (Û_:,iᵀ · Û_:,j) ⊙ (V̂_i,: · V̂_j,:ᵀ)     con Û, V̂ normalizadas
coherence_i = 1 − 1 / (R + λ·I)⁻¹_ii   ∈ [0, 1)
```

Una `coherence_i → 1` indica que la dirección `i` cae en el span de
las otras: no añade capacidad aunque siga actualizándose. Por eso
este camino **NO** está gobernado por la puerta de actividad
(`update_score`). El uso de la matriz de **correlación** (factores
normalizados a norma 1) en lugar del Gram crudo es **invariante a
escala**, lo que evita que un ridge global fijado por la dirección
mayor empuje `coherence ≈ 0` para direcciones redundantes de baja
energía — defecto detectado y corregido por los autores en el
mismo ciclo de diseño.

## Invención 8 — Loss-Guard como núcleo de decisión (ElasticRank v3)

**Fecha de primera publicación:** 2026-05-15
**Archivos clave:** `src/mztrain/elastic_rank.py`,
`src/mztrain/engine.py` (`_apply_elastic_topology_guarded`,
`_snapshot_elastic_state`, `_restore_elastic_state`)

Inversión del flujo de decisión: los proxies de peso (espectral,
update, coherencia) sólo generan **candidatos**; el delta de pérdida
**medido** sobre un batch de prueba (probe batch, eval / no_grad,
batch fijo) es el árbitro final. Cada compactación se aplica como
**tentativa**: antes/después se mide `rel_loss_delta` y, si supera
`elastic_rank_loss_guard_threshold` (1e-3 por defecto), se revierte
**estructuralmente** el estado:

- factores (U, S, V),
- buffers (wake_gate, sleep_mask),
- estado de optimizador (`exp_avg`, `exp_avg_sq`),
- contadores internos (`_states`, bank, warmup),
- cooldown por capa `_rollback_cd`.

El probe se ejecuta sólo cuando el plan dorme efectivamente
(grow/revive puros se saltan para no penalizar revivals
crecientes). Este diseño hace ElasticRank **safe-by-construction**
en escenarios sin redundancia explotable.

## Invención 9 — Probe de sensibilidad de pérdida por dirección

**Fecha de primera publicación:** 2026-05-15
**Archivos clave:**
`ElasticRankController.probe_loss_sensitivity` en
`src/mztrain/elastic_rank.py`; `bench/elastic_bench_real.py --probe`

Diagnóstico no destructivo que mide, por dirección singular `i`, el
delta relativo de pérdida al fijar `wake_gate[i]=0` sobre un batch
fijo. Los autores demostraron empíricamente que la correlación de
Spearman entre los proxies weight-space (espectral, update,
coherencia) y el delta-loss real es ≈ **0.13** — proxies **no
predictivos** — lo que motiva la adopción del loss-delta medido
como único criterio (Invención 8). La heurística booleana
`heterogeneous` se ajustó por los autores para requerir tanto
`p90_abs_rel_delta > 1e-3` como `ratio > 2`, evitando falsos
positivos por CV sobre valores ≈ 0.

## Invención 10 — VRAM Governor con gating de crecimiento de rango

**Fecha de primera publicación:** 2026-05-15
**Archivos clave:** `src/mztrain/vram_governor.py`,
integración en `src/mztrain/engine.py`

Núcleo MVP, validado en GPU real (RTX 4060, 15/15 PASS,
`bench/governor_gpu_check.py`):

- Sensor de memoria CUDA-guarded con lector inyectable
  (`mem_reader=`), permitiendo tests deterministas sin contaminación
  cross-test por `torch.cuda.max_memory_allocated()`.
- Presión EMA con 3 modos (`normal` / `preventive` / `emergency`) e
  **histéresis** para evitar oscilación.
- **Veto de crecimiento de rango** bajo presión: la decisión de la
  invención 2 (rank growth) pasa por `approve_rank_growth()` y
  puede ser bloqueada.
- **Recuperación de OOM transitorio** con un único reintento.

Diseño **deliberadamente acotado**: rank-budget en bytes, control de
precisión, control de checkpointing y acoplamiento con ElasticRank
están definidos como roadmap pero no implementados, por la decisión
documentada de no construir sobre proxies no predictivos sin
validación en GPU real.

## Invención 11 — Activation checkpointing con compresión MNEME/INT8

**Fecha de primera publicación:** 2025-01-15
**Archivos clave:** `src/mztrain/checkpoint.py`

Las activaciones intermedias se comprimen durante el forward
(MNEME ZSpace si disponible, fallback INT8 con escala por tensor) y
se descomprimen on-demand durante el backward, ahorrando memoria de
activaciones a cambio de recomputación.

## Invención 12 — Refactorización periódica que invalida el sleep mask

**Fecha de primera publicación:** 2026-05-15
**Archivos clave:** `src/mztrain/refactorize.py`,
`ElasticRankController.reset_after_refactorize()`

`refactorize_model` reconstruye `U/S/V` sobre una base SVD nueva. Si
ElasticRank sólo aplicara su mecanismo de gracia, el `sleep_mask`
anterior congelaría direcciones equivocadas (porque `mask_gradients`
corre por step, no gated por grace). La invención introduce un
**reset coordinado** que limpia `sleep_mask`, `wake_gate` y `_states`
del controlador (manteniendo el sleep bank para reutilización
futura) tras cada refactorización.

---

## Conclusión técnica honesta (no se oculta)

Los autores documentan, como parte de su trabajo y para fortalecer la
robustez del prior art, las siguientes observaciones empíricas que
delimitan el alcance real de las invenciones:

- **ElasticRank v1 + v2** son inertes durante el entrenamiento
  desde cero de transformers reales (medido sobre ZCodeBERT 4 capas,
  rangos 48–128, 40 épocas): no existe redundancia per-direction
  explotable, las direcciones permanecen casi-ortogonales y los
  proxies no predicen la utilidad real (Spearman ≈ 0.13). La utilidad
  comprobada de ElasticRank se concentra en regímenes
  **approximately-low-rank-weight** (fine-tune desde un checkpoint
  de bajo rango, compresión de modelo ya entrenado al estilo MNEME).
- **El probe de sensibilidad de pérdida y el loss-guard como núcleo
  de decisión** son por tanto las invenciones de mayor valor
  general, ya que prescinden de proxies no predictivos y operan
  sobre la señal directa.
- **VRAM Governor MVP** está validado en GPU; el resto del roadmap
  (precision autopilot, byte-level rank budget, multi-objective
  scoring) se declara como propuesta de diseño no implementada y
  por tanto **no se reivindica como prior art reducido a la
  práctica** salvo por su descripción en este documento.

Esta transparencia es deliberada: la publicación defensiva es más
fuerte si describe con precisión lo que SÍ funciona, lo que NO, y por
qué — porque eso es lo que impide a terceros patentar refinamientos
obvios.

---

## Anclaje temporal recomendado

Para reforzar la fecha cierta de esta publicación, se recomienda:

- Commit firmado GPG conteniendo este archivo y el `SEAL.json`.
- Timestamp RFC 3161 sobre `SEAL.json`.
- Publicación en Software Heritage / Zenodo (con DOI).
- OpenTimestamps sobre el hash SHA-256 de `SEAL.json`.
