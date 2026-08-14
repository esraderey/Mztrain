# Peritaje matemático profundo — núcleo de MZTrain (2026-08-10)

Auditoría de **verdad matemática** (¿la ecuación que el código implementa es la correcta?), previa a las
pruebas empíricas. Método: 5 peritos derivaron cada fórmula por ruta independiente y la atacaron con
contraejemplos a mano; sin ejecutar el framework. El director re-verificó las cargas fuertes con
`numpy`/lectura sobre la matemática abstracta (no sobre MZTrain).

## Veredicto

**El núcleo matemático es fundamentalmente sólido.** El forward factorizado, Adam, el adjunto (backward)
FP8, los compresores de gradientes contractivos, y la señal de coherencia de ElasticRank están
**verificados correctos por derivación independiente**. Los defectos se concentran en dos sitios:

1. **Dos errores reales, ambos en paths OPT-IN** (no el default), misma clase de fallo — colapso/expansión
   que diverge a NaN: la compresión RANDOM_K y el optimizer GaLore.
2. **Un patrón de afirmaciones sobrevendidas**: docstrings que prometen más de lo que la fórmula entrega
   (energía espectral, varianza de EPSI, porcentajes de memoria). La matemática mayormente es correcta;
   la **documentación** engaña.

Recuento: **2 alto · 6 medio · 7 bajo/nitpick.** Ninguno en el camino por defecto (ZCompressedAdam +
forward factorizado + sin compresión de gradientes), que quedó verificado sano.

---

## ALTO — errores reales (paths opt-in)

**A1 · RANDOM_K + error-feedback diverge geométricamente** — `gradient.py:182-185` + EF en `:120,137`.
Reescala por `n/k` (insesgado pero **expansivo**: `E‖g−C(g)‖² = (1/p−1)‖g‖²`, factor 9 con p=0.1). El
error-feedback exige un operador contractivo (factor < 1); realimentar un error amplificado hace divergir
el buffer: recurrencia `b_t = c(G+b_{t−1})`, `‖e_t‖ ≈ 3^t`, NaN en ~5–15 pasos. Diverge siempre que
`k/n < 1/2`. **Verificado (numpy):** factor medido 9.02; divergencia geométrica; control sin `n/k` acotado
(aísla la causa). El docstring afirma lo contrario ("contractivo… EF lo mejora"). Profundiza el M8 del
peritaje de código con la tasa exacta. **Arreglo:** quitar el `n/k` (queda contractivo) **o** excluir
RANDOM_K del error-feedback.

**A2 · `ZGaLoreOptimizer` cuantiza `v` (2º momento) con INT8 lineal → colapso `v→0` → paso `1/eps`** —
`projector.py:390`. `ZCompressedAdam` corrige esto con log-cuantización (`_compress_v`); GaLore reusa
`_compress_state` (lineal) para `m` **y** `v`, con `compress_states=True` por defecto. **Verificado
(lectura):** `state['exp_avg_sq'] = self._compress_state(v_val)`. Es la misma clase que los bugs #2/#3, en
otra dimensión, y **no cubierta por sus tests** (el test de round-trip usa input bien condicionado, sin
outlier). **Contraejemplo:** bloque de `v` con `abs_max=4.0`, target `v=0.01` → `q=round(0.3175)=0` →
`v̂=0` → `denom=eps=1e-8` → paso ~1e7× el correcto → divergencia. Se dispara cuando un bloque de 2048 tiene
rango dinámico de `v` > ~254× (frecuente). **Arreglo:** usar log-quant para `v` en GaLore (como
ZCompressedAdam), o no comprimir `v`.

---

## MEDIO — afirmaciones sobrevendidas y una imperfección estadística

**M1 · "Energía espectral" es en realidad el número de condición `1 − σ_min/σ_max`** —
`scheduler.py:332-333`, log `:338`, import muerto `randomized_svd :291`. NO calcula
`E(r)=Σσ_i²/Σσ_j²`; usa solo los dos valores extremos, ignora la forma del espectro. **Contraejemplo:**
`S_A=[1,1,1,1,0.001]→0.999` vs `S_B=[1,0.03,0.02,0.015,0.01]→0.99` — declara la capa PLANA (A, debería
crecer) como "más suficiente" que la EMPINADA (B). Triple sobreventa: fórmula falsa + SVD inexistente +
etiqueta `E(r)` en el log. Verificado leyendo.

**M2 · EPSI preserva la varianza AGREGADA, no la POR-NEURONA** — `layers.py:194-202`, docstring afirma
`Var(y)≈Var(y_dense)`. Un escalar global `alpha` iguala `Σ_j Var(y_j)=‖W‖_F²` (exacto) pero no restaura
filas vaciadas por la truncación. **Verificado (numpy):** `W=[[2,0],[0,1]]` → neurona 0 = 5.0 (+25%),
neurona 1 = **0.0** (colapsa); suma 5=5. La afirmación es exacta para el agregado, falsa por-neurona.
Profundiza el M6 del peritaje de código.

**M3 · ConEF anunciado INT4/12.5%, el código hace INT8/25%** — `gradient.py:9,49-50,57` (claim) vs `:313`
+ `:298` (código INT8). Promesa de memoria inflada 2×. No corrompe la matemática. Verificado leyendo.

**M4 · Precisión de la log-cuantización de `v` sobrevendida** — `optimizer.py:71-73`, afirma "17% relativo";
una sola coordenada al piso `1e-30` (log=−69) infla `abs_max` y lleva el error a ~24% en todo el bloque. La
propiedad anti-colapso SÍ se cumple (verificado: `v=[1e-12,100]` preserva el pequeño); solo la cifra de
precisión es ~4× optimista.

**M5 · `ZSparseFactorizedLinear` fuerza EPSI y no puede desactivarlo** — `layers.py:686+` llama
`super().__init__` sin `epsi_scaling` → el residual sparse se mide contra `alpha·W_r` (contaminado), no
contra `W_r`. El `preserve_map`/`epsi_scaling=False` (fix de M6 del código) **no se propaga** a la variante
sparse; además `ZSparse.grow_rank` re-hace SVD sin EPSI → init y grow con políticas opuestas.

**M6 · Coherencia (ElasticRank v2) usada de forma NO iterativa → sobre-marca conjuntos colineales** —
`elastic_rank.py:490-522`. `coherence_i=R²_i` es correcta **por dirección**, pero mide redundancia contra
TODAS a la vez: en un conjunto dependiente, cada miembro tiene `R²_i→1`. La selección estadística correcta
(VIF) es iterativa (quitar una, recomputar, repetir); el código marca todas en una pasada. **Contraejemplo:**
3 direcciones en un span 2-dim → las 3 con coherencia→1 → podría dormir las 3 (rango real permite quitar 1).
El probe interno (leave-one-out) comparte el punto ciego (cada una da `Δ≈0` sola). **Atrapado por el
loss-guard del engine**, que mide `L(post-compactación del grupo)` (confirmado en la auditoría de código) →
medio; sería alto si el guard usara el probe por-dirección.

---

## BAJO / NITPICK

- **B1** FP8: cuantiza con escala BF16 pero des-escala con FP32 → sesgo de escala ~0.8% (dominado por el
  ruido FP8 e4m3 ~6%). `precision.py:471-483`.
- **B2** `_fp8_quantize` redondea a rejilla ENTERA, no a la rejilla FP8 E5M2 que su nombre afirma
  (insesgado, pero no es el formato anunciado). `precision.py:361-363`.
- **B3** `ZAdaptiveOptimizer` divide la varianza del bloque parcial por `block_size` (padding con ceros
  diluye la media) → paso hasta 4× en el último bloque; casi nunca se dispara (dims múltiplos de 8).
  `adaptive_optimizer.py:139-148`.
- **B4** Fórmula de `alpha` en el docstring lleva una raíz cuadrada espuria (código correcto). `layers.py:141,187`.
- **B5** `reconstruct_weight`/`refactorize` usan `S` crudo, ignoran `wake_gate` → la re-SVD resucita
  direcciones atenuadas; solo con ElasticRank activo (no-default). `layers.py:399-402`.
- **B6** "leverage estadístico" es realmente VIF `[R⁻¹]_ii` (no el hat-leverage `h_ii`); solo nombre.
  `elastic_rank.py:365-367`.
- **B7** `wake_score`: `update_at_sleep` no acotado desbalancea el 0.6/0.4 nominal; afecta solo el ORDEN de
  revivir (loss-guard-safe). `elastic_rank.py:153-162`.

---

## Superficie verificada como CORRECTA (lo que las pruebas PUEDEN confiar)

Derivado por ruta independiente y confirmado:

- **Forward factorizado = exactamente `x·Wᵀ+b`.** Convenciones `V:(r×in)`, `U:(out×r)`, cadena
  `x·Vᵀ·diag(S)·Uᵀ = x·Wᵀ`; verificado con matriz no-cuadrada 2×3; bias una vez; broadcast de `S` correcto.
- **`ZCompressedAdam.step` = AdamW canónico línea a línea**: `eps` FUERA del `√`, weight-decay decoupled,
  bias-correction por-parámetro, signo de descenso. Idéntico a `torch.optim`.
- **Adjunto (backward) FP8 correcto**: `grad_x=grad_y·W`, `grad_W=grad_yᵀ·x` (transposiciones verificadas
  con 2×2). Escala FP8 forward correcta (cuantiza dividiendo, des-escala multiplicando la misma).
- **Las 5 políticas del scheduler** (CONSTANT/LINEAR/EXPONENTIAL/COSINE/ADAPTIVE): monótonas, límites y
  signo de ADAPTIVE correctos.
- **Compresores de gradientes contractivos**: TOP_K (δ=k/n), 1-bit (escala óptima L2 `‖g‖₁/b`), INT8
  (round-trip acotado), SVD (Eckart-Young). EF con orden canónico correcto para los 4 métodos contractivos.
- **Coherencia ElasticRank v2 = R²_i** (coeficiente de determinación), ∈[0,1), señal en la dirección
  correcta (redundante→1, ortogonal→0); ridge evita NaN; AND v1 correcto; EMAs y probe apples-to-apples.
- **INT8 lineal de `m`** (con `·127`, bugs #2/#3 corregidos); **GaLore proyección** ortonormal consistente;
  **APOLLO** `v` correcto y NO comprimido (no sufre A2); **`estimate_memory_savings`** (confirma el fix de
  bias B3 del peritaje de código).

---

## Implicaciones para "antes de iniciar pruebas"

- **El camino por defecto es matemáticamente confiable** para los experimentos: `ZCompressedAdam`, forward
  factorizado, sin compresión de gradientes (o TOP_K/1-bit/INT8), scheduler. La matemática está verificada.
- **EVITAR o arreglar antes de experimentar:** `gradient_compression=RANDOM_K` (A1) y `optimizer=GaLore` con
  `compress_states=True` (A2) — ambos divergen a NaN. Son los dos únicos errores de corrección.
- **No confiar en las cifras de los docstrings** al diseñar experimentos: la "energía espectral" es
  condición (M1), EPSI preserva energía agregada no varianza por-neurona (M2), la memoria de ConEF es 25%
  no 12.5% (M3), la precisión de log-quant es ~24% no 17% (M4).
- **ElasticRank**: la señal de coherencia es correcta; su única debilidad (M6, sobre-marcado de conjuntos
  colineales) depende del loss-guard del engine para la seguridad — un experimento con ElasticRank debe
  confirmar que el guard mide la pérdida post-compactación (lo hace) y no fiarse solo del probe.

**Destino:** A1 y A2 son errores de corrección → forja (con las anclas: divergencia de RANDOM_K, colapso de
`v` en GaLore). El resto son afirmaciones a corregir en docstrings + fragilidades acotadas.
