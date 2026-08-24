# PREREGISTRO — T9: las tres banderas abiertas de ElasticShape v1

**Fecha:** 2026-08-21, ANTES de correr un solo run de T9 (el smoke de mecánica y la
calibración del banco, ambos declarados abajo, ya se ejecutaron).
**Precondición:** T8 confirmó el claim v1 a escala 11M-equiv (ratio 0.541 ≤ 0.7, 3/3 seeds).
**Alcance declarado como NO probado en T8-VEREDICTO:** escala mayor, growths **encadenados**
en entrenamiento real, y **bf16**. T9 ataca exactamente esas tres.

## Estado de partida — calibración del banco (hecho antes, con resultados a la vista)

El harness de T8 se perdió con los scratchpads viejos; T9 corre sobre un banco
**reconstruido** desde el protocolo publicado (T4/T6/T8) que importa el módulo real
`src/mztrain/elastic_shape.py` de la v1.3.2 sellada. Se declara todo lo ya medido:

| Ancla publicada | valor del arco | banco T9 (seed 0) | Δ |
|---|---|---|---|
| S1 D_full d=384 | 2.7362 (T4) | 2.8043 | +0.068 |
| S1 D_small d=192 | 3.0316 (T4) | 3.0506 | +0.019 |
| S1 F_64@384 | 3.3817 (T4) | 3.3459 | −0.036 |
| S1 F_192@384 (= R de T8) | 2.83–2.94 (T6) | **3.3970 (estancado)** | +0.5 |
| S1 F_192@384 seed 1 / seed 2 | 2.8300 / 2.9249 (T6) | 2.8107 / 2.8451 | −0.02 / −0.08 |

Las cuatro condiciones reproducen dentro del ruido de seed (σ₁≈0.068). El seed 0 de
F_192@384 **no** reprodujo: se quedó en la meseta ~3.4. Esto no es un fallo del banco —
es **bimodalidad del fenómeno**, ya presente en los datos publicados del arco
(T6 `lr/F_64/0.0012`: seeds 0 y 2 dieron 2.654/2.654 y el seed 1 se estancó en 3.568).
El char-LM se sienta en una meseta unigrama/bigrama ~3.4-3.6 y "rompe" a ~2.6-2.95 en un
paso incierto. La diferencia entre reproducir y no reproducir es **de qué lado del
paso 4000 cae la ruptura**. Vocab del banco = 1118 (train+val) vs 1013 del arco
(train-only): +40 320 params en el embedding, declarado, sin efecto en las anclas.

## Las tres banderas y sus reglas (fijadas AQUÍ)

Definiciones comunes, idénticas a T8: **R** = endpoint desde cero, 4000 pasos;
**Q** := media de `val_bpc_full` final de los R válidos; **T_R** := media de su reloj de
TRAIN puro; **M** = morph (fase densa → cirugía → fase ancha), con el **mismo endpoint
exacto** que R (verificado al parámetro en el smoke); **T_M** := primer reloj-de-train de M
con `val_bpc_full` ≤ Q. La cirugía **cuenta** dentro del reloj de M. Los evals no cuentan
en ningún lado. `ratio` := media(T_M)/media(T_R).

### Regla anti-estancamiento (asimétrica y a la contra del claim)

Un run **R** con BPC final > **3.20** se declara ESTANCADO (firma inequívoca de meseta:
los runs que rompen caen ≤2.95, los estancados quedan ≥3.2) y se **excluye de Q y T_R**,
porque un R estancado *infla* Q y le regalaría el resultado al morph. Un run **M**
estancado **nunca** se excluye: cuenta como fallo del morph. Los excluidos se reportan.
Por esta regla y por la calibración de arriba, S1 corre seeds {0,1,2,**3**}: el 0 ya se
sabe estancado y se sustituye, con el hecho declarado aquí antes de mirar nada más.

### T9-A — ESCALA (¿el claim sobrevive al pasar de 11M-equiv a ~39M?)

S2 del arco: d=768, L=8, h=8. **Endpoint F_384@768 = 38.84M params** (5.1× el endpoint de
T8). M: fase 1 densa d=384 por 2000 pasos → un evento (`factorize=True, new_d=768`,
noise 1e-3, LrWarmup 200/floor 0.1) → fase 2 hasta 4000 pasos. Seeds {0,1,2} en orden.

- **CONFIRMA a escala:** ratio_A ≤ 0.7 con **todos** los seeds válidos alcanzando Q.
- **MUERTE a escala:** ratio_A > 0.85, o ≥1 seed no alcanza Q dentro del cap de fase 2.
- **INCONCLUSO:** 0.7 < ratio_A ≤ 0.85.
- **Predicción del autor:** ratio_A ∈ [0.40, 0.65] — *mejor* que en T8. Mecanismo: la fase
  densa es relativamente más barata a S2 (81.8 ms/paso vs 195.6 del ancho = 0.42) que a S1
  (26.3 vs 54.8 = 0.48), así que el mismo número de pasos baratos compra más ventaja.
  Confianza media. Riesgo declarado: la deriva de la cirugía podría crecer con d.

### T9-B — GROWTHS ENCADENADOS (¿dos cirugías en vez de una?)

S1, endpoint idéntico (F_192@384). **M2 (encadenado):** denso 192 por 1000 pasos →
cirugía a d=288 → 1000 pasos → cirugía a d=384 → fase final hasta 4000 pasos.
**M1 (control, réplica de T8):** denso 192 por 2000 pasos → una cirugía a 384 → ídem.
Mismo presupuesto de pasos pre-endpoint (2000) en ambos. Seeds válidos de S1.

- **H-CHAIN (encadenar no rompe nada):** M2 alcanza Q en todos los seeds válidos **y**
  ratio_M2 ≤ 1.15 · ratio_M1.
- **MUERTE de H-CHAIN:** ≥1 seed de M2 no alcanza Q, o ratio_M2 > 1.30 · ratio_M1
  (encadenar cuesta caro: dos transitorios y dos warmups no se amortizan).
- **INCONCLUSO:** 1.15 < ratio_M2/ratio_M1 ≤ 1.30.
- **Secundario preregistrado (mecánico):** deriva de logits por evento y ΔBPC relativo
  pre/post por evento. Banda esperada por T8 sobre modelos ENTRENADOS: ≤1.5%.
- **Predicción del autor:** H-CHAIN sobrevive con ratio_M2/ratio_M1 ∈ [0.95, 1.15]; la fase
  intermedia a 288 es más barata que 384 y puede incluso compensar el segundo transitorio.
  Confianza media-baja: es la primera vez que se encadena en entrenamiento real.

### T9-C — bf16 (la bandera F-grande-bf16, abierta desde T6)

S1, mismo diseño que M1/R pero **entrenando bajo `torch.autocast(cuda, bf16)`** con params
fp32 y **eval SIEMPRE en fp32** (regla T6). La cirugía ocurre en fp32 (los params lo son).
R_bf16 y M1_bf16, seeds válidos.

- **H-BF16-claim:** ratio_C ≤ 0.7 con todos los seeds válidos alcanzando Q_bf16.
- **H-BF16-calidad:** |Q_bf16 − Q_fp32| ≤ 2σ₁ = **0.136** (paridad, criterio de T6-C4).
- **MUERTE:** ratio_C > 0.85, o divergencia de calidad > 0.136, o cualquier NaN.
- **Secundario:** speedup de reloj de train bf16/fp32 en R (T6-C4 esperaba <1.3× a S1 por
  overhead de modelo chico).
- **Predicción del autor:** paridad de calidad sí; ratio parecido al de fp32; speedup
  1.10-1.25× a S1. Confianza media-alta en paridad, baja en el speedup.

## Límites declarados de antemano

Un solo schedule por brazo (sin tuning); un solo LR (3e-4, el del arco — T6-C1 mostró que
1.2e-3 da mucha mejor calidad, pero cambiarlo aquí rompería la comparabilidad con T4/T6/T8);
2-3 seeds por condición; una sola tarea (char-WT2); el encadenado se prueba solo a S1 y la
escala solo en fp32 — el cruce escala×encadenado×bf16 queda fuera. El endpoint es el MISMO
modelo en R y en M por construcción en los tres brazos (verificado al parámetro).

## Presupuesto y regla de parada (fijada antes)

Presupuesto autorizado: 2-4 h de GPU (RTX 4060). Coste estimado con las velocidades ya
medidas: brazo B ≈56 min, brazo C ≈33 min, brazo A ≈115 min (3 seeds) → ≈3.4 h. Orden de
ejecución: **B → C → A** (barato primero; la escala, que es lo caro, al final).
**Regla de parada:** si el reloj total llega a 4 h, el brazo A se reporta con los seeds
completados (mínimo 2 para emitir veredicto; con 1 seed el brazo A queda INCONCLUSO por
presupuesto, no por resultado). La parada es por tiempo, nunca por lo que muestren los datos.

**Artefacto:** `t9_results.json` (dump atómico incremental, reanudable por nombre de run).
