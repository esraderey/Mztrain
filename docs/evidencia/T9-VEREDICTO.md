# T9 — Veredicto: las tres banderas abiertas de ElasticShape v1

**Fecha:** 2026-08-21. **Preregistro:** `PREREGISTRO-T9-banderas.md`, escrito antes de los datos
(con la calibración del banco declarada dentro). **Datos:** `t9_results.json` (26 runs, 0 NaN).
Protocolo del arco: char-WT2, batch 16, seq 256, AdamW lr 3e-4 wd 0.01 sin schedule, referencia
de 4000 pasos; reloj de TRAIN puro **con la cirugía dentro**; `val_bpc_full` sobre todo el val
set **siempre en fp32**. Módulo bajo prueba: `src/mztrain/elastic_shape.py` de la v1.3.2 sellada,
importado tal cual desde el paquete.

## Resumen

| Bandera | Regla preregistrada | Resultado | Veredicto |
|---|---|---|---|
| **A — escala** (endpoint 38.8M, 5.1× T8) | ratio ≤ 0.7, todos los seeds alcanzan Q | **0.476** (3/3) | **CONFIRMADO A ESCALA** |
| **B — encadenado** (2 cirugías) | ratio_M2 ≤ 1.15·ratio_M1, todos alcanzan Q | 0.496 vs 0.574 → **0.864** | **H-CHAIN SOBREVIVE** (encadenar es *mejor*) |
| **C — bf16** | ratio ≤ 0.7, \|ΔQ\| ≤ 0.136, sin NaN | ratio **0.689**, ΔQ **0.0195**, 0 NaN | **CONFIRMADO** (por poco) |

**26 runs, 0 NaN, 15/15 morphs alcanzaron Q, 0 exclusiones por estancamiento.** 3.6 h de GPU
(RTX 4060), dentro de la regla de parada de 4 h declarada. Las tres banderas que T8-VEREDICTO
dejó explícitamente abiertas quedan cerradas en positivo.

Además, **réplica independiente de T8** sobre un banco reconstruido desde cero: ratio 0.574
(T8: 0.541) con 4/4 seeds. El claim v1 no dependía de aquel harness perdido.

## T9-A — Escala (S2: d=768, L=8, h=8; endpoint F_384@768 = 38 842 368 params)

Referencia R: **Q = 2.6375**, **T_R = 852.5 s** (3/3 seeds válidos, ninguno estancado; BPC
{2.6472, 2.6562, 2.6091}). Encaja con T4 a esta escala: F_128@768 daba 2.9003 y el denso
completo 2.5946 — nuestro rango 384 cae en medio, donde debe.

| Seed | deriva cirugía | ΔBPC pre/post | T_M (alcanza Q) | BPC a reloj = T_R | BPC final |
|---|---|---|---|---|---|
| 0 | 1.50 % | +1.32 % | 388.6 s | 2.1654 | 2.1188 |
| 1 | 0.18 % | **−0.08 %** | 437.6 s | 2.1617 | 2.1071 |
| 2 | 1.21 % | +0.74 % | 391.3 s | 2.1545 | 2.1096 |

- media(T_M) = 405.8 s → **ratio = 0.476 ≤ 0.7 → CLAIM CONFIRMADO A ESCALA**, 3/3 seeds.
- **El claim MEJORA con la escala:** 0.574 a S1 (11M-equiv) → 0.476 a S2 (38.8M). El mecanismo
  estaba preregistrado como predicción y se midió: la fase barata cuesta **0.490** de la ancha a
  S1 pero **0.421** a S2 (26.5/54.0 vs 88.7/210.8 ms/paso). El mismo número de pasos baratos
  compra más ventaja cuanto mayor es el endpoint.
- **A reloj igual (852.5 s) el morph rinde 2.154–2.165 contra Q = 2.6375:** ventaja permanente
  de ~0.48 BPC.
- **La deriva de la cirugía NO crece con d** (0.18–1.50 % a d=768 frente a 0.69–2.23 % a d=384),
  que era el riesgo explícito declarado en el preregistro. El seed 1 incluso *mejoró* al operar.
  Coste de reloj de la cirugía ≈1.0 s sobre runs de ~1000 s (0.1 %).
- VRAM pico del endpoint: 2.79 GB — lejos del acantilado WDDM (~7 GB) medido en T5.

## T9-B — Growths encadenados (S1 fp32; endpoint F_192@384 = 7 620 096 en ambas ramas)

Referencia R: **Q = 2.8801**, **T_R = 208.7 s** (4/4 válidos; BPC {2.9792, 2.8114, 2.8224,
2.9074}). T8 midió Q=2.9021 / T_R=194.1 s: reproducido dentro del ruido.

| Condición | T_M por seed (s) | media | ratio |
|---|---|---|---|
| M1 simple (2000 pasos densos → 1 cirugía 192→384) | 114.6 / 114.3 / 121.1 / 129.1 | 119.8 | **0.574** |
| M2 encadenado (1000 → cirugía 192→288 → 1000 → cirugía 288→384) | 93.8 / 115.7 / 101.5 / 102.9 | 103.5 | **0.496** |

`ratio_M2/ratio_M1 = 0.864` ≤ 1.15 → **H-CHAIN sobrevive**. Y no solo "no rompe nada":
encadenar alcanza la calidad objetivo **14 % antes** que la cirugía única y termina con mejor
BPC final (media 2.219 vs 2.241). 8/8 morphs alcanzaron Q.

**Mecanismo (medido, no conjeturado).** El encadenado entra a la forma final más tarde pero
mucho más adelantado:

| | entra al endpoint en | con BPC | cruza Q en |
|---|---|---|---|
| M1 simple | t = 50–58 s | 3.571–3.598 | 119.8 s |
| M2 encadenado | t = 68–75 s | **3.092–3.165** | 103.5 s |

Gasta ~20 s más de reloj en formas baratas y llega **~0.45 BPC por delante**. Es la premisa
T4/T6 (lo chico-denso compra más calidad por segundo temprano) aplicada de forma **recursiva**:
cada peldaño intermedio vuelve a comprar barato. Sugiere que el óptimo no es "un salto" sino una
**escalera de crecimiento** — hipótesis nueva, NO probada aquí (v1 no barre schedules).

**Coste de la cirugía:** 0.18–0.38 s por evento (≈0.2 % del reloj). Deriva de logits por evento
0.69–2.23 %; ΔBPC relativo pre/post 0.00–0.95 %. El seed 3 dio la deriva más alta (2.23 %), por
encima de la banda 0.24–1.4 % que reportó T8: **la banda honesta sobre modelos entrenados es
0.2–2.3 %**, no la de T8.

## T9-C — bf16 (S1; autocast en train, params fp32, eval fp32)

Referencia bf16: **Q = 2.8996**, **T_R = 146.8 s** (4/4 válidos). Morph simple: T_M
{90.8, 105.0, 101.9, 106.8} → media 101.1 s → **ratio 0.689 ≤ 0.7 → CONFIRMADO**, 4/4 alcanzan Q.

- **H-BF16-calidad:** \|Q_bf16 − Q_fp32\| = **0.0195** ≤ 0.136 → **PARIDAD**. Cierra en positivo
  la bandera F-grande-bf16 que T6-C4 dejó abierta, al menos a esta escala.
- **Speedup de reloj:** **1.422×** en la referencia — por encima del 1.3× que T6-C4 consideraba
  improbable a S1 por overhead de modelo chico.
- **0 NaN.** La cirugía es idéntica a la de fp32 (deriva por seed 1.71/1.37/1.13/2.23 %, los
  mismos valores que su gemelo fp32 a tres decimales), como debe ser: los parámetros son fp32 y
  el autocast solo afecta al forward/backward de train.

**Hallazgo mecánico (el que importa).** bf16 no acelera las dos fases por igual:

| fase | fp32 | bf16 | speedup |
|---|---|---|---|
| densa d=192 (barata) | 26.50 ms/paso | 23.86 | 1.111× |
| ancha F_192@384 (cara) | 54.04 ms/paso | 39.51 | **1.368×** |

bf16 acelera preferentemente **la fase que el morph intenta evitar**, así que le come ventaja
relativa: el ratio pasa de 0.574 (fp32) a 0.689 (bf16). **Parte del beneficio del morph es un
artefacto de la ineficiencia de fp32 en matmuls grandes.** Corolario honesto: cuanto mejores
sean los kernels de la fase ancha (bf16 hoy, fp8 mañana), menor será el margen del morph. El
claim sigue vivo en bf16 — pero con menos aire: 0.689 contra un umbral de 0.7.

## Hallazgo transversal: la varianza a seed FIJO supera a la varianza entre seeds

Durante la calibración, `R/S1/seed 0` dio **3.3970** (estancado en la meseta). El mismo seed, el
mismo código y el mismo generator, ejecutados de nuevo dentro del driver, dieron **2.9792**. La
diferencia son kernels CUDA no deterministas amplificados por una **bifurcación**: el char-LM se
sienta en una meseta unigrama/bigrama ~3.4–3.6 y "rompe" a ~2.6–2.95 en un paso incierto; a 4000
pasos, unos runs rompen y otros no.

Consecuencias para el arco entero, no solo para T9:

1. **σ₁ = 0.068 (T4) subestima la incertidumbre real.** Esa σ se midió entre seeds *que todos
   rompieron*. La dispersión real run-a-run cerca de la bifurcación es de ~0.4 BPC.
2. Los datos publicados ya lo contenían sin que se nombrara: T6 `lr/F_64/0.0012` dio
   {2.6543, **3.5684**, 2.6539} — un seed estancado entre dos que rompieron.
3. Las comparaciones de T9 son **inmunes por construcción**: Q sale de 3–4 referencias y la
   regla anti-estancamiento es asimétrica (excluye R estancados, que inflarían Q y favorecerían
   al morph; nunca excluye un M). En este experimento no hubo que excluir a nadie.
4. **Recomendación:** cualquier claim del arco medido a 4000 pasos en esta tarea debería
   reportar ≥3 réplicas *del mismo seed*, o declarar que la métrica es "BPC a paso fijo con
   bifurcación abierta".

## Alcance honesto de T9

Válido para: char-WT2; LR 3e-4 (T6-C1 mostró que 1.2e-3 da mucha mejor calidad — no se tocó para
no romper comparabilidad con T4/T6/T8); un schedule por brazo, sin tuning; 4 seeds a S1 y 3 a
S2; autocast bf16 con params fp32. **No probado:** el cruce escala×encadenado×bf16; escaleras de
más de dos peldaños; otros repartos de pasos entre fases; otras tareas; fp8; distribuido. El
endpoint es el MISMO modelo en R y en M en los tres brazos, verificado al parámetro
(7 620 096 en S1; 38 842 368 en S2).

## Estado de ElasticShape tras T9

M1 certificado (21 tests) · M2 certificado (19 tests) · M4 integrado en v1.3.x · T8 claim
confirmado a 11M-equiv · **T9: confirmado a 38.8M, con growths encadenados y en bf16.**
Lo que sigue, si se quiere: la **escalera de crecimiento** (≥3 peldaños, barrido de schedule),
que es la hipótesis que T9-B abrió y no cierra.
