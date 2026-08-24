# PREREGISTRO — T10: el claim bajo tasa de aprendizaje ajustada, y la hipótesis de estabilidad

**Fecha:** 2026-08-22, ANTES de correr ningún run de T10.
**Precondición:** T9 cerró escala, encadenado y bf16 (T9-VEREDICTO.md). El propio T9 declaró en
§7 que la amenaza a la validez más grave seguía en pie: **todo el arco corre con LR 3e-4, que
T6-C1 demostró subóptimo para todas las condiciones.**
**Banco:** el mismo de T9 (`bank.py`), ya calibrado contra las anclas de T4 (Δ ≤ 0.07 BPC).
Módulo bajo prueba: `src/mztrain/elastic_shape.py` de mztrain 1.3.2, sin modificar.

## Las dos preguntas

### Q1 — Supervivencia: ¿gana el morph cuando la línea base está bien afinada?

T6-C1 midió que a LR 1.2e-3 la calidad mejora mucho para todos (D_small pasa de 3.027 a 2.413 y
supera al D_full de T4). El claim de ElasticShape se ha medido siempre en el régimen malo. Si el
morph solo gana porque la referencia estaba mal entrenada, no es un método: es un artefacto.

### Q2 — Estabilidad: ¿la fase densa inocula contra la degeneración a LR alto?

T6-C1 registró que a 1.2e-3 el **factorizado se desestabiliza** (F_64: una semilla de tres
degeneró a 3.5684 frente a 2.6543/2.6539) mientras el **denso aguanta estable** (D_small: 2.391 /
2.401 / 2.447, σ pequeña). El morph empieza denso y termina factorizado. Si hereda la estabilidad
de su primera fase, entonces la técnica no solo ahorra reloj: es **la forma de entrenar
factorizado a LR alto sin que reviente**, que es un beneficio distinto e independiente del
primero.

## Diseño

Todo a escala S1 (endpoint `F_192@384`, 7 620 096 params), fp32, **LR 1.2e-3** salvo donde se
diga, resto del protocolo del arco idéntico (char-WT2, batch 16, seq 256, AdamW wd 0.01 sin
schedule, 4000 pasos de referencia, `val_bpc_full` en fp32, reloj de train puro con la cirugía
dentro). **6 semillas {0..5} por condición** — el doble que T9, porque Q2 es una pregunta sobre
frecuencia de un suceso raro y con 3 semillas no se distingue 1/3 de 0/3.

- **R_hi:** `F_192@384` desde cero, LR 1.2e-3. Da Q y T_R.
- **M_hi:** morph (2000 pasos densos d=192 → cirugía única a 384 → fase ancha), LR 1.2e-3 en
  ambas fases, LrWarmup 200 pasos suelo 0.1 tras la cirugía.
- **Control de replicación (3 semillas):** `F_64@384` desde cero a LR 1.2e-3. Sirve para saber si
  la inestabilidad que midió T6 se reproduce en este banco y si es un fenómeno **de rango bajo**
  o general. No entra en ningún veredicto: es diagnóstico.

## Definición de DEGENERACIÓN (fijada aquí)

Un run degenera si su `val_bpc_full` final es **> 3.20** o si aparece cualquier NaN. El umbral es
el mismo que la regla anti-estancamiento de T9 y separa sin ambigüedad los dos modos observados
en el arco: los runs sanos a este LR caen en 2.3-2.7 y los degenerados en ≥3.5.

## Reglas de decisión

### Q1 — Supervivencia

Con Q y T_R calculados **excluyendo las referencias degeneradas** (regla asimétrica de T9: una
referencia degenerada infla Q y le regalaría el resultado al morph; un morph degenerado nunca se
excluye y cuenta como fallo).

- **SOBREVIVE:** ratio = media(T_M)/media(T_R) ≤ 0.7 con todos los morphs no degenerados
  alcanzando Q.
- **MUERTE:** ratio > 0.85, o ≥2 de 6 morphs no alcanzan Q dentro del tope de fase 2.
- **INCONCLUSO:** 0.7 < ratio ≤ 0.85, o 1 de 6 no alcanza Q.
- **Predicción del autor (honesta):** ratio ∈ [0.55, 0.85], confianza **baja**, con posibilidad
  real de inconcluso o muerte. Razonamiento: a LR alto la fase ancha también aprende mucho más
  rápido, así que la ventaja de los pasos baratos se comprime — el mismo mecanismo que en T9-C
  hizo que bf16 le comiera margen al morph (0.574 → 0.689). Si esa lógica se sostiene, el ratio
  a LR ajustado debería empeorar respecto al 0.574 de T9-B. Esta es la prueba que puede matar el
  claim y así se declara.

### Q2 — Estabilidad

Sea `p_R` la fracción de R_hi degenerados y `p_M` la de M_hi degenerados, sobre 6 semillas cada
una.

- **H-STAB CONFIRMADA:** `p_M < p_R` y `p_R − p_M ≥ 2/6` (al menos dos referencias más
  degeneradas que morphs).
- **MUERTE de H-STAB:** `p_M ≥ p_R` (el morph no protege, o empeora).
- **INCONCLUSO:** `0 < p_R − p_M < 2/6`.
- **NO EVALUABLE:** si `p_R = 0`. La inestabilidad de T6 se midió a rango 64 y aquí corremos
  rango 192, mucho más cerca del denso; es perfectamente posible que el fenómeno no aparezca. En
  ese caso la hipótesis no se confirma ni se mata: **no hay fenómeno que proteger**, y así se
  reporta. El control `F_64@384` dirá si la inestabilidad existe a rango bajo en este banco.
- **Predicción del autor:** `p_R` ∈ {0, 1/6, 2/6} y `p_M` = 0. Confianza baja en la magnitud,
  media en el signo. Creo más probable el desenlace NO EVALUABLE que la confirmación, porque el
  rango 192 puede no ser inestable.

## Límites declarados

Una escala, una tarea, un LR alto (1.2e-3, el mejor del barrido de T6 — no se barre otra vez),
un solo reparto de pasos (50/50, heredado sin ajustar), morph de un solo salto (el encadenado de
T9-B no se cruza con esto), fp32. La comparación de ratios contra T9-B (0.574 a 3e-4) es
**entre experimentos**, con el mismo banco y protocolo salvo el LR, que es la variable
manipulada.

## Presupuesto y regla de parada

Coste estimado con las velocidades medidas en T9: R_hi 6 × ~280 s + M_hi 6 × ~370 s + control
3 × ~150 s ≈ **75 min**. Orden: control → R_hi → M_hi. Si el reloj total llega a 2 h, se reporta
con las semillas completadas (mínimo 4 por condición para emitir veredicto de Q1; menos de 4
deja Q1 inconcluso **por presupuesto**, no por resultado). La parada es por tiempo, nunca por lo
que muestren los datos.

**Artefacto:** `t10_results.json` (volcado atómico incremental, reanudable por nombre de run).
