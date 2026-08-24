# T14 — Veredicto: el corpus no era el cuello (y varias lecturas eran artefactos del corte)

> **CRIBADO POSTERIOR (2026-08-22).** Una revisión profunda de toda la serie encontró que
> **ninguna corrida de T14 había convergido en el paso 6000** — todas seguían cayendo con
> pendientes de +0.016 a +0.063 BPC por cada 500 pasos. En consecuencia, **las magnitudes de este
> veredicto son valores en un corte arbitrario, no propiedades de los modelos**, y tres lecturas
> que se dieron por buenas resultaron ser artefactos del corte. Ver la sección «Cribado» al
> final. **El veredicto preregistrado (L2) sobrevive**, porque compara dos corridas en el mismo
> corte y con el mismo protocolo.

**Fecha:** 2026-08-22. **Preregistro:** `PREREGISTRO-T14-datos-vs-escala.md`, escrito antes de los
datos (con una corrección declarada también antes: los dos corpus comparten conjunto de
validación). **Datos:** `t14_results.json` (7 runs) + `t14_controles.json` (4 runs) + celdas de
T9/T10 reutilizadas. Condición morph, fp32, 6000 pasos equivalentes.

## La pregunta

En WT-2 se había medido que **ajustar el LR ganaba más (−0.1464 BPC) que multiplicar los
parámetros por 5.1 (−0.1290 BPC)**. Dos lecturas incompatibles: o el corpus estaba saturado
(2.26 épocas) y por eso los parámetros rendían poco, o rinden poco de verdad.

## El 2×2 en los dos corpus

Misma validación en ambos (1 142 150 caracteres, literalmente el mismo fichero):

| | LR 3e-4 | LR 1.2e-3 |
|---|---|---|
| **WT-2** 7.6M | 2.2408 (n=4) | **2.0944** (n=6) |
| **WT-2** 38.8M | **2.1118** (n=3) | 2.2007 (n=1) |
| **WT-103** 7.6M | 2.2460 (n=2) | **2.0792** (n=2) |
| **WT-103** 38.8M | **2.1158** (n=1) | 2.1845 (n=1) |

## Veredicto: L2 CONFIRMADA — los parámetros rinden poco de verdad

| Magnitud | WT-2 | WT-103 | cociente |
|---|---|---|---|
| Ganancia por 5.1× parámetros (a LR 3e-4) | +0.1290 | +0.1301 | **1.01×** |

La regla preregistrada exigía **≥1.5×** para concluir que el corpus limitaba. Salió **1.01**.
Once veces más datos, cubriendo 6 958 artículos distintos en vez de unos cientos, no movieron la
ganancia por parámetros ni un 1 %.

**La predicción del autor falla otra vez** (séptima de la jornada): esperaba
`G_param(WT-103) ∈ [0.20, 0.35]`. El error es el de siempre — transferir a este régimen la
intuición de las leyes de escala, medidas en corpus grandes con tokenización BPE.

## El hallazgo que no se buscaba: la tasa óptima cambia de signo con la escala

| Ganancia por pasar de 3e-4 a 1.2e-3 | WT-2 | WT-103 |
|---|---|---|
| en 7.6M | **+0.1464** | **+0.1668** |
| en 38.8M | **−0.0889** | **−0.0687** |

En el modelo chico, subir la tasa gana mucho. En el grande, **pierde**. Es el comportamiento
esperable —la tasa óptima se encoge al crecer el modelo— pero obliga a **corregir cómo se venía
formulando el resultado en esta sesión**: se llamaba «el LR ajustado» al 1.2e-3 apoyándose en
T6-C1, que solo barrió la escala S1. Para 38.8M ese valor no es el ajustado, es el pasado de
vueltas.

La comparación corregida, con **cada escala en su mejor LR de los dos probados**:

| Corpus | mejor de 7.6M | mejor de 38.8M | gana |
|---|---|---|---|
| WT-2 | **2.0944** | 2.1118 | el chico, por 0.0174 |
| WT-103 | **2.0792** | 2.1158 | el chico, por 0.0366 |

El modelo cinco veces menor gana en los dos corpus. El resultado no era un artefacto de comparar
contra una configuración mal elegida: se sostiene comparando lo mejor contra lo mejor.

## Lo que limita de verdad: el presupuesto de cómputo

Con 24 576 000 tokens:

- el modelo de 7.6M tiene el **16 %** de los tokens que pediría Chinchilla (20 tok/param);
- el de 38.8M tiene el **3 %**.

El modelo grande no es peor: está **mucho más infra-entrenado** que el chico, y por eso pierde.
Esa es la explicación que reconcilia todo lo demás, y corrige el diagnóstico inicial de esta
sesión, que culpaba al corpus.

## Controles: ¿era un artefacto del tratamiento de los datos?

**C1 — truncado (sin GPU): DESCARTADO.** Los 120M caracteres usados son el 22.3 % del corpus,
cubren 6 958 de 29 867 artículos (23.3 %, proporcional) y la distancia de variación total entre
ese trozo y el siguiente es **0.0027**: el corte es representativo.

**C2 — vocabulario: EFECTO REAL Y CUANTIFICADO.** WT-103 tiene 2989 caracteres frente a 1118, y
ese softmax mayor cuesta **+0.0118 BPC** (mismo corpus, misma receta, solo cambia el vocabulario:
2.0922 → 2.1040). Consecuencia doble:

1. La afirmación «once veces más datos no cambiaron nada» era **demasiado tajante**. Descontada la
   penalización, más datos ganaron entre **+0.008 y +0.027 BPC** — real, pero un orden de magnitud
   menor que el efecto del LR o el de los parámetros.
2. **El veredicto no cambia:** la penalización es idéntica en todas las celdas de WT-103, así que
   **se cancela en las diferencias internas** (`G_param`, `G_lr`), que son las que usan las reglas.

**C3 — ¿estaba WT-2 saturado?: NO.**

| | épocas | train | val | brecha |
|---|---|---|---|---|
| WT-2 | 2.26 | 2.0543 | 2.0873 | **+0.0331** |
| WT-103 | 0.20 | 2.1148 | 2.0823 | −0.0325 |

Tras 2.26 épocas la ventaja sobre datos vistos es de **0.033 BPC**: no hay memorización que
explique nada. **Defecto de este control, declarado:** el BPC de entrenamiento se midió sobre el
*primer trozo* del corpus, no sobre ventanas aleatorias de todo él, así que mezcla memorización
con dificultad intrínseca del trozo — de ahí la brecha negativa en WT-103, donde el modelo vio ese
trozo 0.2 veces. La medición limpia usaría el mismo muestreo que el entrenamiento. Con esa
reserva, el número de WT-2 es lo bastante pequeño como para sostener la conclusión.

## Consecuencia práctica

Para la tabla de capacidad de esta máquina: entrenar el modelo de 118M en los 2.4 días medidos lo
dejaría en el **3 %** de los tokens que ese tamaño pide. **Antes de subir de escala conviene
gastar el presupuesto en más tokens sobre un modelo menor.** Y la tasa de aprendizaje debe
re-barrerse en cada escala: heredarla de una escala menor es un error medido, no teórico.

## Alcance honesto

Dos escalas, dos tasas (sin barrido fino), tokenización de caracteres, 1-2 semillas por celda,
6000 pasos, fp32. Las celdas de 38.8M tienen **una sola semilla** en tres de los cuatro casos, así
que sus diferencias individuales (~0.07) están por encima del ruido esperado pero no medido. El
truncado a 120M caracteres elimina la saturación sin decir nada sobre corpus mayores. No se probó
ninguna tasa intermedia, que es donde estaría el óptimo de cada escala.

**Artefactos:** `t14_results.json`, `t14_controles.json`, `log_T14.txt`, `log_T14ctrl.txt`,
`t14.py`, `t14_controles.py`.

## Cribado: qué encontró la revisión profunda

**Integridad mecánica: 72 runs de T9-T14, 0 fallos.** Curvas consistentes, `final_bpc` igual al
último punto, pasos y relojes monótonos, cero NaN, `T_M` recalculable desde las curvas, params
correctos en todas las escalas y vocabularios. El problema no estaba en el código ni en los datos.

**Inconsistencia detectada, CORREGIDA Y RE-MEDIDA.** T14 introdujo un `val_bpc` propio con un tope
de 4000 ventanas, mientras que T9/T10/T11/T13 evalúan las **4461** del conjunto completo.

El arreglo fue de causa raíz —**una sola métrica en todo el arco**: `t14.val_bpc` *es* ahora
`bank.val_bpc_full`, que acepta corpus en int16 mediante un cast que es no-op sobre int64—, con
un ancla que verifica que el BPC de un modelo fijo es **bit a bit idéntico** antes y después del
cambio (3.8491040135450234), de modo que los 53 runs de T9-T13 quedan intactos.

La única celda que mezclaba métricas en una comparación entre experimentos
(`wt2/S2/lr1.2e-3`, comparada contra valores de T9) **se volvió a ejecutar** con la métrica
unificada:

| | BPC |
|---|---|
| métrica vieja (4000 ventanas) | 2.1894 |
| **métrica única (4461 ventanas)** | **2.2007** |
| corrección | **+0.0113** |

**Nota de método que merece quedar escrita:** se había estimado esta corrección en +0.0019
midiéndola sobre *otro* modelo. La real es **seis veces mayor**. Asumir que el delta se
transfiere entre modelos es exactamente el error que esta serie lleva siete veces cometiendo, y
aquí lo evitó re-ejecutar en vez de extrapolar. `G_lr(38.8M)` en WT-2 pasa de −0.0776 a
**−0.0889**: el signo y la conclusión no cambian.

**El hallazgo de fondo: todas las magnitudes dependen del corte.** Medias por paso en WT-103:

| paso | G_param (5.1× params) | G_lr en 7.6M | G_lr en 38.8M |
|---|---|---|---|
| 2500 | +0.3631 | +0.5715 | +0.3116 |
| 4000 | +0.2474 | +0.3460 | **+0.0195** |
| 4500 | +0.2112 | +0.2775 | **−0.0035** |
| 6000 | +0.1301 | +0.1668 | −0.0687 |

Y quién va delante, con cada escala en su mejor LR:

| paso | 7.6M | 38.8M | gana |
|---|---|---|---|
| 3000 | 2.4696 | 2.4453 | **el grande** |
| 4000 | 2.2348 | 2.3139 | el chico por 0.0791 |
| 5000 | 2.1341 | 2.1861 | el chico por 0.0520 |
| 6000 | 2.0792 | 2.1158 | el chico por 0.0366 |

**Tres afirmaciones de este veredicto quedan rebajadas a «cierto en el paso 6000»:**

1. *«El modelo de 7.6M gana al de 38.8M»* — cierto a 6000, **falso a 3000**, y el margen se cierra
   monótonamente (0.079 → 0.052 → 0.037). La tendencia apunta a que el grande adelanta si se
   continúa. Esto es coherente con la explicación de infra-entrenamiento —y de hecho la refuerza—
   pero desmiente la formulación categórica.
2. *«La tasa óptima cambia de signo con la escala»* — el signo de `G_lr(38.8M)` se invierte entre
   los pasos 4000 y 4500. A 4000 el LR alto todavía ayudaba al modelo grande. No es una propiedad
   de la escala: es dónde está cada modelo en su trayectoria al cortar.
3. *«5.1× parámetros ganan 0.130 BPC»* — es el valor a 6000; a 2500 eran 0.363.

**Lo que NO está afectado:**

- **El veredicto preregistrado L2.** El cociente `G_param(WT-103)/G_param(WT-2) = 1.01×` compara
  dos corridas en el mismo corte, mismo protocolo y misma validación. La conclusión «el corpus no
  era el cuello» se sostiene entera.
- **Los claims del morph (T8-T12: 0.541, 0.574, 0.476, 0.542).** Su métrica es `T_M`, el primer
  instante en que el morph alcanza la Q **final y preregistrada** de su referencia. Es una carrera
  contra un objetivo fijo: no depende de dónde se corte el morph, solo de cuándo cruza.

**Lección de método:** comparar «BPC final» entre configuraciones que no han convergido mide
quién va por delante en un punto arbitrario, no calidad alcanzable. Para afirmar algo sobre
calidad hace falta o converger, o declarar explícitamente que la magnitud es «a presupuesto fijo
de N pasos» — que es lo que este veredicto debería haber dicho desde el principio.