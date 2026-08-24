# T11 — Veredicto: el mecanismo, y la corrección a T10

**Fecha:** 2026-08-22. **Preregistro:** `PREREGISTRO-T11-mecanismo.md`, escrito antes de los datos.
**Datos:** `t11_results.json` (12 runs nuevos, 0 NaN) + dos celdas reutilizadas de
`t10_results.json`, como el preregistro declaró. LR 1.2e-3, S1, endpoint `F_192@384`, 6 semillas
por celda, mismo banco y protocolo que T10.

## El 2×2

Tasa de degeneración (BPC final > 3.20 o NaN) sobre 6 semillas:

| | sin rampa de LR | con rampa de LR |
|---|---|---|
| **Desde cero** | **4/6** *(T10)* | **0/6** |
| **Morph** | **0/6** | **0/6** *(T10)* |

**Veredicto preregistrado: AMBOS CONTRIBUYEN.** Cada factor por separado basta para eliminar la
degeneración; la protección está **sobredeterminada**. Ni la hipótesis del autor («el mérito es
de la fase densa», que exigía `p(R_warm) ≥ 3/6`) ni su alternativa («el mérito es de la rampa»,
que exigía `p(M_nowarm) ≥ 3/6`) describen lo que ocurre.

La predicción del autor vuelve a fallar, la tercera vez consecutiva en esta serie. Estaba
apoyada en un argumento razonable —la degeneración se decide entre los pasos 2000 y 3000 y una
rampa de 200 pasos actúa mucho antes— y el argumento resultó irrelevante: una rampa temprana
cambia toda la trayectoria posterior, no solo su propia ventana.

## Corrección 1 — El claim de estabilidad de T10 se reescribe

T10 concluyó, con estas palabras, que «en este régimen el morph deja de ser una optimización de
coste y pasa a ser **la forma fiable** de llegar al modelo factorizado». **Eso era falso en la
parte de «la forma».** Una rampa de LR de 200 pasos al inicio —que no cuesta nada, no requiere
cirugía y es práctica estándar— elimina la degeneración por completo: 0 de 6, y además con mejor
calidad final (2.4536 de media contra 2.6182 de las referencias sanas sin rampa).

Lo que **sí** sobrevive de aquel hallazgo, dicho con precisión:

1. **El factorizado entrenado desde cero sin rampa es frágil a LR alto:** 4/6 corridas perdidas,
   sin NaN ni aviso. Esto sigue en pie y es un dato útil por sí mismo.
2. **El morph es insensible a esa decisión de hiperparámetro:** 0/6 con rampa y 0/6 sin ella. La
   referencia, no: 4/6 sin rampa y 0/6 con ella. El morph no es *la única* forma de evitar el
   fallo, pero sí es **la que no depende de acertar con la rampa**.
3. La afirmación fuerte —«el morph protege»— pasa a ser la afirmación débil —«el morph no
   necesita protección»—. Es menos vistosa y es la que aguantan los datos.

## Corrección 2 — El ratio de T10 estaba medido contra una línea base débil

Si una rampa mejora la referencia de 2.6182 a 2.4536, entonces la referencia de T10 no era la
mejor disponible, y el ratio calculado contra ella exageraba la ventaja del morph. El recálculo
contra la **línea base fuerte** (desde cero, LR ajustado **y** rampa; 0/6 degeneradas,
Q = 2.4536, T_R = 213.0 s):

| Comparación | media T_M | ratio |
|---|---|---|
| Morph con rampa, contra línea base fuerte | 115.5 s | **0.542** |
| Morph sin rampa, contra línea base fuerte | 128.5 s | 0.603 |
| *Morph con rampa, contra línea base sin rampa (lo publicado en T10)* | *94.1 s* | *0.442* |

> **Verificado por T12:** este 0.542 mezcla `T_M` de la sesión de T10 con `T_R` de la de T11.
> T12 descubrió que el reloj deriva hasta un 10.6 % entre sesiones por térmica, así que la mezcla
> podía sesgar el número. Se comprobó: las sesiones T10 y T11 difieren un 0.2 % y el ratio
> corregido es **0.543**. Se sostiene — ahora verificado en vez de supuesto.

**El claim de velocidad sobrevive**: 0.542 con 6/6 morphs alcanzando Q, muy por debajo del
umbral de 0.7, y en línea con el 0.574 que T9-B midió a LR 3e-4. Pero **el 0.442 de T10 queda
retirado**: era producto de una referencia mal equipada. El número honesto a LR ajustado es
**0.542**.

La rampa tras la cirugía sí aporta velocidad al morph, aunque no protección: 0.542 con ella
frente a 0.603 sin ella.

## Corrección 3 — «terminó mejor» hay que decirlo a reloj igual

T10 escribió que el morph «llegó antes y terminó mejor». El BPC final del morph (2.0944) se
alcanza tras 6000 pasos y 257.1 s, mientras que la referencia gasta 4000 pasos y 213.0 s: es
**21 % más reloj**, así que comparar los dos BPC finales no es comparar a coste igual.

La comparación limpia, al reloj de la línea base fuerte (213.0 s): el morph va en **2.1276** de
media frente a Q = 2.4536, una ventaja de **0.326 BPC**. La conclusión aguanta; la formulación
de T10 era imprecisa.

## Auditoría del banco (previa a dar T11 por bueno)

Antes de aceptar estos resultados se corrió una auditoría sobre el banco y los datos
(`audit.py`), con el sesgo deliberado de buscar errores que **favorezcan** las conclusiones:

1. La rampa de LR hace lo declarado (suelo 1.2e-4 → base 1.2e-3, monótona, se queda en base).
2. `R_warm` difiere de `R` **solo** en la rampa: diff del código sin más cambios sustantivos.
3. El LR y el weight decay **sobreviven a la cirugía** (`apply_event` construye un optimizador
   nuevo; se verificó que hereda 1.2e-3 y 0.01, no un valor por omisión).
4. Integridad de los 41 runs de T9+T10+T11: `final_bpc` coincide con el último punto de cada
   curva, relojes y pasos monótonos, cero NaN, `T_M` almacenado idéntico al recalculado desde la
   curva, y ningún run marcado como «no alcanza Q» que en realidad cruzara.
5. Asimetría de presupuesto detectada y corregida (sección anterior).
6. El umbral de degeneración **no es determinante** a LR alto: los 27 BPC finales de T10+T11 se
   reparten en 23 runs ≤ 2.678 y 4 runs ≥ 3.597 — margen de **0.919 BPC** sin nada entre medias.
   Cualquier umbral en [2.7, 3.5] clasifica igual.
7. **Matiz que la auditoría destapó y conviene declarar:** ese margen depende del régimen. A
   LR 3e-4 (T9) lo sano vive en 2.81-2.99 y el umbral de 3.20 queda a solo **0.207 BPC** del
   run sano más alto; el único estancamiento observado a esa tasa (3.3970, en la calibración)
   está 0.404 por encima del sano más alto. Sigue habiendo separación, pero **es tres veces más
   fina que a LR alto**. En T9 la regla anti-estancamiento nunca llegó a aplicarse —4/4 y 3/3
   referencias válidas—, así que no afectó a ningún veredicto; queda anotado para quien reutilice
   el umbral a tasas bajas.

No apareció ningún defecto de código que afecte a las conclusiones.

## Amenaza abierta que esto destapa

T8 y T9 midieron sus ratios contra referencias **sin rampa de LR**. A 3e-4 el efecto de una rampa
debería ser mucho menor que a 1.2e-3 —una tasa baja no necesita que la suavicen— pero eso es una
expectativa, no una medición. **Los ratios 0.541 (T8), 0.574 (T9-B), 0.476 (T9-A) y 0.689 (T9-C)
están medidos contra una línea base que ahora sabemos mejorable**, y cuánto los mueve es una
pregunta abierta. Corregirlo cuesta una hora de GPU por experimento y debería hacerse antes de
que esos números viajen a ninguna parte.

## Lo que queda sin explicar

Sabemos **qué** protege; no sabemos **por qué**. Una rampa de 200 pasos con suelo 0.1 al inicio
evita un fallo que se materializa entre los pasos 2000 y 3000, y no hay en estos datos nada que
diga cómo. La hipótesis del autor —que el morph se salta la bifurcación porque entra a la forma
factorizada ya rota, en BPC ~3.05— sigue siendo compatible con todo lo observado, pero T11
demuestra que **no es necesaria**: la referencia con rampa rompe la meseta por su cuenta. Dos
caminos distintos al mismo sitio.

## Alcance honesto

Seis semillas por celda, una escala, una tarea, un LR alto, una sola configuración de rampa (200
pasos, suelo 0.1, sin barrido). Dos de las cuatro celdas son datos de T10 reutilizados: mismo
banco, mismas semillas, mismo hardware y misma sesión, pero no la misma ejecución. El diseño
identifica **cuál de los dos factores lleva el crédito** y responde «los dos»; no mide la
magnitud de ninguno ni descarta un tercer factor. La generalización a otras tasas, escalas o
tareas no está probada.

**Artefactos:** `t11_results.json`, `log_T11.txt`, `t11.py`.
