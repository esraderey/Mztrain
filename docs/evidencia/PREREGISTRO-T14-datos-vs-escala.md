# PREREGISTRO — T14: ¿era el corpus el que limitaba, o de verdad los parámetros rinden poco?

**Fecha:** 2026-08-22, ANTES de correr ningún run de T14.

## De dónde sale la pregunta

Con los datos de hoy sobre char-WikiText-2, tras 6000 pasos equivalentes:

| Modelo | LR | BPC final |
|---|---|---|
| 7.6M (F_192@384) | 3e-4 | 2.2408 |
| 38.8M (F_384@768) | 3e-4 | 2.1118 |
| 7.6M — el mismo modelo chico | 1.2e-3 | 2.0944 |

- Multiplicar los parámetros por **5.1** ganó **−0.1290 BPC**.
- Ajustar la tasa de aprendizaje ganó **−0.1464 BPC**.

El modelo chico bien afinado supera al cinco veces mayor mal afinado. Hay dos lecturas
incompatibles y el experimento existe para separarlas:

- **L1 — el corpus estaba saturado.** WT-2 tiene 10 892 990 caracteres de entrenamiento y 6000
  pasos son **2.26 épocas**. Un modelo mayor no tiene nada nuevo que aprender ahí, así que la
  ganancia por parámetros sale artificialmente baja.
- **L2 — en este régimen los parámetros rinden poco de verdad**, y el resultado se mantendría con
  datos de sobra.

## Diseño

Corpus nuevo: **char-WikiText-103**, truncado a los primeros **120M caracteres** de train (11×
WT-2). Con 6000 pasos × 16 × 256 = 24 576 000 tokens, eso son **0.20 épocas**: el dato deja de
ser el limitante, que es justo lo que se quiere manipular. La validación es la de WT-103.

**2×2 completo en cada corpus** (escala × LR), condición morph (2000 pasos densos → cirugía →
4000 anchos), fp32, protocolo del arco en todo lo demás:

| | LR 3e-4 | LR 1.2e-3 |
|---|---|---|
| **7.6M** `F_192@384` | WT-103 ×2 semillas | WT-103 ×2 semillas |
| **38.8M** `F_384@768` | WT-103 ×1 semilla | WT-103 ×1 semilla |

Además se corre la celda que **falta en WT-2** — 38.8M a LR 1.2e-3, 1 semilla — para que el 2×2
de WT-2 esté completo y las dos tablas sean comparables. Sin ella, la «ganancia por parámetros»
de WT-2 solo está medida a LR bajo.

Definiciones: `G_param(LR) = BPC(7.6M) − BPC(38.8M)` al mismo LR; `G_lr(escala) = BPC(3e-4) −
BPC(1.2e-3)` a la misma escala. Ambas positivas = mejora.

## Reglas de decisión

- **L1 CONFIRMADA (el corpus limitaba):** en WT-103, `G_param ≥ G_lr` en al menos uno de los dos
  LR, **y** `G_param(WT-103) ≥ 1.5 × G_param(WT-2)` al mismo LR. Es decir: con datos de sobra los
  parámetros pasan a rendir sustancialmente más que en WT-2.
- **L2 CONFIRMADA (los parámetros rinden poco de verdad):** en WT-103, `G_param < G_lr` en ambos
  LR **y** `G_param(WT-103) ≤ 1.2 × G_param(WT-2)`.
- **INCONCLUSO:** cualquier otro patrón — en particular si el orden se invierte pero la magnitud
  apenas cambia, o al revés.
- **Secundarios descriptivos:** BPC absolutos en ambos corpus (no comparables entre sí: son
  tareas distintas); si el LR ajustado sigue siendo el mejor en las dos escalas; y si con 0.2
  épocas aparece o desaparece la bifurcación/meseta que domina WT-2.

## Predicción del autor

**Espero que L1 se confirme**: con 11× más datos y un quinto de época, el modelo grande debería
tener margen real y `G_param` debería crecer. Mi mejor apuesta es `G_param(WT-103) ∈ [0.20, 0.35]`
frente a los 0.129 de WT-2, y que supere a `G_lr`.

Confianza **baja**, y por una razón concreta además del historial: la ganancia por LR también
podría crecer en el corpus nuevo, y entonces el orden se mantendría por motivos que no tienen
nada que ver con la saturación. Si eso pasa, el resultado será INCONCLUSO y habrá que decirlo.

Historial del día a la vista: **seis predicciones falladas** (T10 ×2, T11, T12, transitorio
extrapolable, colapso del espectro) y una acertada a medias (T13). El patrón es siempre el mismo
—suponer que un mecanismo transfiere de un régimen a otro— y esta predicción vuelve a hacerlo:
transfiere la intuición de las leyes de escala, medidas en corpus grandes con tokenización BPE, a
un modelo de caracteres diminuto. Desconfíese en consecuencia.

## Límites declarados

Una tokenización (caracteres), dos escalas, dos tasas, 1-2 semillas por celda, 6000 pasos, fp32.
El truncado a 120M caracteres es una decisión de presupuesto: elimina la saturación pero no
prueba nada sobre corpus aún mayores.

**Corrección hecha ANTES de los datos, al montar el pipeline:** este preregistro afirmaba que los
BPC de WT-2 y WT-103 no serían comparables. Es falso — **los dos corpus comparten exactamente el
mismo conjunto de validación** (1 142 150 caracteres, el mismo fichero), así que la tarea de
evaluación es idéntica y solo cambian los datos de entrenamiento. Eso hace la comparación entre
corpus **más informativa** de lo que se había declarado. Queda un caveat menor: el vocabulario de
caracteres difiere (2989 en WT-103 frente a 1118 en WT-2, por la mayor diversidad unicode), lo
que agranda el embedding (8.34M frente a 7.62M params en S1) y añade clases nunca observadas al
softmax. El efecto sobre el BPC final debería ser pequeño, pero no es cero y no se ha medido.

**Artefacto:** `t14_results.json`. **Presupuesto:** ~66 min. Parada por reloj a 1.5 h.
