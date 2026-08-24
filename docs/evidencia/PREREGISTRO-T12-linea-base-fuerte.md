# PREREGISTRO — T12: ¿aguantan los números de T9 contra una línea base con rampa?

**Fecha:** 2026-08-22, ANTES de correr ningún run de T12.
**Precondición:** T11 demostró que una rampa de LR de 200 pasos mejora la referencia desde cero
—a LR 1.2e-3, de Q = 2.6182 a Q = 2.4536— y que el ratio de T10 recalculado contra esa línea
base fuerte pasa de 0.442 a 0.542. T11 declaró la amenaza que esto abre: **T8 y T9 midieron
todos sus ratios contra referencias sin rampa.**

## Pregunta

¿El ratio 0.574 de T9-B sobrevive cuando la referencia recibe el mismo trato de LR que recibe el
morph? A 3e-4 la rampa debería importar menos que a 1.2e-3 —una tasa baja no necesita que la
suavicen— pero eso es una expectativa, y las tres últimas expectativas del autor en esta serie
salieron falsas.

## Diseño

S1, endpoint `F_192@384`, **LR 3e-4** (el del arco), fp32, semillas {0,1,2,3} — las mismas de
T9-B. Completa el 2×2 a tasa baja, igual que T11 lo completó a tasa alta:

| | sin rampa | con rampa |
|---|---|---|
| **Desde cero** | *(T9-B, ya medido: Q = 2.8801, T_R = 208.7 s)* | **R_warm_lo** ← nueva |
| **Morph** | *(no se mide: fuera de la pregunta)* | *(T9-B: rampa solo tras la cirugía)* |
| **Morph con rampa también al inicio** | — | **M_warm_lo** ← nueva |

- **R_warm_lo:** `F_192@384` desde cero con `LrWarmup(200, suelo 0.1)` al inicio. Define la línea
  base fuerte a 3e-4: Q' y T_R'.
- **M_warm_lo:** el morph de T9-B con una rampa **también al inicio** de la fase densa, además de
  la que ya lleva tras la cirugía. Si se le da la rampa a la referencia, dársela también al morph
  es lo simétrico; sin esta celda, la corrección penalizaría solo a un lado.

Los morphs de T9-B se reutilizan tal cual para recalcular su ratio contra Q' y T_R'.

## Reglas de decisión

- **LOS NÚMEROS DE T9 AGUANTAN:** ratio de los morphs de T9-B contra la línea base fuerte ≤ 0.7,
  con los 4 alcanzando Q'.
- **MUERTE del claim a 3e-4:** ratio > 0.85, o ≥2 morphs no alcanzan Q' dentro de su presupuesto.
- **INCONCLUSO:** 0.7 < ratio ≤ 0.85, o 1 de 4 no alcanza Q'.
- **Secundario descriptivo:** ratio de `M_warm_lo` (el morph también reforzado) contra la misma
  línea base, y si la rampa inicial cambia algo en la fase densa.
- **Secundario declarado:** tasa de degeneración de `R_warm_lo`. A 3e-4 el driver de T9 no produjo
  ninguna referencia estancada en 4/4, así que **no se espera fenómeno que corregir**; se reporta
  como dato, no como veredicto.

## Predicción del autor

Ratio ∈ [0.58, 0.72]. Es decir: espero que el número empeore respecto al 0.574 publicado —porque
la rampa mejorará algo la referencia— pero que se quede en el filo del umbral o justo por debajo.
Confianza **baja**, y lo digo con el historial a la vista: en T10 fallé las dos predicciones y en
T11 la tercera. Si esta también falla, el patrón deja de ser mala suerte y pasa a ser una
propiedad de mis intuiciones sobre este sistema, que es en sí mismo un dato del cuaderno.

Contemplo explícitamente el desenlace incómodo: si el ratio sube por encima de 0.7, **el claim de
T9-B queda inconcluso o muerto** y hay que reescribir el RFC antes de que salga a ninguna parte.
Ese es el motivo de correr esto ahora y no después.

## Límites

Una escala, una tarea, una tasa, 4 semillas, una configuración de rampa. Corrige la línea base de
**T9-B**; los ratios de T9-A (escala S2) y T9-C (bf16) quedan sin corregir por presupuesto, y su
corrección se declara pendiente en el veredicto.

## Presupuesto

R_warm_lo 4 × ~280 s + M_warm_lo 4 × ~370 s ≈ **45 min**. Parada por tiempo a 1.5 h; mínimo 3
semillas por celda para emitir veredicto.

**Artefacto:** `t12_results.json`.
