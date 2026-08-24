# PREREGISTRO — T13: promediar los pesos justo antes de la cirugía

**Fecha:** 2026-08-22, ANTES de correr ningún run de T13.
**Origen:** idea del usuario («aprendizaje predictivo: ir guardando checkpoints y usarlos»).
La versión hacia adelante —extrapolar la trayectoria— quedó **falsada** en una prueba previa
(`extrapolacion_results.json`): saltar en la dirección reciente empeora el BPC monótonamente con
α, en los tres regímenes probados (fase densa +0.078, transitorio post-cirugía +0.093,
estabilizado +0.247 ya con α=0.5). La versión hacia atrás —promediar— sí funcionó en una sonda
puntual: **la media de los dos últimos checkpoints dio −0.0431 BPC**, equivalente a entre 50 y
100 pasos de entrenamiento real, a coste cero.

## Hipótesis

Promediar los pesos **justo antes de la cirugía** le regala al morph esos pasos gratis en el
momento en que más caros son. El ensanchado preserva la función de forma exacta, así que una
mejora en θ antes de operar se propaga íntegra al modelo crecido.

**Por qué este punto y no otro:** la objeción estándar al promediado es que deja obsoleto el
estado del optimizador. Aquí no aplica — la conversión densa→factorizada **resetea el optimizador
de todos modos** (parametrizaciones no conmensurables, decisión documentada en la SPEC). El
promedio sale gratis precisamente aquí.

## Diseño

S1, LR 1.2e-3, endpoint `F_192@384`, fp32, 6 semillas {0..5}, protocolo de T10/T11.

- **M_ctl:** morph estándar (2000 pasos densos → cirugía → rampa 200/0.1 → fase ancha).
- **M_avg:** idéntico salvo que en el paso 2000, **antes** de operar, θ se sustituye por la media
  de los snapshots de los pasos 1900 y 2000. El coste del promedio se cuenta en el reloj.

**Ambos brazos corren en esta misma sesión, intercalados por semilla** (ctl/s0, avg/s0, ctl/s1,
avg/s1, …). Es la corrección que impone T12: el reloj deriva hasta un 10.6 % entre sesiones por
térmica, y una comparación pareada e intercalada la neutraliza.

`Q = 2.4536` se hereda de la línea base fuerte de T11 (es un valor de calidad, no un reloj: la
deriva térmica no lo afecta). `T_R` **no** se hereda — el veredicto se emite sobre la diferencia
pareada entre los dos brazos, no sobre un ratio contra una referencia de otra sesión.

## Reglas de decisión (pareadas)

Sea Δᵢ = T_M(avg, semilla i) − T_M(ctl, semilla i), y g = −media(Δᵢ)/media(T_M(ctl)) la ganancia
relativa media.

- **CONFIRMA:** ≥5 de 6 semillas mejoran (Δᵢ < 0) **y** g ≥ 3 %.
- **MUERTE:** ≤2 de 6 mejoran, **o** g ≤ 0.
- **INCONCLUSO:** cualquier otro patrón.
- **Secundarios preregistrados:** (a) ¿mejora el BPC en el instante de promediar? — se mide antes
  y después del promedio, y la hipótesis exige que mejore en ≥5/6 semillas; (b) deriva de la
  cirugía y BPC post-cirugía en ambos brazos; (c) BPC final a 6000 pasos.

## Predicción del autor

**Espero INCONCLUSO, y probablemente muerte por umbral.** El razonamiento es aritmético: la
sonda midió que el promedio vale 50-100 pasos, y la fase ancha del morph tarda ~2750-3250 pasos
en alcanzar Q. Un ahorro de ~75 pasos es un **2-3 %**, justo por debajo del umbral de
confirmación del 3 %. Para que esto confirme, el promedio tendría que rendir bastante más en el
punto pre-cirugía que en el punto donde lo sondeé.

Confianza media-baja — y con el historial del día encima: **cinco predicciones fallidas
consecutivas** (T10 ×2, T11, T12, y la de que el transitorio post-cirugía sería el más
extrapolable). Todas por el mismo error: suponer que un mecanismo transfiere de un régimen a
otro. Esta predicción comete exactamente el mismo pecado —transfiere la magnitud medida en un
punto estabilizado al punto pre-cirugía—, así que conviene desconfiar de ella tanto como de las
anteriores.

Declaro por adelantado el desenlace que me haría cambiar de opinión sobre el método entero: si
g ≥ 10 %, el promediado pre-cirugía deja de ser un truco marginal y pasa a ser parte del
protocolo del morph.

## Límites

Una escala, una tarea, un LR, 6 semillas, una sola configuración de promedio (2 snapshots
separados 100 pasos). No se barre ni el número de snapshots ni su separación, que la sonda
sugiere que importan (la media de 6 ya empeoraba). No se prueba promediar en otros puntos del
entrenamiento ni en la referencia desde cero.

**Artefacto:** `t13_results.json`. **Presupuesto:** 12 runs × ~370 s ≈ 75 min.
