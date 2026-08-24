# T13 — Veredicto: el promedio funciona, y no sirve para acelerar

**Fecha:** 2026-08-22. **Preregistro:** `PREREGISTRO-T13-promedio-precirugia.md`, escrito antes
de los datos. **Datos:** `t13_results.json` (12 runs pareados, 0 NaN). LR 1.2e-3, S1, endpoint
`F_192@384`, 6 semillas, diseño pareado e intercalado por semilla.

## Resumen

| Pregunta | Regla preregistrada | Resultado | Veredicto |
|---|---|---|---|
| ¿Acelera el promedio pre-cirugía? | ≥5/6 mejoran y ganancia ≥3 % | 2/6, ganancia **−2.58 %** | **MUERTE** |
| ¿Mejora el promedio los pesos? | ≥5/6 mejoran | **6/6**, −0.0175 BPC | **CONFIRMADO** |

El mecanismo que propuso el usuario **existe y es reproducible**; lo que no hace es convertirse
en reloj. Las dos cosas son ciertas a la vez y conviene decirlas juntas.

## El efecto gratis es real

Promediar los pesos de los pasos 1900 y 2000 mejora el BPC en las **seis** semillas:

| Semilla | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Δ BPC al promediar | −0.0187 | −0.0295 | −0.0154 | −0.0085 | −0.0167 | −0.0161 |

Media **−0.0175 BPC**, coste de cómputo nulo (una suma de tensores). Menor que los −0.0431 que
midió la sonda en un punto estabilizado, lo que ya sugería que la magnitud depende del punto del
entrenamiento — otra vez el mismo patrón: los mecanismos no transfieren entre regímenes.

## Y se evapora antes de que sirva de nada

Diferencia de BPC entre el brazo promediado y el control, a pasos emparejados:

| Paso | avg − ctl | semillas en que mejora |
|---|---|---|
| 2250 (250 tras la cirugía) | **+0.0173** | **0/6** |
| 2500 | +0.0070 | 1/6 |
| 2750 | +0.0039 | 1/6 |
| 3000 | −0.0003 | 2/6 |
| 3500 | −0.0048 | 3/6 |
| 6000 (final) | −0.0010 | 3/6 |

La ventaja de −0.0175 no solo desaparece: **se invierte durante los primeros 250 pasos tras la
cirugía**, donde el brazo promediado va peor en las seis semillas. Recupera la paridad hacia el
paso 3000 y termina en un empate dentro del ruido (−0.0010 BPC final, 3/6).

**Mecanismo propuesto** (compatible con lo observado, no probado aquí): el punto promediado tiene
menor pérdida pero **no está sobre la trayectoria del optimizador**. Es el fondo del valle con
velocidad cero; el entrenamiento posterior tiene que reconstruir el momento que la trayectoria ya
llevaba, y eso cuesta más de lo que el promedio regaló. Es exactamente el motivo por el que la
práctica estándar de SWA usa el promedio como **artefacto de evaluación y despliegue**, nunca
como punto desde el que continuar. El preregistro ya declaraba esa advertencia; ahora está medida
en este banco.

Nótese que el reseteo del optimizador —el argumento por el que este punto parecía ideal, ya que
la conversión densa→factorizada descarta el estado de Adam de todos modos— **no salva el
problema**. El estado se pierde igual en ambos brazos; lo que penaliza al promediado es la
posición de θ, no el momento almacenado.

## Un fallo de diseño que conviene declarar

`T_M` está cuantizado a la cadencia de evaluación: 250 pasos ≈ 14 s. El efecto que se buscaba,
estimado a partir de la sonda, era de ~75 pasos ≈ 4 s. **La resolución del instrumento era tres
veces más gruesa que el efecto**, y eso lo hace visible en los datos: cinco de las seis semillas
cruzaron Q en el mismo paso (3250) en ambos brazos, y la sexta (s4) decidió el signo de la media
entera por caer un eval antes en el control (+15.8 s de los +16.4 s totales).

Excluyendo esa semilla la ganancia media seguiría siendo negativa (+0.33 s de retraso medio), así
que el veredicto no cambia. Pero la lección de método sí queda: **para efectos pequeños, la
métrica de cruce discreto no vale; hay que medir la diferencia continua entre curvas**, que es lo
que se hizo en la sección anterior y lo que produjo el diagnóstico útil.

## Predicción del autor

Predije «INCONCLUSO, y probablemente muerte por umbral», razonando que la ganancia sería positiva
pero demasiado pequeña (2-3 %) para cruzar el 3 %. **El veredicto acertó y el razonamiento no**:
la ganancia no fue pequeña y positiva, fue **negativa**. Es la primera vez en la serie que el
desenlace cualitativo coincide con lo predicho, tras cinco fallos; pero el mecanismo previsto era
otro, así que cuenta como acierto a medias.

También queda declarado que no se cruzó el umbral que yo mismo fijé para cambiar de opinión sobre
el método (ganancia ≥10 %): no está ni cerca.

## Diagnóstico posterior: qué mata exactamente al promedio (n=1, identifica mecanismo)

El veredicto de arriba atribuyó el fallo a «el fondo del valle con velocidad cero». Eso era una
hipótesis; al medirla (`diagnostico_promedio.py`, una semilla) resultó **imprecisa**, y una
segunda hipótesis del autor —colapso del espectro— quedó **falsada**.

**Hipótesis del espectro: FALSADA.** Se conjeturó que promediar cancelaría las direcciones
oscilantes, hundiendo la cola de valores singulares y dejando esas direcciones casi congeladas al
factorizar (el gradiente de U y V escala con S). Los números lo desmienten: en el punto
pre-cirugía, s_min pasa de 0.33927 a 0.33901, el rango efectivo de 123.50 a **124.04** (sube) y
la energía de la cola del 1.978 % al 1.987 %. El espectro no se mueve.

**Lo que sí se mueve: el gradiente.** ‖grad‖ en el punto pre-cirugía: 1.8782 (original), 1.4109
(promedio suave w=0.25), **1.0542** (promedio w=0.5) — una caída del 44 %. Tras la cirugía:
4.29 / 3.39 / 2.75.

**Pero ese daño es transitorio y no explica el resultado.** A 50 pasos de la cirugía las tres
condiciones tienen el mismo gradiente (~2.2). Y sin embargo el control mantiene la ventaja en
pérdida: en los primeros 250 pasos el control desciende **−0.0895 BPC** y el promediado solo
**−0.0508**, un 76 % menos de progreso, con los gradientes ya igualados desde el paso 50.

**Explicación que queda en pie:** el promedio **no crea progreso, lo adelanta**. El punto
promediado es la posición suavizada alrededor de la cual la trayectoria ya oscila — es decir,
adonde el entrenamiento llegaría por su cuenta en esos ~250 pasos. Se cobra por adelantado un
avance que iba a ocurrir igual, y encima se paga el transitorio. Eso reconcilia los tres hechos:
la mejora inmediata es real, el gradiente cae (el descenso local ya está agotado ahí), y ambos
brazos convergen al mismo sitio con el control ligeramente delante.

Corolario de diseño: la ganancia del promedio solo es libre **cuando no hay un “después” en el
que devolverla** — es decir, al final del entrenamiento.

## Qué sobrevive de la idea

1. **Promediar checkpoints da calidad gratis** — 6/6, −0.0175 BPC, coste nulo. Úsalo para
   **evaluar y desplegar**: al final del entrenamiento, evalúa la media de los últimos 2-3
   checkpoints en vez del último. Es gratis y mide mejor.
2. **No lo uses para continuar entrenando.** Medido: cuesta más de lo que da.
3. **La versión hacia adelante está muerta** (`extrapolacion_results.json`): saltar en la
   dirección reciente empeora el BPC en los tres regímenes probados, monótonamente con el tamaño
   del salto.

## Alcance honesto

Una escala, una tarea, un LR, 6 semillas, una sola configuración de promedio (2 snapshots a 100
pasos). No se barrió el número de snapshots ni su separación —la sonda sugiere que ambos
importan—, ni se probó promediar en otros puntos, ni con LR cíclico (el régimen donde SWA rinde
más), ni mantener un promedio en paralelo sin tocar los pesos de entrenamiento, que es la forma
canónica de SWA y la única que estos datos **no** contradicen.

**Artefactos:** `t13_results.json`, `log_T13.txt`, `t13.py`, `extrapolacion.py`,
`extrapolacion_results.json`.
