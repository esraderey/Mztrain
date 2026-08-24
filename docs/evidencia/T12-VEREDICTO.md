# T12 — Veredicto: los números de T9-B aguantan, y por un motivo que no era el previsto

**Fecha:** 2026-08-22. **Preregistro:** `PREREGISTRO-T12-linea-base-fuerte.md`, escrito antes de
los datos. **Datos:** `t12_results.json` (8 runs, 0 NaN) + celdas de T9-B reutilizadas.
LR 3e-4, S1, endpoint `F_192@384`, semillas {0,1,2,3}.

## Resumen

| Pregunta | Resultado |
|---|---|
| ¿La rampa de LR mejora la referencia a 3e-4? | **No: la empeora en 4/4 semillas**, +0.1345 BPC de media |
| ¿Hay que corregir el 0.574 de T9-B? | **No.** Su línea base ya era la mejor configuración disponible |
| ¿Sobrevive el claim de T9-B? | **Sí, sin cambios: 0.574** |

La amenaza que T11 destapó —«T8 y T9 midieron contra referencias sin rampa»— queda cerrada para
T9-B, y no porque el número nuevo sea mejor, sino porque **el número viejo ya era el correcto**.

## El dato central: el efecto de la rampa cambia de signo con el LR

| Semilla | R sin rampa (T9-B) | R con rampa (T12) | efecto |
|---|---|---|---|
| 0 | 2.9792 | 3.1155 | +0.1362 peor |
| 1 | 2.8114 | 2.9648 | +0.1534 peor |
| 2 | 2.8224 | 2.9954 | +0.1730 peor |
| 3 | 2.9074 | 2.9828 | +0.0754 peor |
| **media** | **2.8801** | **3.0146** | **+0.1345 peor** |

A LR 1.2e-3 la misma rampa **rescataba** a la referencia (T11: de 4/6 degeneradas y Q = 2.6182,
a 0/6 y Q = 2.4536). A LR 3e-4 la **estorba** en las cuatro semillas. La lectura mecánica es
directa: una rampa paga donde hay inestabilidad que prevenir, y donde no la hay solo gasta pasos
a tasa reducida. La misma intervención, dos regímenes, signos opuestos.

El morph también empeora con rampa inicial, pero mucho menos: +0.0165 / +0.0118 / +0.0271 /
+0.0100 (media +0.016, frente a +0.134 de la referencia). Es coherente con lo que ya se sabía:
la fase densa a 3e-4 es estable de por sí y no gana nada con que la suavicen.

## Por qué el «recálculo» de 0.461 se descarta

El script produjo, como estaba previsto en el diseño, un recálculo de los morphs de T9-B contra
la nueva línea base: **ratio 0.461**, mejor que el 0.574 publicado. **Ese número se descarta**, y
conviene dejar escrito por qué, porque es exactamente la clase de cifra que uno se quedaría si no
lo pensara:

1. `Q' = 3.0146` es un objetivo **más fácil** que `Q = 2.8801`. Medir contra una línea base peor
   no es corregir un sesgo: es fabricar uno.
2. `T_R' = 230.9 s` procede de la sesión de T12 y los `T_M` de los morphs proceden de la sesión
   de T9, que corrió un **10.6 % más rápida** (208.7 s frente a 230.9 s por el mismo trabajo:
   4000 pasos del mismo modelo). Mezclar relojes de dos sesiones infla la ventaja del morph.

El criterio correcto es **mejor contra mejor**: la referencia en su mejor configuración a 3e-4
(sin rampa, 2.8801) contra el morph en la suya (sin rampa inicial). Eso es exactamente lo que
midió T9-B, y da **0.574**.

Como comprobación interna, dentro de la sesión de T12 y con ambas condiciones igualmente
penalizadas (las dos con rampa inicial), el ratio es 0.457. Es coherente —la rampa daña más a la
referencia que al morph— pero no es el titular: es la comparación entre dos configuraciones
subóptimas.

## Deriva de reloj entre sesiones: un caveat metodológico nuevo

Trabajo idéntico, sesiones distintas:

| Sesión | reloj de R (4000 pasos) |
|---|---|
| T9 (primera hora, GPU fría) | 208.7 s |
| T10 | 212.6 s |
| T11 | 213.0 s |
| T12 (tras horas de carga) | 230.9 s |

Hasta un **10.6 %** de diferencia por térmica. Consecuencia: **un ratio solo es legítimo si `T_M`
y `T_R` salen de la misma sesión.** Se verificaron los ratios ya publicados:

- T8, T9, T10: referencia y morphs corrieron en el mismo proceso. Sin contaminación.
- **T11** calculó su 0.542 con `T_M` de la sesión de T10 y `T_R` de la de T11 — una mezcla. Se
  comprobó: esas dos sesiones difieren un **0.2 %**, y el ratio corregido es **0.543**. El número
  publicado se sostiene, por poco margen y ahora verificado en vez de supuesto.

## Sobre las predicciones del autor

Predije un ratio entre 0.58 y 0.72, es decir, que el número empeoraría. Falló: la premisa
—que la rampa mejoraría la referencia a 3e-4— era falsa, así que no hubo corrección que aplicar.

Es el **cuarto fallo consecutivo** (T10 ×2, T11, T12), y el patrón ya es legible: en los cuatro
casos el error fue **asumir que un mecanismo transfiere de un régimen a otro**. La lógica de bf16
aplicada al LR alto (T10-Q1), la del rango 64 aplicada al rango 192 (T10-Q2), la ventana temporal
de la degeneración aplicada a la rampa (T11), la rampa de LR alto aplicada a LR bajo (T12). En
este sistema los mecanismos **no transfieren entre regímenes**, y esa es probablemente la
lección más reutilizable de toda la serie.

## Alcance honesto

Cuatro semillas, una escala, una tarea, una configuración de rampa. **Los ratios de T9-A (escala
S2) y T9-C (bf16) siguen sin corregir** por presupuesto; dado que T12 muestra que a 3e-4 la rampa
no mejora la referencia, es razonable esperar que tampoco los mueva —ambos corren a 3e-4—, pero
eso es exactamente el tipo de expectativa que esta serie lleva cuatro veces falsando. Queda
declarado como pendiente, no como resuelto.

**Artefactos:** `t12_results.json`, `log_T12.txt`, `t12.py`.
