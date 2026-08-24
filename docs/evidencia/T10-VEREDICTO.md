# T10 — Veredicto: el claim bajo LR ajustado, y una propiedad que no buscábamos

> **CORREGIDO POR T11 (2026-08-22).** El 2×2 de `T11-VEREDICTO.md` demostró que una rampa de LR
> de 200 pasos al inicio elimina la degeneración de la referencia (0/6), de modo que el morph
> **no es la única forma** de evitar el fallo: la afirmación fuerte de la sección «Lo que esto
> cambia» queda retirada. Y como la rampa también mejora la referencia (Q 2.6182 → 2.4536), el
> ratio de Q1 recalculado contra esa línea base fuerte es **0.542**, no 0.442. Lo que sobrevive:
> el factorizado desde cero **sin rampa** es frágil (4/6), el morph es insensible a esa decisión
> (0/6 con y sin rampa), y el claim de velocidad se sostiene con 0.542 ≤ 0.7.

**Fecha:** 2026-08-22. **Preregistro:** `PREREGISTRO-T10-lr-estabilidad.md`, escrito antes de los
datos, con predicciones del autor y umbrales de muerte. **Datos:** `t10_results.json` (15 runs,
0 NaN). Banco y protocolo de T9; **la única variable manipulada es el LR** (3e-4 → 1.2e-3, el
mejor del barrido de T6-C1). Endpoint `F_192@384` (7 620 096 params), fp32, 6 semillas por
condición.

## Resumen

| Pregunta | Regla preregistrada | Resultado | Veredicto |
|---|---|---|---|
| **Q1 — supervivencia** | ratio ≤ 0.7 con todos los morphs alcanzando Q | **0.442**, 6/6 | **SOBREVIVE**, y mejora sobre el 0.574 de T9-B |
| **Q2 — estabilidad** | p_R − p_M ≥ 2/6 con p_M < p_R | p_R = **4/6**, p_M = **0/6** | **H-STAB CONFIRMADA** |

La predicción del autor para Q1 —que el LR ajustado comprimiría la ventaja, por analogía con lo
que bf16 le hizo al morph en T9-C— **queda falsada**: la ventaja no se comprime, crece. La
predicción para Q2 —que el desenlace más probable era «no evaluable» porque la inestabilidad de
T6 se había medido a rango 64— también **queda falsada**: a rango 192 el fenómeno no solo existe,
es mucho más frecuente.

## Los datos

| Semilla | R_hi (desde cero) | M_hi (morph) | T_M |
|---|---|---|---|
| 0 | 3.6393 **degenerada** | 2.0920 | 104.3 s |
| 1 | 2.5659 | 2.0869 | 87.6 s |
| 2 | 3.5984 **degenerada** | 2.1050 | 100.6 s |
| 3 | 2.6704 | 2.0930 | 88.4 s |
| 4 | 3.5967 **degenerada** | 2.1073 | 91.1 s |
| 5 | 3.7783 **degenerada** | 2.0820 | 92.5 s |

Q = 2.6182 y T_R = 212.8 s, calculados **solo con las dos referencias sanas** (regla asimétrica:
una referencia degenerada infla Q y le regalaría el resultado al morph; un morph degenerado nunca
se excluye).

## Q1 — El claim sobrevive al LR ajustado, y mejora

media(T_M) = 94.1 s sobre T_R = 212.8 s → **ratio = 0.442**, con 6/6 morphs alcanzando Q.
Comparado con el 0.574 que dio la misma condición a 3e-4 (T9-B, mismo banco, mismo protocolo),
la ventaja del morph **crece** cuando la línea base está bien afinada.

**Sensibilidad al valor de Q.** Q procede de solo dos referencias sanas, así que conviene
comprobar que el veredicto no depende de esa media. Repitiendo el cálculo contra la **mejor**
referencia sana (Q = 2.5659, el listón más exigente posible con estos datos), los seis morphs
siguen alcanzándola y el ratio pasa a **0.472** — sigue muy por debajo del umbral de confirmación
de 0.7. El resultado es robusto al criterio.

La razón por la que el margen no se estrecha, a diferencia de bf16: el LR alto acelera el
aprendizaje de **ambas** fases, no solo de la cara. La fase densa a 1.2e-3 llega a BPC ~3.03-3.09
antes de la cirugía, frente a ~3.57 a 3e-4. El morph entra a la forma ancha mucho más adelantado,
y eso compensa con creces lo que gana la referencia.

## Q2 — La fase densa protege contra la degeneración

**Cuatro de seis referencias degeneran** (semillas 0, 2, 4, 5: BPC final 3.60-3.78, atascadas en
la meseta) frente a **cero de seis morphs**. Diferencia 4/6 = 0.667, muy por encima del umbral
preregistrado de 2/6.

- **Significación:** test exacto de Fisher unilateral sobre la tabla 2×2, **p = 0.0303**. Con
  seis semillas por brazo es el resultado más extremo posible, y aun así el margen de error de
  un experimento así de pequeño hay que declararlo: p = 0.03 es evidencia, no demostración.
- **Consistencia interna:** los seis morphs terminan en 2.082-2.107 (σ = 0.009, rango 0.025). Las
  referencias sanas terminan en 2.566 y 2.670. El morph no solo evita el fallo: produce
  resultados **mucho más reproducibles** que su alternativa.
- **Control diagnóstico (`F_64@384` a 1.2e-3, 3 semillas):** 2.6703 / 2.6052 / 2.6781, ninguna
  degenerada. La inestabilidad que T6 atribuyó al rango bajo **no se reprodujo a rango 64** en
  tres corridas, mientras que a rango 192 aparece en cuatro de seis. La lectura provisional —y
  hay que subrayar lo provisional, con tres corridas de control— es que el fenómeno **no es de
  rango bajo**: podría ser peor cuanto más rango tiene el modelo factorizado, lo contrario de lo
  que sugería T6 con una sola semilla caída.

## Lo que esto cambia

Hasta T10 el claim era sobre **reloj**: el morph llega antes al mismo sitio. T10 añade un eje
distinto y, en la práctica, más importante:

> A la tasa de aprendizaje que de verdad rinde, entrenar el factorizado desde cero **falla dos de
> cada tres veces**. El morph no falló ninguna, y además llegó antes y terminó mejor.

Dicho de otro modo: en este régimen el morph deja de ser una optimización de coste y pasa a ser
**la forma fiable de llegar al modelo factorizado**. Un usuario que entrene desde cero a 1.2e-3
tiene dos tercios de probabilidad de tirar el presupuesto de cómputo, y el fallo no avisa: no hay
NaN ni excepción, solo una curva que se queda plana en 3.6.

Esto también reinterpreta hacia atrás el desfase de T4/T6. Parte de lo que se midió como «el
factorizado aprende peor» puede ser, en realidad, **«el factorizado se atasca con probabilidad
alta y el promedio entre semillas mezcla dos poblaciones distintas»**. No es lo mismo un método
peor que un método frágil: el segundo se arregla, y T10 sugiere cómo.

## Alcance honesto

- **Seis semillas por brazo, una escala, una tarea, un LR alto.** p = 0.0303 con n = 6+6 es
  evidencia razonable de un efecto grande, no una medición de su magnitud. La tasa real de
  degeneración podría estar en cualquier lugar entre ~0.3 y ~0.9.
- **El control de rango 64 tiene solo tres corridas.** La afirmación «el fenómeno empeora con el
  rango» es una **hipótesis** derivada de comparar 0/3 contra 4/6, no un resultado.
- **No se probó** el encadenado a LR alto, ni bf16 a LR alto, ni la escala S2 a LR alto, ni otras
  tasas intermedias entre 3e-4 y 1.2e-3 (donde estaría la frontera de la inestabilidad).
- **Hay DOS modos de fallo, no uno** (corrección posterior al primer redactado de este veredicto,
  al inspeccionar las curvas). Cuatro referencias degeneraron, pero no de la misma forma: las
  semillas 0, 2 y 4 **nunca rompieron la meseta** (planas en ~3.6 durante los 4000 pasos; la 4
  además dio un pico a 5.239 en el paso 1500 y volvió), mientras que la semilla 5 **sí rompió**
  —bajó a 2.585 en el paso 3000— y después **colapsó** a 4.020 y 3.778. Es inestabilidad
  post-ruptura, no estancamiento. Cero NaN en los quince runs en ambos modos.
- **La degeneración no se decide en la ventana temprana:** todas las referencias, sanas y
  degeneradas, permanecen en ~3.7 durante los primeros 1000-2000 pasos. Las sanas rompen entre
  el 2000 y el 3000. Esto **rebaja** al warmup como explicación de la protección —una rampa de
  200 pasos actúa mucho antes de donde se juega el resultado— sin exculparlo: una rampa cambia
  toda la trayectoria posterior, no solo su ventana.
- **La fase densa no muestra ninguno de los dos modos:** las seis fases densas del morph
  descienden monótonas de 3.72 a ~3.03, sin meseta y sin picos. La consistencia entre semillas
  es notable (rango 0.064 en el paso 2000).
- El mecanismo por el que la fase densa protege **no está identificado**. La hipótesis natural —la
  fase densa lleva al modelo a una región desde la que el factorizado ya no se atasca— es
  plausible y no está probada. Distinguirla de alternativas (por ejemplo, que el warmup posterior
  a la cirugía sea lo que protege) exige un experimento con controles que T10 no incluye.

## Lo que sigue

1. **Aislar el mecanismo de la protección.** Control obvio: morph **sin** warmup tras la cirugía,
   y referencia desde cero **con** un warmup equivalente al inicio. Si la referencia con warmup
   deja de degenerar, el mérito era del warmup y no de la fase densa. Es barato y es la prueba
   que debe correrse antes de contar esto como propiedad del morph.
2. **Mapear la frontera de inestabilidad** en LR ∈ {6e-4, 8e-4, 1e-3, 1.2e-3}.
3. **Escala:** repetir Q2 a S2, donde una corrida perdida cuesta veinte minutos en vez de cuatro.

**Artefactos:** `t10_results.json`, `log_T10.txt`, `t10.py`, `bank.py`.
