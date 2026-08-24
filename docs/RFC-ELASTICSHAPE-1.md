# RFC-ELASTICSHAPE-1 — Crecimiento de forma en espacio factorizado

**Título:** ElasticShape: entrenar en una forma barata y crecer por cirugía exacta hacia una
forma factorizada ancha
**Autores:** MSC Star Team — Esraderey · Raúl Cruz Acosta
**Fecha:** 2026-08-22 · **Estado:** propuesta con validación empírica preregistrada (T8-T14)
**Implementación de referencia:** `mztrain` 1.3.2 — `src/mztrain/shape_ops.py`,
`src/mztrain/elastic_shape.py` (opt-in; nada se activa sin llamada explícita)
**Artefactos:** `docs/evidencia/` — preregistros, veredictos y datos crudos de T4 a T14, más el
banco reproducible completo en `docs/evidencia/banco-t9-t14/`

---

## Resumen

Un transformador factorizado (cada capa lineal parametrizada como `W = U·diag(S)·V`) cabe en
memoria a escalas donde el equivalente denso no cabe, pero aprende peor por parámetro y por
segundo en la fase temprana del entrenamiento. Este RFC propone explotar esa asimetría en vez
de sufrirla: entrenar primero una red **densa y estrecha**, convertirla a factorizada mediante
una descomposición exacta y **ensancharla por cirugía** hasta la forma objetivo, migrando el
estado del optimizador. La operación preserva la función de la red salvo un residuo acotado y
medido de la normalización por capas.

Dos experimentos preregistrados sostienen la propuesta. **T8** (11 M-equivalentes) midió que el
morfismo alcanza la calidad del entrenamiento desde cero en el 54 % del reloj. **T9**
(2026-08-21) cerró las tres limitaciones que T8 había declarado abiertas: a **38.8 M** el ahorro
sube al 52 % del reloj (ratio 0.476, 3/3 semillas); encadenar **dos** cirugías en vez de una
mejora el ahorro un 14 % más; y bajo **bf16** el efecto sobrevive con paridad de calidad, aunque
con menos margen. T9 reportó además dos hallazgos que corrigen supuestos del propio programa
experimental: la deriva de la cirugía no crece con la anchura, y la varianza de la métrica a
semilla fija supera a la varianza entre semillas.

El alcance es una tarea (WikiText-2 a nivel de carácter), un régimen de tasa de aprendizaje y
dos escalas. El RFC declara ese límite en lugar de extrapolarlo.

---

## 1. Problema

El programa empírico de MZTrain estableció cuatro hechos, todos con preregistro y regla de
decisión previa a los datos:

| Fuente | Hecho | Consecuencia |
|---|---|---|
| T4 (15 runs, 3 escalas) | El desfase de calidad iso-parámetro entre factorizado y denso es estable con la escala: Δ = +0.295 / +0.304 / +0.257 BPC a 11 M / 57 M / 152 M | No hay base empírica para esperar que el desfase se cierre solo al crecer |
| T5 (sondas de capacidad) | En una RTX 4060 el techo práctico denso está en ~280 M parámetros; el factorizado llega mucho más lejos a igual memoria | A escala grande, factorizar no es una opción: es la única forma que cabe |
| T6-C1 (barrido de LR) | Con la tasa de aprendizaje ajustada el desfase **se ensancha** (0.295 → 0.513): el denso aprovecha tasas altas de forma estable, el factorizado se desestabiliza | El desfase no era un artefacto de hiperparámetro |
| T6-C2, T7 | El *annealing* de rango y el rebalanceo de gauge no producen beneficio detectable | Los ataques directos al desfase fracasaron |

La lectura conjunta es incómoda y fértil: el factorizado es peor *aprendiendo* y mejor
*cabiendo*, y ninguna de las dos cosas se arregla por separado. Si el desfase no se cierra con
hiperparámetros ni con dinámica del rango, queda una vía distinta: **no pagar la fase temprana
en la parametrización cara**.

## 2. Propuesta

**ElasticShape** entrena la red en una forma barata durante la fase temprana y la transforma,
en un instante determinado del entrenamiento, en la forma objetivo:

```
denso estrecho (d)  ──factorizar──►  factorizado exacto (rango m = d)
                    ──ensanchar──►  factorizado ancho (d' > d, mismo rango)
```

El modelo resultante es, al parámetro, **el mismo** que se habría entrenado desde cero: misma
arquitectura, mismo número de parámetros, mismos tensores. Lo que cambia es la trayectoria. La
comparación experimental es por tanto limpia por construcción, sin equivalencias aproximadas
entre modelos distintos.

Tres operaciones componen la cirugía: **factorizar** (denso → factorizado, exacta), **ensanchar**
(aumentar `d` con el rango fijo) y **profundizar** (añadir bloques identidad). Un controlador
declarativo las agenda por paso, migra el estado de Adam y aplica una rampa de tasa de
aprendizaje posterior al evento.

Según el conocimiento de los autores —sin búsqueda bibliográfica sistemática en este RFC—, el
crecimiento progresivo se ha explorado en el espacio **denso** (apilado de capas, Net2Net, LiGO,
con ahorros reportados del 30-50 %), pero no **dentro del espacio factorizado**, donde la
operación de ensanchar tiene una forma algebraica distinta y notablemente más barata.

## 3. Fundamento matemático

Se distingue lo que se preserva de forma exacta de lo que no. Autoengañarse aquí produce un
método que funciona en el papel y deriva en la práctica.

### 3.1 Ensanchar es exacto a nivel de capa

Para una capa `y = U·diag(S)·V·x` con `U ∈ ℝ^{out×r}`, `V ∈ ℝ^{r×in}`:

```
V' = [V | 0]   (r × in')     las dimensiones nuevas de entrada se ignoran
U' = [U ; 0]   (out' × r)    las dimensiones nuevas de salida emiten exactamente 0
```

`S` y `r` quedan intactos. La salida vieja es idéntica —misma suma, mismos términos— y la nueva
es cero. El coste es lineal en las dimensiones añadidas: no hay reconstrucción de `W`.

### 3.2 Las dimensiones nuevas necesitan una semilla

Si una dimensión emite cero y nadie la lee, su gradiente es cero y permanece muerta. La
verdad de diseño quedó anclada en un test durante la certificación de M1. El remedio de la
versión 1 es ruido ε en las filas nuevas de `U` y en las columnas nuevas de los *embeddings*
(`noise_scale` relativo, 1e-3 por omisión): la deriva funcional queda acotada por ε y el
gradiente fluye desde el primer paso.

### 3.3 Los layouts estructurados exigen mapas de índices

En un `qkv` fusionado la salida es `[q|k|v]`; al crecer `d` las dimensiones nuevas se insertan
**dentro** de cada bloque y de cada cabeza, no al final. Las primitivas reciben `in_map`/`out_map`
—las posiciones donde viven las dimensiones viejas en el layout nuevo—. Un *zero-pad* ingenuo
corrompe el `reshape` por cabezas; los tests clavan exactamente ese caso.

### 3.4 La escala de la atención se corrige en un solo lado

`scaled_dot_product_attention` divide por `√(head_dim)`. Con el número de cabezas fijo, crecer
`d` aumenta `head_dim` y reescalaría los scores por `√(d_h/d_h')`. La corrección exacta consiste
en multiplicar las salidas del bloque `q` —filas de `U` **y sus entradas de sesgo**— por
`√(d_h'/d_h)`, en `q` **o** en `k`, nunca en ambos: el producto es bilineal. Con las componentes
nuevas en cero, los productos punto no cambian y los scores quedan idénticos.

El sesgo importa: `q_i = U_i·h + b_i`. Escalar solo `U` rompe la identidad cuando `b ≠ 0`. Este
fue uno de los cuatro hallazgos de severidad alta de la auditoría ciega de M1 (§4.3).

### 3.5 La normalización por capas NO se preserva a nivel de red

`LayerNorm` sobre `[x, 0]` altera la media y la desviación por muestra. La dilución de varianza
(factor `√(d'/d)`) se cancela escalando `γ` por `√(d/d')`, pero el residuo del desplazamiento de
la media es **de primer orden** en μ —corrección de la auditoría a una versión anterior de la
especificación, que lo daba por segundo orden—. La deriva relativa de logits medida con ruido
nulo fue de 3.9-5.3 % en un modelo de juguete a razón 1.5×, y ~10 % a razón 4×.

Sobre modelos **entrenados** la deriva resulta mucho menor: 0.24-1.4 % en T8 y 0.18-2.23 % en
T9. La cirugía se mide con una sonda en cada evento y la rampa de tasa de aprendizaje absorbe el
transitorio; no lo borra.

### 3.6 Profundizar es exacto

Un bloque residual nuevo con `U = 0` **y sesgo cero** en `proj` y `fc2` es la identidad `x + 0`.
Con sesgo distinto de cero sería `x + b_proj + b_fc2`. El bloque nace muerto-exacto y despierta
por gradiente.

### 3.7 Denso → factorizado es exacto, y el rango no puede superar la dimensión mínima

`dense_to_factorized` embebe `W` con la descomposición en valores singulares completa a rango
`m = min(out, in)`: exacto salvo error numérico. Un rango mayor que `m` es **inexpresable en la
misma forma**. Crecer el rango por encima de `m` solo es posible por composición —convertir,
ensanchar (lo que sube `m`), y entonces crecer el rango con la maquinaria existente—.

### 3.8 El estado de Adam se migra, no se reinicia

Los momentos `m` y `v` se rellenan con ceros en las filas y columnas nuevas; el contador `step`
se conserva **por valor** (clonado, no por referencia: AdamW lo muta en el sitio). Con `step`
alto y `m = v = 0`, la corrección de sesgo produce un primer paso de hasta
`lr·(1−β₁)/√(1−β₂) ≈ 3.16·lr`: acotado, transitorio, absorbido por la rampa.

La conversión densa→factorizada es la única operación que **sí** reinicia el optimizador: las dos
parametrizaciones no son conmensurables y migrar el estado entre ellas no tendría sentido.

Tras la corrección de escala de la atención (§3.4), el gradiente futuro del bloque `q` escala por
`1/c` con `c = √(d_h'/d_h)`. El estado migrado se corrige en consecuencia: `m /= c`, `v /= c²`.

## 4. Implementación

### 4.1 Primitivas (`shape_ops.py`)

Ocho funciones puras, sin dependencia del motor ni de la configuración:
`widen_factorized_linear`, `widen_embedding`, `widen_layernorm`, `scale_output_rows`,
`zero_block_outputs`, `dense_to_factorized`, `pad_state_tensor`, `pad_adam_entry`.

### 4.2 Controlador (`elastic_shape.py`)

`GrowthEvent` (paso, factorizar, nueva anchura, capas a añadir, escala de ruido) ·
`ShapeSchedule` (eventos por paso, con recuperación de eventos saltados) · `apply_event`
(orquesta la mutación, migra el optimizador, mide la deriva) · `LrWarmup` (rampa lineal con base
persistente en el grupo de parámetros) · `migrate_optimizer` (trasplante y relleno según el
libro mayor de migración).

### 4.3 Endurecimiento por auditoría adversarial

Cada módulo pasó por dos revisiones ciegas e independientes, con lentes distintas, según el
contrato de proceso del proyecto. Los defectos encontrados y corregidos:

| Hallazgo | Severidad | Efecto si no se corrige |
|---|---|---|
| Sesgo ignorado en `scale_output_rows` y `zero_block_outputs` | Alta | La identidad se rompe con `bias=True` — el caso real de mztrain |
| `randn` en CUDA con generador de CPU | Alta | Excepción en tiempo de ejecución |
| Fila de `padding_idx` corrompida con ruido | Alta | Corrupción silenciosa del *embedding* de relleno |
| Residuo de `LayerNorm` documentado como segundo orden | Alta | Especificación falsa; la deriva real es mayor |
| `factorize` redundante vaciaba el optimizador en silencio | Alta | Pérdida total de momento sin aviso |
| Estado de `q` desfasado en `c²` tras la corrección de la atención | Alta | Desajuste sistemático del optimizador |
| Verificación previa ausente antes de mutar | Alta | Un error a mitad dejaba el modelo roto de forma permanente |
| `weight_decay` sobrescrito por el valor por omisión | Media | Cambio silencioso del régimen de regularización |
| Rampa construida sobre otra rampa sin terminar | Media | Pérdida permanente de la tasa objetivo |
| `step` de Adam clonado por referencia | Media | Corrupción del contador entre parámetros |

Estado actual: 40 pruebas específicas del módulo dentro de una batería de 347, en verde.

## 5. Método experimental

**Protocolo** (idéntico en todo el arco, para que las comparaciones entre experimentos sean
legítimas): WikiText-2 a nivel de carácter, lote 16, secuencia 256, AdamW con tasa 3e-4 y
decaimiento 0.01 sin planificador, sesgos desactivados, cabezal atado, número de cabezas fijo.

**Métrica primaria:** `val_bpc_full`, bits por carácter sobre **todo** el conjunto de validación
en ventanas no solapadas, evaluado **siempre en fp32**, también en las condiciones que entrenan
en bf16.

**Reloj:** tiempo de entrenamiento puro, con la cirugía dentro y las evaluaciones fuera.

**Cantidades comparadas:** `Q` es la calidad final de la referencia entrenada desde cero;
`T_R` su reloj; `T_M` el primer instante en que el morfismo alcanza `Q`; el cociente
`T_M/T_R` es la magnitud de decisión.

**Preregistro.** Cada experimento fija por escrito, antes de generar un solo dato, la
hipótesis, el umbral de confirmación, el **umbral de muerte** y la predicción del autor. T8
declaró que si el morfismo no alcanzaba `Q` en ≤0.7 del reloj, ElasticShape se rechazaba como
propuesta y las primitivas quedaban como infraestructura sin reivindicación.

## 6. Resultados

### 6.1 T8 — el juicio inicial (11 M-equivalentes, fp32)

Referencia `F_192@384` desde cero: `Q` = 2.9021, `T_R` = 194.1 s. Morfismo: 2000 pasos densos a
`d = 192`, una cirugía a `d = 384`, fase ancha.

| Semilla | Deriva de la cirugía | T_M | BPC a reloj igual |
|---|---|---|---|
| 0 | 0.24 % | 105.4 s | 2.3394 |
| 1 | 1.40 % | 104.7 s | 2.3708 |
| 2 | 0.79 % | 104.9 s | 2.3526 |

Cociente **0.541** ≤ 0.7 con 3/3 semillas: claim confirmado. A reloj igual el morfismo rendía
2.34-2.37 frente a 2.90.

### 6.2 T9 — las tres limitaciones declaradas (26 runs, 3.6 h de GPU, 0 NaN)

T8 declaró explícitamente tres cosas no probadas: escala mayor, cirugías encadenadas y bf16.
T9 las atacó con la misma disciplina de preregistro.

| Brazo | Regla preregistrada | Resultado | Veredicto |
|---|---|---|---|
| **A — escala** (38.8 M, 5.1× T8) | cociente ≤ 0.7, todas las semillas alcanzan `Q` | **0.476** (3/3) | confirmado |
| **B — encadenado** (dos cirugías) | cociente_B ≤ 1.15 · cociente_A | 0.496 frente a 0.574 → **0.864** | hipótesis sobrevive |
| **C — bf16** | cociente ≤ 0.7, \|ΔQ\| ≤ 0.136, sin NaN | 0.689 · ΔQ 0.0195 · 0 NaN | confirmado |

15 de 15 morfismos alcanzaron `Q`. Ninguna referencia hubo que excluir por la regla
anti-estancamiento (§6.5).

**Réplica independiente.** El banco de T8 se había perdido. T9 lo reconstruyó desde el protocolo
escrito y reprodujo las anclas de T4 con desviaciones ≤0.07 BPC, y la condición de T8 con
cociente 0.574 frente a 0.541. El resultado de T8 no dependía de aquel código concreto.

### 6.3 La ventaja crece con la escala

| Escala | Endpoint | `T_R` | `T_M` medio | Cociente |
|---|---|---|---|---|
| S1 (11 M-equiv) | `F_192@384`, 7.62 M par. | 208.7 s | 119.8 s | 0.574 |
| S2 (38.8 M) | `F_384@768`, 38.84 M par. | 852.5 s | 405.8 s | **0.476** |

El mecanismo estaba preregistrado como predicción y se midió: la fase barata cuesta **0.490** de
la fase ancha a S1 y **0.421** a S2 (26.5/54.0 frente a 88.7/210.8 ms por paso). Cuanto más caro
el destino, más ventaja compra el mismo número de pasos baratos. A reloj igual, el morfismo a S2
rinde 2.154-2.165 frente a `Q` = 2.6375.

La deriva de la cirugía **no** creció con la anchura: 0.18-1.50 % a `d = 768` frente a
0.69-2.23 % en S1 (todos los eventos, `d = 288` y `d = 384`; 1.13-2.23 % en el salto único). Era el riesgo explícito declarado en el preregistro y no se materializó.
En una semilla la operación incluso mejoró el BPC. El coste de reloj de la cirugía fue de ~1.0 s
sobre corridas de ~1000 s.

### 6.4 Encadenar cirugías supera al salto único

| Condición | `T_M` medio | Cociente | BPC final medio |
|---|---|---|---|
| Un salto: 192 → 384 | 119.8 s | 0.574 | 2.241 |
| Dos peldaños: 192 → 288 → 384 | **103.5 s** | **0.496** | **2.219** |

El mecanismo es visible en los relojes: el encadenado entra a la forma final más tarde (68-75 s
frente a 50-58 s) pero mucho más adelantado (BPC 3.09-3.17 frente a 3.57-3.60). Gasta unos 20 s
más en formas baratas y llega ~0.45 BPC por delante.

Es la premisa de T4/T6 aplicada de forma **recursiva**: si lo estrecho compra más calidad por
segundo, cada peldaño intermedio vuelve a comprar barato. La lectura natural —una **escalera de
crecimiento** de tres o más peldaños sería aún mejor— es una hipótesis que este RFC **abre y no
cierra**: T9 no barrió planificaciones.

### 6.5 bf16 conserva el efecto y le quita margen

Referencia en bf16: `Q` = 2.8996 frente a 2.8801 en fp32; diferencia 0.0195, dentro del umbral
de paridad de 0.136 heredado de T6. Aceleración de reloj 1.42×, por encima del 1.3× que T6-C4
consideraba improbable a esta escala. Cero NaN. La cirugía es idéntica a la de fp32, como debe
ser: los parámetros son fp32 y la conversión automática solo afecta al paso de entrenamiento.

El hallazgo relevante es que bf16 **no acelera las dos fases por igual**:

| Fase | fp32 | bf16 | Aceleración |
|---|---|---|---|
| Densa `d = 192` (barata) | 26.50 ms/paso | 23.86 | 1.111× |
| Ancha `F_192@384` (cara) | 54.04 ms/paso | 39.51 | **1.368×** |

bf16 acelera preferentemente la fase que el morfismo intenta evitar, y el cociente pasa de 0.574
a 0.689. De ahí una conclusión que conviene decir en voz alta: **parte del beneficio del
morfismo es un artefacto de la ineficiencia de fp32 en multiplicaciones matriciales grandes.**
Cuanto mejores sean los núcleos de cómputo de la fase ancha —bf16 hoy, fp8 mañana—, menor será
el margen. El efecto sobrevive en bf16, pero con 0.689 frente a un umbral de 0.7.

### 6.6 Hallazgo transversal: la varianza a semilla fija supera a la varianza entre semillas

Durante la calibración, la referencia con semilla 0 dio 3.3970 —estancada en la meseta—. La misma
semilla, el mismo código y el mismo generador, ejecutados de nuevo, dieron 2.9792. La causa son
núcleos CUDA no deterministas amplificados por una **bifurcación**: el modelo de caracteres se
asienta en una meseta de estadística unigrama/bigrama en torno a 3.4-3.6 y «rompe» hacia 2.6-2.95
en un paso incierto. A 4000 pasos, unas corridas han roto y otras no.

Esto tiene consecuencias para el programa entero, no solo para T9:

1. La vara de ruido σ₁ = 0.068 de T4 **subestima** la incertidumbre real. Se midió entre
   semillas que habían roto todas. La dispersión cerca de la bifurcación es de ~0.4 BPC.
2. Los datos publicados ya lo contenían sin que se hubiera nombrado: en T6, la condición
   `F_64` con tasa 1.2e-3 dio {2.6543, **3.5684**, 2.6539}.
3. Las comparaciones de T9 son inmunes por construcción. `Q` procede de tres o cuatro
   referencias, y la regla anti-estancamiento es **asimétrica**: una referencia estancada
   (BPC > 3.20) se excluye, porque inflaría `Q` y regalaría el resultado al morfismo; un
   morfismo estancado **nunca** se excluye. La regla juega en contra de la hipótesis propia.
4. Recomendación operativa: todo claim medido a paso fijo en esta tarea debería reportar
   réplicas **de la misma semilla**, o declarar que la métrica se toma con la bifurcación
   abierta.

## 6.7 Ampliación T10-T14: qué se atacó después de T9

T9 cerró las tres banderas de T8. Los cuatro experimentos siguientes atacaron la propia
propuesta por donde era más débil, con preregistro y umbral de muerte previos a los datos.

**T10 — el claim bajo tasa de aprendizaje afinada.** Era la amenaza número uno de este RFC:
todo el arco corría con LR 3e-4, que T6-C1 había demostrado subóptimo. A 1.2e-3 el claim
**sobrevive** (ratio 0.442 con la línea base de entonces; 0.542 tras la corrección de T11). Y
apareció algo que no se buscaba: **el factorizado entrenado desde cero sin rampa de LR degenera
en 4 de 6 corridas** (Fisher exacto unilateral p = 0.0303), sin NaN y sin aviso — solo una curva
que se queda plana. El morph no degeneró en ninguna.

**T11 — aislar el mecanismo, y la corrección que se hizo a sí mismo.** El 2×2 (fase densa ×
rampa de LR) demostró que **ambos factores bastan por separado**: una rampa de 200 pasos al
inicio elimina la degeneración de la referencia (0/6). Eso **retira la afirmación fuerte de
T10** —el morph no es *la única* forma de evitar el fallo— y la deja en la versión honesta: el
morph es el que **no depende de acertar con la rampa** (0/6 con ella y sin ella). Como la rampa
además mejora la referencia (Q 2.6182 → 2.4536), el ratio se recalculó contra esa línea base
fuerte: **0.542**, y el 0.442 quedó retirado.

**T12 — ¿estaban mal equipadas las líneas base de T8 y T9?** A LR 3e-4 la rampa **empeora** la
referencia en 4 de 4 semillas (+0.1345 BPC): su efecto **cambia de signo con la tasa**. La línea
base de T9-B ya era la mejor disponible y el 0.574 no necesitaba corrección. El experimento
produjo además un recálculo de 0.461 que **se descartó** por medir contra un objetivo más fácil
y mezclar relojes de dos sesiones — la clase de cifra que favorece al autor y hay que tirar.

**T13 — promediado de pesos.** Promediar los últimos checkpoints mejora el modelo en 6/6
semillas (−0.0175 BPC, coste nulo), pero **sustituir** los pesos de entrenamiento por el
promedio a mitad de camino es contraproducente (2/6, −2.58 %): el promedio no crea progreso, lo
**adelanta**, y deja al modelo donde la trayectoria llegaría sola en ~250 pasos, en terreno más
llano. La variante que la evidencia no refuta —acumular el promedio **en paralelo**, sin tocar
los pesos de entrenamiento— está implementada y probada como componente aparte
(`banco-t9-t14/weight_avg.py`, 17 comprobaciones), pendiente de la doble revisión ciega que este
proyecto exige antes de integrar.

**T14 — ¿limitaba el corpus?** Con 11× más datos (char-WikiText-103, 0.20 épocas frente a 2.26)
la ganancia por multiplicar los parámetros por 5.1 pasó de +0.1290 a +0.1301: **cociente 1.01**,
cuando la regla exigía ≥1.5 para culpar al corpus. Lo que limita es el **presupuesto de
cómputo**: 24.6M tokens son el 16 % de lo que pide el modelo pequeño y el **3 %** de lo que pide
el grande.

### Dos correcciones metodológicas que afectan a todo el arco

1. **La varianza a semilla fija supera a la varianza entre semillas.** El mismo seed, código y
   generador dieron 3.3970 y 2.9792 en dos ejecuciones: la tarea es bimodal y el no-determinismo
   de CUDA decide de qué lado cae. σ₁ = 0.068 (T4) subestima la incertidumbre real, que cerca de
   la bifurcación es de ~0.4 BPC. Estaba ya en los datos publicados de T6 sin nombrarse.
2. **No comparar «BPC final» entre configuraciones que no han convergido.** Ninguna corrida de
   T14 lo había hecho al paso 6000 (pendientes de +0.016 a +0.063 BPC por cada 500 pasos), y
   tres lecturas cambian con el corte. El banco incorpora la guarda que lo impide
   (`convergencia.py`: `comparar_final` lanza `NoConvergido` salvo declaración explícita de
   «a presupuesto fijo de N pasos»).

**Los claims del morph no están afectados por (2):** su métrica es el instante en que el morph
alcanza una calidad **fija y preregistrada**, una carrera contra un objetivo, no un BPC en un
corte arbitrario.

### Auditoría del banco

72 corridas de T9-T14 verificadas mecánicamente: 0 fallos. Se comprobó que la rampa de LR hace
lo declarado, que el LR y el weight decay sobreviven a la cirugía (que reconstruye el
optimizador), que cada `T_M` es recalculable desde su curva y que el umbral de degeneración no
decide ninguna clasificación. La única inconsistencia hallada —dos implementaciones divergentes
de la métrica de evaluación— se corrigió de raíz unificándolas, con un ancla que verifica
igualdad **bit a bit** sobre un modelo fijo, y se re-ejecutó la única celda contaminada.

## 7. Amenazas a la validez

- **Una sola tarea.** WikiText-2 a nivel de carácter. La bifurcación descrita en §6.6 es un
  fenómeno de esta tarea; su presencia o ausencia en otras modifica la lectura de la métrica,
  no el mecanismo de la cirugía.
- **Un solo régimen de tasa de aprendizaje.** 3e-4, heredado del arco para mantener
  comparabilidad. T6-C1 mostró que 1.2e-3 da mucha mejor calidad absoluta. Que el efecto
  sobreviva con la tasa ajustada **no está probado**.
- **Una planificación por brazo, sin ajuste.** El reparto 50/50 entre fase barata y fase ancha
  se fijó en T8 sin barrido y se heredó. El óptimo es desconocido.
- **Dos escalas.** 7.6 M y 38.8 M de parámetros. La tendencia entre ambas es favorable; dos
  puntos no son una ley.
- **Un solo tipo de hardware.** RTX 4060. Las conclusiones de reloj dependen del perfil de
  cómputo, y §6.5 muestra que ese perfil **cambia la magnitud del efecto**.
- **Sin cruces.** No se probó escala × encadenado × bf16 simultáneamente.
- **Sin distribuido, sin fp8, sin escaleras de más de dos peldaños.**

## 8. Trabajo propuesto

En orden de valor esperado por coste:

1. **Escalera de crecimiento.** Tres o más peldaños con barrido del reparto de pasos. Es la
   hipótesis que abre §6.4, y la más barata de probar.
2. **El efecto bajo tasa de aprendizaje ajustada.** Repetir T9-B con 1.2e-3 para separar el
   mecanismo del régimen heredado.
3. **fp8 y núcleos rápidos.** §6.5 predice que el margen se estrecha. Es una predicción
   falsable y conviene ejecutarla antes de que la haga otro.
4. **Planificación automática.** Decidir el instante del crecimiento con una señal medida
   —saturación de la fase barata— en lugar de fijarlo a mano.
5. **Escala mayor y modelos de subpalabra**, donde la bifurcación de §6.6 no contamina la
   métrica.

## 9. Reproducibilidad

Todo resultado de este RFC procede de artefactos con volcado atómico e incremental, no de
registros de progreso. Los veredictos se calculan con un analizador que aplica las reglas
preregistradas sobre el JSON de resultados.

- T4-T8: `docs/evidencia/` en el repositorio sellado, con firma Ed25519, resumen SHA-256/SHA-512
  y raíz de Merkle (`SEAL.json`, `MANIFEST.sha256`).
- T9: `D:\mztrain-evidencia-T9\` — preregistro, veredicto, `t9_results.json` (26 corridas),
  banco completo (`bank.py`, `t9.py`, `analyze.py`), registros de las tres ejecuciones y las
  calibraciones contra las anclas de T4.
- La evidencia de T9 está **fuera del árbol sellado** de forma deliberada: incorporarla invalida
  la verificación hasta volver a sellar, y volver a sellar exige la clave privada del titular.

## 10. Decisión solicitada

Este RFC no pide un cambio de comportamiento por omisión: ElasticShape es y sigue siendo
opt-in. Pide tres cosas concretas:

1. **Aceptar el claim en su alcance declarado** —el morfismo alcanza la calidad del
   entrenamiento desde cero en aproximadamente la mitad del reloj, a dos escalas, con cirugías
   encadenadas y en bf16— e incorporar T9 al expediente sellado.
2. **Registrar la escalera de crecimiento** (§6.4) como la línea abierta con mayor valor
   esperado.
3. **Adoptar la corrección metodológica de §6.6** en todo experimento futuro del arco.

---

## Apéndice A — Configuraciones exactas

| | S1 | S2 |
|---|---|---|
| Forma barata (fase 1) | denso `d = 192` | denso `d = 384` |
| Peldaño intermedio (brazo B) | `d = 288` | no probado |
| Forma objetivo | `F_192@384`, L=6, h=6 | `F_384@768`, L=8, h=8 |
| Parámetros del destino | 7 620 096 | 38 842 368 |
| Pasos de referencia | 4000 | 4000 |
| Pasos en forma barata | 2000 | 2000 |
| Tope de la fase ancha | 4000 | 4000 |
| Rampa tras la cirugía | 200 pasos, suelo 0.1 | igual |
| Ruido en dimensiones nuevas | 1e-3 | igual |
| VRAM pico del destino | 1.10 GB | 2.79 GB |

## Apéndice B — Deriva de la cirugía sobre modelos entrenados

| Experimento | Anchura destino | Deriva relativa de logits | ΔBPC pre/post |
|---|---|---|---|
| T8 | 384 | 0.24-1.40 % | ≈0 |
| T9-B, un salto | 384 | 1.13-2.23 % | 0.10-0.62 % |
| T9-B, encadenado (por evento) | 288 y 384 | 0.69-1.58 % | 0.00-0.95 % |
| T9-A | 768 | **0.18-1.50 %** | −0.08 a +1.32 % |

La banda honesta sobre modelos entrenados es **0.2-2.3 %**, más ancha que la de 0.24-1.4 % que
reportó T8 con tres semillas.

## Apéndice C — Trazabilidad

`PREREGISTRO-T4-barrido-escala.md` · `T4-VEREDICTO.md` · `t5_capacity.json` ·
`PREREGISTRO-T6-core.md` · `T6-VEREDICTO.md` · `PREREGISTRO-T7-dinamica.md` · `T7-VEREDICTO.md` ·
`SPEC-elasticshape-v1.md` · `PREREGISTRO-T8-morph.md` · `T8-VEREDICTO.md` ·
`PREREGISTRO-T9-banderas.md` · `T9-VEREDICTO.md` · `t9_results.json`
