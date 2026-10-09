# PREREGISTRO — T20: el hueco iso-parámetro con una línea base sana y la mejor de cuatro candidatas (E1 de RFC-CALIDAD-1)

**Fecha:** 2026-10-06, escrito ANTES de cualquier corrida de T20 (incluido el smoke). El hash queda
en `PREREGISTRO-T20.sha256`, junto al de `analyze_t20.py`, que codifica las reglas, al de `t20.py`,
`diag_t20.py`, el paquete `mztrain_t20/`, los lanzadores, los tests, las dependencias importadas
(T9 `bank.py`, T17 `factor_opt.py`, T18 `t18_por_corrida.py`), las referencias del anclaje
(`t19_smoke.json`, `t19_results.json`), los datos (`wt2_char.pt`, `wt103_char_t20.pt`) y el paquete
`mztrain` entero. `t20.py` se niega a correr si algún fichero del último bloque de ese sello ha
cambiado; la etapa A no arranca sin un smoke completo bajo ese mismo sello; B y C no arrancan sin
la etapa anterior completa y con `t20_decisiones.json` coincidente con las reglas recalculadas.

Especificación: `RFC-CALIDAD-1.md` (secciones 3 y 5.2) y `PROMPT-T20.md`. Este documento fija lo
que el enunciado dejaba abierto y lo que se corre; donde hay diferencia, manda este documento.

## Origen y pregunta

En el montaje del arco T4–T19 (embedding atado N(0,1), AdamW β2 = 0,999, LR 3e-4 constante, sin
recorte), a 18 000 pasos el factorizado F (d=768, r=128) queda en 2,0441 BPC y el denso
iso-parámetro Ds (d=368) en 1,9579: hueco +0,086 en 3/3 semillas (M03). Ese montaje arranca de
forma patológica en proporción a la anchura (BPC inicial ~700 a d=768, ~330 a d=368; M19) y los dos
modelos arrastran un «sumidero» (vector constante que domina el flujo residual; M22–M24). Que el
hueco esté confundido con esa patología es una inferencia: nadie ha entrenado con otro arranque.

**Pregunta principal.** Con una línea base sana («★», igual para todos), a 18 000 pasos y con tres
semillas, ¿cuánto vale el hueco F★ − Ds★? ¿Y alguna de las cuatro candidatas (carriles con bus L,
forma F576, calibre polar Fpolar, no linealidad entre factores Fcola) queda por debajo de Ds★ en 3/3?

**Hipótesis.**
- H_★ (el hueco era del montaje): con ★, F★ < Ds★ en 3/3. Se refuta si Ds★ ≤ F★ en alguna semilla.
- H_X (una candidata cierra el hueco): la finalista X < Ds★ en 3/3. Se refuta si X ≥ Ds★ en alguna.
- H_sumidero (diagnóstico de la expedición): con ★, F★ no reproduce el cuadro del arco. Se refuta si
  las tres finales de F★ dan rms del flujo final > 100, fracción constante > 85 % y parte
  contextual < 0,2 % (sección «Lecturas en CPU»).

La métrica es la de T16–T19, sin cambios: BPC de validación completo (`bank.val_bpc_full`, ventanas
no solapadas de 128 sobre el fichero de validación de WT-2, fp32, chunk 64).

## Línea base ★ (igual para todos los brazos)

| Pieza | Valor | Hecho en que se apoya |
|---|---|---|
| Embedding | `tok` y `pos` ~ N(0, 0,02²), generador propio (semilla 2000 + semilla de la corrida) que no consume el global; el resto del init como en T19 (`fact_lin`, `dense_lin` tal cual) | [R13, R14]; BPC inicial 10,2–11,1 [M20] |
| Optimizador | AdamW, β = (0,9, 0,95), eps 1e-8, wd 0,01 en todos los parámetros | [R12, R8]; F necesita el weight decay [M07] |
| Recorte | norma global del gradiente a 1,0, entre `backward` y `step` (se registra la norma previa) | [R11] |
| LR | rampa de 200 pasos desde 0,1·pico (convención de pasos de `LrWarmup`: tras t pasos, 0,1 + 0,9·min(1, t/200)); después, coseno hasta 0 en el último paso del presupuesto (6000 o 18 000) | [R16] |
| LR pico | calibrado por brazo en {3e-4, 6e-4, 1,2e-3, 2,4e-3} (etapa A) | [R17]; depende de la escala [M39] |
| Receta de T18 y promedio de pesos | fuera | la ganancia decae sin meseta [M09] |
| Corpus | char-WT103: primeros 120 000 000 caracteres de train, vocabulario de WT-2 (1118) más `<otro>` (id 1118; 7009 apariciones, el 0,006 %); validación: el tensor de WT-2 (idéntico al de WT-103). En GPU en int16; lotes a `long` | [M15, M38, M39] |
| Resto | lote 16 × 128, fp32, semillas 17, 29 y 43, generador de lotes 1000 + semilla (mismos índices que `bank.get_batch`) | protocolo T16–T19 |

El presupuesto de 18 000 pasos muestrea el 31 % del trozo de WT-103 (en WT-2 serían 3,4 épocas).

## Brazos (iso-parámetro; vocabulario 1119)

| Brazo | Forma | Parámetros | Productos por token (cuenta del RFC) | Qué cambia respecto a F |
|---|---|---|---|---|
| `Ds` | denso d=368, L8, H8 | 13 472 112 | 14,17M | línea base densa |
| `F` | factorizado d=768, r=128 | 13 570 816 | 15,01M | — |
| `L` | d=768; cada lineal = BD (G=8, un bloque por cabeza) + bus `ZFactorizedLinear` r=56; init BD como `nn.Linear` con el fan-in del bloque, bus `svd`; ambas partes × 1/√2 | 13 568 512 (−2304) | como F | deja de ser bajo rango puro |
| `F576` | factorizado d=576, r=173 | 13 498 336 | 14,58M (−3 %) | ahorro frente al denso de igual forma: 4,5× → 2,5× |
| `Fcola` | `cola_lin(128)` de T17: y = U·(S ∘ gelu(V·x)) | 13 570 816 | como F | la capa deja de ser lineal |
| `Fpolar` | F reparametrizado Û·diag(e^σ)·V̂, normas de columnas de Û y filas de V̂ fijas a 1; mismo init que F (σ = ln S) | 13 570 816 | como F (+0,02 % por renormalizar) | tres relojes: η_U = η₀·√(768/out), η_V = η₀·√(768/in), η_σ = 10·η₀; sin wd en U, V, σ; σ ← σ + ln(1 − 3·wd·η_F(t)) tras cada paso, con η_F(t) el LR del brazo F ese paso |

Orden del paso en `Fpolar`: backward → quitar la parte radial de ∂L/∂U (por columna) y ∂L/∂V (por
fila) → recorte global 1,0 → step de AdamW → renormalizar → decaer σ. Por homogeneidad,
û·∂L/∂û = v̂·∂L/∂v̂ = ∂L/∂σ, así que la parte radial es redundante con el gradiente de σ.

Alineación de los carriles con el GPT de `mztrain.elastic_shape` (qkv fusionado `[q | k | v]`, cada
uno por cabezas contiguas de 96): el bloque g de `qkv` lee las dims [96g, 96g+96) y escribe las filas
blk·768 + 96g + j (q, k y v de la cabeza g); `proj` devuelve la cabeza g a su carril; `fc1` lleva
el carril g a las neuronas [384g, 384g+384); `fc2` las devuelve. Comprobado en CPU: con el bus a
cero, perturbar el carril g de la entrada de un bloque solo cambia el carril g de su salida.

## Smoke (GPU; no es dato; solo tras el «corre»)

1. **Anclaje.** El harness configurado como T19 (WT-2, embedding N(0,1), AdamW β2 = 0,999, LR 3e-4
   con `LrWarmup(200, 0,1)` y constante después, sin recorte) repite `Ds/s17` de T19 **bit a bit**
   en los pasos 250 y 500: 4,138897 y 3,822008 (`t19_smoke.json`; el 500 coincide con
   `t19_results.json`). Un desajuste o un NaN para todo (salida 3) y se investiga.
2. **Los seis brazos bajo ★**, 500 pasos con coseno sobre 500, LR 6e-4 (Fpolar 1,2e-3), semilla 17,
   evaluación cada 250: pérdida finita y minutos por 1000 pasos de cada uno.
3. **Reanudación** desde un checkpoint del paso 150 (dentro de la rampa): idéntica en `Ds`
   (exigible: si no lo es, salida 5, el smoke no cuenta como completo y la etapa A no arranca);
   en el resto se reporta (F no es determinista a d=768, M14). Un smoke anterior queda archivado
   como `t20_smoke.prev-<fecha>.json`, nunca pisado.

## Diseño (20 corridas; 343,8–360,0 min con los costes planos de 9 y 27 min por corrida y 1,2× para L)

Cada etapa tiene modo obligatorio y explícito (`--etapa A|B|C`), verifica el sello y que la
anterior está completa. Las decisiones entre etapas las calcula `analyze_t20.py` (funciones puras
sobre `t20_results.json`) y quedan escritas en `t20_decisiones.json` antes de arrancar la
siguiente; la siguiente las recalcula y se niega a arrancar si no coinciden. No interviene ningún
juicio humano.

- **Etapa A, calibración** (6000 pasos, semilla 17, coseno sobre 6000; 7 corridas): `Ds` y `F` con
  6e-4 y 1,2e-3, en este orden: Ds 6e-4, F 6e-4, Ds 1,2e-3, F 1,2e-3; después, cada uno con 2,4e-3
  si ganó 1,2e-3 o con 3e-4 si ganó 6e-4; y `Ds` con β2 = 0,999 en su mejor LR.
- **Etapa B, criba** (6000 pasos, semilla 17; hasta 4 corridas): las candidatas que deja la regla
  del reloj, en el orden Fpolar, L, Fcola, F576, al LR de `F` (Fpolar al doble; su decaimiento de
  σ usa el LR de `F`).
- **Etapa C, finales** (18 000 pasos, semillas 17, 29 y 43; 9 corridas): `Ds` (su LR y su β2), `F`
  (su LR, β2 = 0,95) y la finalista X (el LR con el que pasó), con los brazos rotados por semilla y
  `Ds/s17` primero: s17 [Ds, F, X]; s29 [F, X, Ds]; s43 [X, Ds, F]. Sin finalista: solo `Ds` y `F`
  (6 corridas).
- Evaluación de validación cada 500 pasos; BPC de train sobre 64 ventanas fijas (semilla 1900)
  cada 1000 pasos y en el último; checkpoint final de **todas** las corridas en `ckpt/` (modelo,
  optimizador, calendario, generador); por corrida: VRAM de pico, estado de AdamW, parámetros,
  minutos por 1000 pasos (reloj de pared de la corrida, con evaluaciones), norma del gradiente
  antes del recorte (media y máximo por tramo de 500) y fracción de pasos recortados.
- Freno térmico adaptativo desde el primer paso (controlador de T18: pausa por paso si la GPU está
  ≥ 76 °C, hasta 0,3 s); `t20.py` no entrena si el hilo del freno no está vivo; enfriado a ≤ 60 °C
  antes de cada corrida. Vigilante externo `vigia_t20.ps1`: aviso 78 °C; parada a 83 °C sostenida
  3 lecturas, a 86 °C inmediata o tras 6 lecturas sin dato; enfría a 60 °C y relanza la misma
  etapa; máximo 2 reinicios. **El freno invalida tok/s: este experimento no hace claims de
  velocidad**; los minutos por 1000 pasos solo sirven a la regla del reloj.

## Reglas de decisión (fijadas antes; código: `analyze_t20.py`)

El BPC de decisión de una corrida es su BPC de validación en el último paso. Pérdida no finita,
corrida incompleta o ausente cuentan como +∞; «degenerada» si termina en ≥ 3,0 (frontera inclusiva,
tolerancia 1e-9 en todas las comparaciones).

1. **LR de un brazo (A):** el de menor BPC de sus tres corridas; en empate, el menor LR (también
   para elegir el tercero: si 6e-4 y 1,2e-3 empatan, gana 6e-4 y se corre 3e-4). Si el elegido
   está en el borde de la rejilla {3e-4 … 2,4e-3}, se declara, sin más corridas.
2. **β2 del denso (A):** el de menor BPC entre 0,95 y 0,999; en empate, 0,95. Solo el denso recibe
   este control: es un sesgo a su favor, y se declara.
3. **Reloj (tras A):** total = minutos reales de A (reloj de pared de sus 7 corridas, sin las
   esperas de enfriado) + proyección de B y C con los minutos por 1000 pasos medidos (media de
   las corridas de A de cada brazo; m_F para Fpolar, Fcola y F576; 1,2·m_F para L; la finalista
   se proyecta con el mayor coste de las candidatas que quedan). Si pasa de 360 min, se quitan
   candidatas de la criba por este orden: Fpolar, Fcola, F576. Nunca se quitan la calibración,
   L ni las finales de Ds y F. Si aun así supera el tope, se declara y se corre.
   Una candidata con BPC no finito en la criba no pasa la puerta, aunque F haya fallado.
4. **Puerta de la criba (B):** pasa la candidata con BPC ≤ BPC(F en A, su mejor corrida) − 0,03.
   Finalista: la de menor BPC; si otra difiere de ella en menos de 0,01, la primera en el orden
   Fpolar, L, Fcola, F576. Usa en la final el LR con el que pasó.
5. **VICTORIA:** X < Ds en 3/3, pareado por semilla, en el paso 18 000.
6. **EL HUECO ERA DEL MONTAJE:** F < Ds en 3/3, con Ds válido (finito y no degenerado) en las
   tres semillas (se reporta junto a la etiqueta de X).
7. **RECORTA:** X < F en 3/3 y media(F − X) ≥ 0,02, sin victoria. Una F fallida cuenta +∞ en
   esa semilla y la media de decisión es +∞; se reporta también la media sobre las semillas con F
   válida.
8. **MUERE:** cualquier otro resultado de X. Las candidatas que no fueron a la final quedan como
   «no pasó la criba», con su diferencia frente a F; las que quitó el reloj, como «no corrida».
9. **INCONCLUSO:** si Ds da pérdida no finita o termina degenerado en alguna semilla, no se
   declara victoria (la etiqueta pasa de VICTORIA a INCONCLUSO; RECORTA y MUERE, que comparan con
   F, se mantienen). Con F fallida en una semilla, F cuenta +∞ en esa semilla.
10. **Sin finalista:** la etapa C corre Ds y F; la etiqueta de X es SIN_FINALISTA y la conclusión
    es que ninguna de las cuatro causas mueve a F a 6000 pasos.

**Descriptivos (no cambian la etiqueta).** Sobreajuste: si el BPC de validación de un brazo sube
más de 0,02 desde su mínimo o la brecha val − train supera 0,15 en una corrida de la final, se
reporta también la etiqueta en el mínimo de validación de cada corrida. Baches: saltos de más de
0,3 BPC entre dos evaluaciones consecutivas. Tabla por brazo y semilla a 6k/12k/18k, hueco pareado
y su media, pendiente de los últimos 1000 pasos.

**Sin exclusiones ni repesca.** NaN o degeneración cuentan como fallo de esa semilla. Sin reajuste
de hiperparámetros tras ver datos. Una corrida cortada por el freno o el vigilante se repite
idéntica desde cero; queda anotada en `t20_results.json` («reintentos») y se declara en
`DESVIACIONES-T20.md`. Toda comparación se declara «a presupuesto fijo» (6000 o 18 000 pasos).

## Lecturas en CPU (preregistradas; sin efecto en las etiquetas; `diag_t20.py`)

Sobre los checkpoints finales y los de F y Ds de la etapa A: rms del flujo residual final,
fracción constante de su energía, parte dependiente del contexto tras la LN final y su dimensión
efectiva, bloques muertos (rms de escritura < 0,05) y rms de `tok`; ortogonalidad y rango estable
de los factores; alineación entre escritura y lectura; fracción visible del gradiente (8 lotes
del corpus de T20); histograma de saliencia de F (decide E8); cuánto se mueve el bus de L desde
su init. **El diagnóstico del sumidero queda refutado si las tres finales de F reproducen el
cuadro del arco: rms final > 100, fracción constante > 85 % y parte contextual < 0,2 %.** Al lado
se ponen los valores del arco (expedición: F 273–281, 96 %, 0,10–0,13 %; Ds 66–67, 96–97 %,
0,53–0,55 %).

## Predicciones (antes de correr)

Del RFC, contrastables en E1:
- Fpolar tolera un LR pico doble que el de F★: en la criba no diverge y no tiene ningún salto > 0,3
  BPC entre evaluaciones.
- La curva temprana de L (evaluaciones hasta el paso 3000) va por debajo de la de F★ al mismo LR.
- Si F★ reproduce el cuadro del arco, el diagnóstico del sumidero queda refutado.

Probabilidades que dieron innovadores y críticos (`expedicion/`; juicios, no medidas):

| Candidata | Innovador que la propuso | Crítico A | Crítico B | Crítico C |
|---|---|---|---|---|
| Fpolar (calibre polar) | A: gana E1 ~15 %; atribuible al calibre ~6 % | — | sobrevive con cambios, sin cifra | 1/10 |
| L (carriles con bus) | B: ~15 % | 2/10 | — | 2/10 |
| F576 (la forma) | A: ~20 %; atribuible a la forma ~10 % | — | «bajar P(gana) a lo que el mecanismo sostiene», sin cifra | 2/10 |
| Fcola | adopción del director (T17), sin cifra propia | — | — | — |
| F★ sola gana a Ds★ | innovador A: ~10 % | | | |

Del autor de este preregistro (confianza baja):

| Predicción | P |
|---|---|
| Anclaje de `Ds/s17` exacto en los pasos 250 y 500 | 0,9 |
| Ds★ (18k, media) < 1,9579 y F★ < 2,0441 (los dos mejoran al arco) | 0,8 |
| Hueco F★ − Ds★ a 18k ∈ [0,02, 0,08] | 0,5; < 0,02: 0,15; > 0,08: 0,35 |
| EL HUECO ERA DEL MONTAJE (F★ < Ds★ en 3/3) | 0,10 |
| LR elegido: Ds ∈ {1,2e-3, 2,4e-3}; F ∈ {6e-4, 1,2e-3} | 0,6; 0,65 |
| β2 = 0,95 gana en Ds | 0,6 |
| El reloj no quita ninguna candidata | 0,75 |
| Alguna candidata pasa la puerta (≥ 0,03 sobre F★ a 6000) | 0,35; finalista L si alguna pasa: 0,4 |
| Fpolar no diverge al doble de LR y sin saltos > 0,3 | 0,7 |
| Etiqueta: VICTORIA 0,12; RECORTA 0,25; MUERE o SIN_FINALISTA 0,63 | |
| 0 NaN y 0 degeneradas en Ds★ (las 3 finales) | 0,9 |
| F★ no reproduce el cuadro del arco (diagnóstico no refutado) | 0,7 |
| Ninguna final con sobreajuste marcado (val sube > 0,02 o brecha > 0,15) | 0,8 |

## Consecuencias (sección 5.4 del RFC)

- **VICTORIA o RECORTA:** E10 (3× pasos) y E6 (ablación de ★) antes de anunciar nada; cualquier
  integración iría como opción desactivada por defecto y tras E10.
- **EL HUECO ERA DEL MONTAJE:** releer el arco T4–T19 entero; lo demás pasa a ser mejora, no cierre.
- **MUERE o SIN_FINALISTA:** E5 (curva de forma) y E9 (optimizador espectral).
- El claim de calidad por parámetro de MZTrain solo cambia con VICTORIA (y, para F, con el hueco
  del montaje) sostenidos en 3/3.

## Amenazas a la validez

1. Una escala (13,5M), carácter, fp32, seq 128.
2. La criba elige con una semilla a 6000 pasos: las finales protegen la afirmación, no la elección.
   La puerta de 0,03 equivale a 1,9–2,5 desviaciones del hueco entre semillas del arco.
3. El LR elegido a 6000 pasos se transfiere a 18 000 apoyándose en [R16] (medido con decaimiento
   lineal; se supone que vale con coseno).
4. El denso se calibra solo en LR y β2; weight decay, rampa y umbral de recorte no se barren.
5. F576 usa el LR de F★ (sesgo en su contra, declarado); Fpolar corre a un solo LR (el doble).
6. El cambio de corpus rompe la comparación directa con las cifras absolutas del arco; el smoke
   conserva el anclaje bit a bit con T19 (WT-2) para garantizar que el harness es el mismo.
7. F no es determinista a d=768 (M14); no se sabe si lo es bajo ★. El anclaje bit a bit solo es
   exigible al denso.
8. A iso-parámetro con embedding, los brazos de d=768 tienen un 3,1 % menos de parámetros fuera
   del embedding que Ds: sesgo pequeño en contra de F, no corregido.
9. Tres semillas: σ con n=3 es una estimación pobre.
10. Los minutos por 1000 pasos llevan el freno: la regla del reloj es conservadora, no una medida.

## Reglas operativas

- **No se lanza nada en GPU hasta que el usuario diga «corre»**, tampoco el smoke. Todo se prueba
  en CPU (`CUDA_VISIBLE_DEVICES=-1`), se sella y se espera.
- No se toca `D:\mztrain` ni las carpetas T9 y T16–T19: se importan, sin bytecode.
- Ningún subagente ejecuta `t20.py` ni los lanzadores; los revisores trabajan con réplicas.
- Para parar: `parar_t20.ps1` (mata el árbol, lista lo que queda, comprueba la GPU).
- Consola cp1252: nombres ASCII en código y logs; `PYTHONIOENCODING=utf-8`.

## Artefactos

- `t20.py`, `analyze_t20.py`, `diag_t20.py`, `mztrain_t20/` (`datos.py`, `sano.py`, `carriles.py`,
  `polar.py`), `vigia_t20.ps1`, `lanzar_t20.ps1`, `parar_t20.ps1`, `tests/` (todo en CPU; el
  vigilante con temperaturas simuladas; integración vigilante ↔ cadena con disparo térmico a mitad
  de una corrida de la etapa B).
- `wt103_char_t20.pt` (corpus; sellado), `t20_smoke.json`, `t20_results.json`,
  `t20_decisiones.json`, `t20_analysis.json`, `diag_t20.json`, `freno_t20.csv`, `vigia_t20.log`,
  `log_T20_*.txt`, `ckpt/`, `ckpt_smoke/`.
- `DESVIACIONES-T20.md` y `T20-VEREDICTO.md`.
