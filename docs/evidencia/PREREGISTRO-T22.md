# PREREGISTRO — T22: ¿un optimizador espectral cierra lo que AdamW no? (E9 de RFC-CALIDAD-1)

**Fecha:** 2026-10-07, escrito ANTES de cualquier corrida de T22 (incluido el smoke). El hash queda
en `PREREGISTRO-T22.sha256`, junto al de `analyze_t22.py` (las reglas), `t22.py`, `diag_t22.py`, el
paquete `mztrain_t22/` (los optimizadores), los lanzadores, los tests, el harness sellado de T20 que
T22 importa (`t20.py`, `analyze_t20.py`, `diag_t20.py`, `mztrain_t20/`), `t20_results.json` y
`PREREGISTRO-T20.sha256` (las corridas de T20 que se reutilizan y el sello bajo el que se hicieron),
el corpus, las dependencias de T9/T17/T18/T19 y el paquete `mztrain` entero. `t22.py` se niega a
correr si algún fichero del último bloque del sello ha cambiado o si el harness de T20 ya no es el
de su propio sello v1 (`t20.verificar_sello`: cadena de custodia de las corridas reutilizadas); la
etapa A no arranca sin un smoke completo bajo el mismo sello; la B no arranca sin la A completa y con `t22_decisiones.json`
coincidente con las reglas recalculadas.

**Origen.** T20 (E1) cerró SIN FINALISTA: con la línea base sana ★ el factorizado F (768, 128) queda
+0,051 BPC por detrás del denso iso-parámetro Ds (368) en 3/3 a 18 000 pasos. T21 (E5) midió la
curva de forma: subir r/d mejora a F de forma monótona pero se satura pronto (la mejor forma, F448,
a −0,024 de F★, no pasa la puerta), el denso con la forma de F también pierde con Ds★, y las
lecturas en CPU apuntan a que el rango estable de los factores no crece con el rango disponible.
El RFC (secciones 2.2, 2.3 y 5.4) manda E9: con el mismo modelo y los mismos parámetros, cambiar solo
el optimizador de los factores (Spectron) bajó la PPL entre un 6 % y un 17 % frente a AdamW a 94M–454M
[R1; S2-A13], y su ablación atribuye la mayor parte a la ortogonalización (Muon), que es aplicable al
denso [R1; D05]. La comparación justa exige calibrar otro optimizador en los dos brazos.

**Preguntas.** (1) ¿Spectron, adaptado a la forma U·diag(S)·V, mejora a F★ a 18 000 pasos con tres
semillas, y cuánto del hueco F★ − Ds★ cierra? (2) ¿Muon mejora al denso Ds★? (3) ¿F con Spectron
gana al mejor denso (Ds★ o Ds con Muon) en 3/3?

**Hipótesis.**
- H_spec (el hueco es de optimización): Fspec cierra al menos la mitad del hueco de T20: media por
  semilla de (Fspec − mejor denso) ≤ 0,5 × media de (F★ − Ds★) = 0,0253. Se refuta si no.
- H_muon (la ortogonalización también ayuda al denso): Dsmuon < Ds★ en 3/3 y ≥ 0,01 de media. Se
  refuta si no.
- H_X (victoria): Fspec < min(Ds★, Dsmuon) en 3/3, pareado por semilla.

Métrica: la de T16–T21, sin cambios (BPC de validación completo, `bank.val_bpc_full`, fp32).

## Diseño

Línea base ★, corpus, lote, semillas y harness **idénticos a T20** (`t20.py` importado, no copiado:
T22 añade dos brazos a `t20.ARMS` y sustituye `t20.build` para montar el optimizador espectral;
`t20.pasos` y `t20.run` no cambian). Los modelos y su init son exactamente los de Ds y F con la misma
semilla (test). Las corridas de T20 que cumplen el mismo protocolo se **reutilizan**, con su clave y
el sello v1 de T20 anotados: Ds★ a 6e-4 y F★ a 3e-4 de la etapa A (6000 pasos, semilla 17) y las
seis finales de Ds★ y F★ (etapa C, 18 000 pasos, semillas 17/29/43).

| Brazo | Modelo | Optimizador de las matrices de los bloques | AdamW ★ (LR fijo, el calibrado en T20) |
|---|---|---|---|
| Dsmuon | Ds: denso d=368 (13 472 112) | Muon en las 32 matrices (qkv, proj, fc1, fc2) | tok, pos, LayerNorm: 6e-4 (Ds★) |
| Fspec | F: factorizado (768, r128) (13 570 816) | Spectron adaptado en U y V de las 32 capas | tok, pos, LayerNorm y S: 3e-4 (F★) |
| Ds★ (referencia, T20) | denso 368 | AdamW ★ a 6e-4 | — |
| F★ (referencia, T20) | (768, 128) | AdamW ★ a 3e-4 | — |

**Los optimizadores** (`mztrain_t22/espectral.py`; constantes en `t22_results.json → protocol.espectral`):
- Común a los dos (Jordan et al. 2024; Liu et al. 2025): momento con Nesterov, μ = 0,95 (buf ← μ·buf +
  g; u = g + μ·buf); ortogonalización de u con 5 iteraciones de Newton-Schulz (coeficientes 3,4445,
  −4,7750, 2,0315; normalización de Frobenius con ε = 1e-7; traspuesta si hay más filas que columnas),
  en fp32; **escala RMS de Moonshot**: la actualización ortogonalizada se multiplica por
  0,2·√max(m, n), con lo que su RMS por entrada es ≈ 0,2·η, la de AdamW, y η y el weight decay se
  expresan en las unidades de AdamW de ★; weight decay desacoplado 0,01 (W ← W·(1 − η·wd)), el de ★.
  Recorte global del gradiente a 1,0 antes del paso, como en ★: para la parte ortogonalizada,
  Newton-Schulz normaliza la escala del momento, así que el recorte solo entra por la mezcla, dentro
  del momento, de gradientes recortados con factores distintos; para el grupo AdamW es el de ★.
- **Muon** (Dsmuon): W ← W·(1 − η·wd) − η·0,2·√max(m, n)·NS5(u).
- **Spectron adaptado** (Fspec; Algoritmo 1 de [R1], escrito para un producto W = A·B, llevado a
  tres factores): a primer orden ΔW = ΔU·diag(S)·V + U·diag(S)·ΔV, así que las normas que acotan la
  actualización del producto son ‖diag(S)·V‖₂ para el paso de U y ‖U·diag(S)‖₂ para el de V. Se
  definen σ_A := ‖U·diag(S)‖₂ y σ_B := ‖diag(S)·V‖₂ (no es que W = A·B), estimadas por iteración de
  potencia con arranque caliente (10 iteraciones al arrancar desde un vector aleatorio de un generador
  propio, semilla 3000 + semilla; después 1 por paso; la estimación queda por debajo de la norma
  exacta), leídas ANTES del paso; U ← U·(1 − η·wd) − η·0,2·√max(out, r)·NS5(u_U)/(σ_A + σ_B + 1), y
  V igual con √max(r, in) y NS5(u_V). S es 1-D y, como en Muon todo lo que no es matriz, va en
  AdamW ★ con el LR de F★ (3e-4), igual que tok, pos y las LayerNorm. Spectron publica el momento
  como EMA sin Nesterov: tras Newton-Schulz la dirección es la misma salvo el término de Nesterov,
  que aquí se usa en los dos brazos por uniformidad. Con la escala RMS, la cota de [R1] sobre la
  actualización del producto pasa de η a η·0,2·√max(m, n), a primer orden y salvo el sobreimpulso
  de Newton-Schulz (valores singulares de NS5 hasta ≈ 1,2) y el paso simultáneo de AdamW sobre S;
  se encoge cuando crecen las normas de los factores. **El weight decay no lleva el factor
  1/(σ_A + σ_B + 1)** (forma publicada: decaimiento desacoplado con η): como el paso sí lo lleva
  (≈ 0,14–0,24 al arrancar según la capa), la razón decaimiento/paso en U y V es 4–7 veces la de
  Muon o la de AdamW ★ al mismo η. Es una elección declarada (amenaza 8); las normas de U y V se leen.
- Unidades: el RMS por entrada de la actualización de [R1] es η_R1/√max(m, n) y el de aquí
  0,2·η (el factor 1/(σ_A + σ_B + 1) es común y se cancela), así que η_R1 = 0,01, el valor que [R1]
  muestra estable en su ablación de LR (Figura 12), equivale a η = 0,01/(0,2·√max(m, n)): 1,8e-3 en
  los factores de lado 768 y 0,9–1,0e-3 en los de lado 2304–3072. La rejilla de abajo está centrada
  en ese rango.
- El estado del optimizador (momentos, vectores de potencia, AdamW) entra en el checkpoint; la
  reanudación es idéntica en CPU (tests) para los dos brazos.

**Smoke (GPU; no es dato; solo tras el «corre»).** Dsmuon y Fspec a 1,2e-3, 500 pasos con coseno
sobre 500, semilla 17; pérdida finita y minutos por 1000 pasos de cada uno (los de Ds y F del smoke de T20
se copian al lado en `t22_smoke.json`, solo para el reloj); reanudación desde el paso 150 **reportada, no exigible**
(como en T21: en GPU solo el denso de 368 con AdamW fue bit a bit determinista; aquí ningún
reintento reanuda desde un checkpoint). Un smoke anterior se archiva, no se pisa.

**Etapa A, calibración por caminata** (6000 pasos, semilla 17, coseno sobre 6000; 6 a 8 corridas,
~1,5–2,5 h con el freno). Para cada brazo, η del optimizador espectral (el grupo AdamW queda fijo):
1. dos LRs iniciales, **1,2e-3 y 2,4e-3** (intercalados: Fspec 1,2e-3, Dsmuon 1,2e-3, Fspec 2,4e-3,
   Dsmuon 2,4e-3);
2. el tercero: el doble del mayor si ganó el mayor; la mitad del menor si ganó el menor o hubo empate
   (también entre dos +∞);
3. un cuarto, en la misma dirección, solo si el tercero ganó; nunca más de 4 por brazo. Un óptimo en
   el extremo de una caminata de 4 se declara «en el borde».
El RFC pedía 3 LRs por brazo; el cuarto condicional es una extensión declarada (+10–15 min) porque
la escala del LR de un optimizador nuevo es la mayor incertidumbre del diseño.

**Etapa B, finales** (18 000 pasos, semillas 17/29/43; 6 corridas, ~3,5–4,5 h con el freno): Fspec y
Dsmuon, cada uno a su mejor LR de la etapa A, rotados por semilla (17: Fspec, Dsmuon; 29: Dsmuon,
Fspec; 43: Fspec, Dsmuon). No hay puerta: el RFC fija 12 corridas y la medida a 18 000 pasos con
tres semillas es el resultado de E9 sea cual sea el signo. Un brazo solo va a la final si su mejor
corrida de la A es válida (finita y no degenerada). Si Fspec pasa la puerta de E1 (≤ F★ − 0,03 a
6000) se anota como descriptivo.

## Reglas de decisión (fijadas antes; código: `analyze_t22.py`)

El BPC de decisión es el de validación en el último paso; no finito, incompleta, ausente o
degenerada (≥ 3,0) cuentan +∞ en todas las reglas. Fronteras inclusivas con tolerancia 1e-9. Las
claves de la final llevan el prefijo `Bf.`.

1. **Mejor LR de un brazo:** el de menor BPC entre los corridos; en empate, el menor.
2. **Denso de referencia, por semilla:** el menor válido entre Ds★ (final de T20) y Dsmuon (final de
   T22). Si Dsmuon no tuvo final o falla en una semilla, en esa semilla la referencia es Ds★.
3. **VICTORIA:** Fspec < denso de referencia en 3/3, pareado por semilla, con Fspec válida en las
   tres. **INCONCLUSO** la sustituye si en alguna semilla no hay denso válido.
4. **RECORTA:** Fspec < F★ (final de T20) en 3/3, con F★ válida en las tres, y media(F★ − Fspec) ≥
   0,02, sin victoria.
5. **MUERE:** cualquier otro resultado, incluido que Fspec no tenga final válida (se anota el motivo).
6. **MUON AYUDA AL DENSO** (descriptivo; no cambia la etiqueta): Dsmuon válida en 3/3, Dsmuon < Ds★
   en 3/3 y media(Ds★ − Dsmuon) ≥ 0,01. **MUON PERJUDICA AL DENSO**: Dsmuon válida en 3/3 y Dsmuon >
   Ds★ en 3/3 (una final no válida no es evidencia de perjuicio: se reporta como fallo, aparte).
7. **CIERRA LA MITAD DEL HUECO** (H_spec; descriptivo): media(Fspec − denso de referencia) ≤ 0,5 ×
   media(F★ − Ds★). Se reporta la fracción del hueco cerrada, 1 − hueco_spec/hueco★ (puede ser > 1 o
   negativa).

**Descriptivos (no cambian la etiqueta):** a 6000 pasos, Fspec − F★, Fspec − Ds★, Dsmuon − Ds★ y la
puerta de E1; la curva media por paso (6000/12 000/18 000) de los cuatro brazos de la final; el
tamaño del estado del optimizador (Muon y Spectron no llevan segundo momento en las matrices);
sobreajuste (subida > 0,02 desde el mínimo o brecha val − train > 0,15) y baches (> 0,3) por corrida;
las lecturas en CPU (`diag_t22.py`): las de `diag_t20` sobre Fspec y Dsmuon (flujo, factores,
alineación, gradiente visible, saliencia de S) más las normas espectrales (σ_A, σ_B, ‖U‖₂, ‖V‖₂ en
Fspec; ‖W‖₂ y rango estable en Dsmuon) y, sobre los checkpoints de F★ y Ds★ de T20, las mismas
normas y, en F★, la ortogonalidad y el rango estable de los factores con el mismo código.
El RFC predice que la ortogonalización mantiene U y V cerca de la ortonormalidad y mueve el espectro
de W (M27: con AdamW el rango estable cae de 91–105 a 5–16): `ortog_U`, `ortog_V` y `rango_estable`
lo miden.

**Sin exclusiones ni repesca.** NaN o degeneración cuentan como fallo de esa corrida (y +∞ en las
reglas). Una corrida cortada por el freno o el vigilante se repite idéntica desde cero y se anota
en «reintentos» y en `DESVIACIONES-T22.md`. Toda comparación se declara «a presupuesto fijo». Los
minutos llevan el freno térmico: no hay claims de velocidad ni de sobrecoste.

## Predicciones (antes de correr; confianza baja)

| Predicción | P |
|---|---|
| mejor LR de Fspec: 2,4e-3 | 0,35; 1,2e-3: 0,35; 4,8e-3: 0,20; 6e-4: 0,10 |
| mejor LR de Dsmuon: 2,4e-3 | 0,35; 1,2e-3: 0,35; 4,8e-3: 0,20; 6e-4: 0,10 |
| alguna caminata llega a 4 LRs | 0,5 |
| Fspec pasa la puerta de E1 a 6000 (≤ F★ − 0,03) | 0,40 |
| RECORTA o mejor (Fspec < F★ en 3/3 y ≥ 0,02) | 0,45 |
| CIERRA LA MITAD DEL HUECO | 0,25 |
| VICTORIA | 0,10 |
| MUON AYUDA AL DENSO | 0,50; PERJUDICA: 0,15 |
| 0 NaN y 0 degeneradas en A y B | 0,60 (optimizador nuevo; [R1] publica estabilidad a LR alto) |
| ortog_U y ortog_V de Fspec por debajo de los de F★ (más ortonormales) | 0,70 |
| rango estable de W en Fspec mayor que en F★ en los cuatro tipos | 0,55 |
| el reloj: etapa A entre 1,5 y 2,5 h; etapa B entre 3,5 y 5 h, con el freno | 0,6 |

Del RFC (2.2, 2.3): la ganancia publicada de Spectron sobre AdamW es del 6–17 % de PPL a 94M–454M en
subpalabras; a 13M en carácter, con S entrenable y con AdamW ya calibrado y recortado (★), es una
extrapolación (H15 del mapa: sin precedente por debajo de 60M ni sobre tres factores).

## Amenazas a la validez

1. Una escala, carácter, fp32, seq 128; una semilla en la calibración; tres en la final.
2. La caminata de LR cubre como mucho un factor 32 (de 3e-4 a 9,6e-3, nunca los dos extremos a la
   vez) y puede acabar en un borde (se declara).
3. El grupo AdamW (embedding, LayerNorm, S) queda al LR calibrado en T20 para su brazo y no se
   recalibra junto al espectral: aísla el efecto del optimizador de las matrices, pero una
   interacción con el LR del embedding no se mide.
4. Spectron se adapta a tres factores y se expresa con la escala RMS de Moonshot: no es la receta
   publicada al pie de la letra (S en AdamW, Nesterov, cota reescalada). Es una PROPUESTA declarada.
5. Las corridas reutilizadas de T20 son del mismo harness y las mismas semillas, en otra sesión; F no
   es determinista a d=768 en GPU.
6. El coste por paso de Newton-Schulz es mayor en el denso (matrices 368 × 1104–1472) que en el
   factorizado (factores 128 × n); los minutos no se comparan.
7. Muon y Spectron no llevan segundo momento en las matrices: la memoria del estado baja; no es
   una medida de calidad y no entra en las etiquetas.
8. En Fspec el weight decay de U y V no lleva el factor 1/(σ_A + σ_B + 1) del paso: a igual η,
   U y V decaen entre 4 y 7 veces más por unidad de paso que en Muon o en AdamW ★. Si Fspec pierde,
   una parte puede ser de ese decaimiento y no de la ortogonalización; lo distinguiría una
   ablación (wd = 0 o wd escalado por el mismo factor), fuera de E9.

## Reglas operativas

- **No se lanza nada en GPU hasta que el usuario diga «corre»**, tampoco el smoke.
- No se toca `D:\mztrain` ni las carpetas T9 y T16–T21: se importan o se leen. Los checkpoints y
  resultados de T22 van en su carpeta.
- Ningún subagente ejecuta `t22.py`, `t20.py` ni los lanzadores; los revisores trabajan con réplicas.
- Para parar: `parar_t22.ps1`.

## Artefactos

`t22.py`, `analyze_t22.py`, `diag_t22.py`, `mztrain_t22/espectral.py`, `vigia_t22.ps1`,
`lanzar_t22.ps1`, `parar_t22.ps1`, `tests/`; `t22_smoke.json`, `t22_results.json` (con la referencia
de T20 dentro), `t22_decisiones.json`, `t22_analysis.json`, `diag_t22.json`, `freno_t22.csv`,
`vigia_t22.log`, `log_T22_*.txt`, `ckpt/`, `ckpt_smoke/`; `DESVIACIONES-T22.md` y `T22-VEREDICTO.md`.
