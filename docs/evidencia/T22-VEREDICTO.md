# T22 — Veredicto: el optimizador espectral mejora a los dos brazos por igual (≈ 0,065 BPC) y el hueco a iso-parámetro no se mueve; Fspec recorta a F★ y gana al denso AdamW, pero pierde con el denso Muon en 3/3

**Fecha:** 2026-10-08. **Experimento:** E9 de `RFC-CALIDAD-1.md` (sección 5.4; mecanismo en 2.2 y 2.3).

**Preregistro:** `PREREGISTRO-T22.md`, sellado antes del smoke y de cualquier corrida
(`PREREGISTRO-T22.sha256`, bloque v1 de 2026-10-07 17:45, 59 ficheros: harness de T22, harness
sellado de T20 que importa, `t20_results.json` y el sello de T20, tests, dependencias, corpus y
`mztrain` entero). Los dos sellos (T22 v1 y T20 v1) siguieron vigentes al terminar; `t22.py`
comprobó los dos en cada lanzamiento.

**Desviaciones:** `DESVIACIONES-T22.md`: **ninguna de protocolo.** Un incidente operativo (§1): el
monitor de la sesión seguía `vigia_t22.log` con `tail -F`, que bloquea la escritura del vigía; el
vigía siguió vigilando y solo perdió líneas de su log (corregido a las 18:19). Smoke y cadena en un
solo lanzamiento cada uno, 0 cortes, 0 reintentos, 0 NaN, 0 degeneradas, 13/13 checkpoints, stderr
vacío, 0 disparos térmicos (máximo 81 °C).

**Datos:** `t22_smoke.json`, `t22_results.json` (13 corridas nuevas y, dentro, las 8 de T20 que se
reutilizan con su clave y su sello), `t22_decisiones.json` (decisiones de las reglas, sin juicio
humano), `t22_analysis.json`, `diag_t22.json`, `freno_t22.csv`, `vigia_t22.log`, `log_T22_*.txt`,
`ckpt/` (13 checkpoints), `ckpt_smoke/`. Tablas del anexo: `tablas_t22.py` → `t22_tablas.md`.

**Protocolo.** Línea base sana ★, corpus, lote, semillas y harness de T20 (importado, no copiado).
Dos brazos nuevos con **el mismo modelo y el mismo init** que Ds★ y F★: `Dsmuon` (Muon en las 32
matrices del denso) y `Fspec` (Spectron adaptado a U·diag(S)·V en las 32 capas del factorizado; S,
embedding y LayerNorm en AdamW ★ al LR calibrado en T20). Etapa A: caminata del LR espectral a
6000 pasos con semilla 17 (dos iniciales, tercero hacia el ganador, cuarto si el tercero gana).
Etapa B: finales de los dos brazos a 18 000 pasos con tres semillas, rotadas, sin puerta.
**Todo es «a presupuesto fijo» y las referencias de T20 son las finales de Ds★ y F★.**

## Resultado

Finales a 18 000 pasos (BPC de validación, último paso):

| Brazo | Optimizador de las matrices | LR | s17 | s29 | s43 | media |
|---|---|---|---|---|---|---|
| Dsmuon | Muon | 1,2e-3 | **1,6228** | **1,6259** | **1,6216** | **1,6234** |
| Fspec | Spectron adaptado | 4,8e-3 | 1,6749 | 1,6774 | 1,6786 | 1,6770 |
| Ds★ (T20) | AdamW ★ | 6e-4 | 1,6847 | 1,6919 | 1,6919 | 1,6895 |
| F★ (T20) | AdamW ★ | 3e-4 | 1,7413 | 1,7368 | 1,7421 | 1,7401 |

Diferencias pareadas por semilla:

| | s17 | s29 | s43 | media |
|---|---|---|---|---|
| Fspec − Dsmuon (denso de referencia en las tres) | +0,0521 | +0,0515 | +0,0570 | **+0,0535** |
| F★ − Ds★ (hueco de T20) | +0,0567 | +0,0449 | +0,0502 | +0,0506 |
| F★ − Fspec (lo que gana el factorizado con Spectron) | +0,0664 | +0,0594 | +0,0635 | +0,0631 |
| Ds★ − Dsmuon (lo que gana el denso con Muon) | +0,0619 | +0,0660 | +0,0703 | +0,0661 |
| Fspec − Ds★ (factorizado espectral frente a denso AdamW) | −0,0097 | −0,0145 | −0,0133 | −0,0125 |

## Reglas aplicadas (`t22_decisiones.json`)

1. **Mejor LR.** Fspec: 4,8e-3 (caminata completa de 4: 1,2e-3 → 1,8645; 2,4e-3 → 1,8262; 4,8e-3
   → 1,8149; 9,6e-3 → 1,8353). Dsmuon: 1,2e-3 (6e-4 → 1,7780; 1,2e-3 → 1,7496; 2,4e-3 → 1,7628).
   Ningún óptimo en el borde. A 6000 pasos, Fspec − F★ = −0,102 y Dsmuon − Ds★ = −0,103; Fspec
   pasa la puerta de E1 (≤ F★ − 0,03) y queda −0,038 por debajo de Ds★.
2. **Denso de referencia por semilla:** Dsmuon en las tres (menor que Ds★ en 3/3).
3. **VICTORIA: NO.** Fspec pierde con el denso de referencia en 3/3, por 0,052–0,057.
4. **RECORTA: SÍ.** Fspec < F★ en 3/3 y media(F★ − Fspec) = 0,063 ≥ 0,02. **Etiqueta: RECORTA.**
5. **MUON AYUDA AL DENSO: SÍ** (Dsmuon < Ds★ en 3/3, media 0,066). No perjudica.
6. **CIERRA LA MITAD DEL HUECO: NO.** hueco_spec = +0,0535 frente a hueco★ = +0,0506: fracción del
   hueco cerrada **−6 %** (el hueco no baja; sube 0,003, dentro del σ entre semillas).

**Descriptivos.** Ninguna corrida degenerada, sin baches (saltos > 0,3), sin sobreajuste (brecha
val − train entre −0,01 y +0,03; mínimo de validación en el último paso en las 13). Dispersión entre
semillas a 18 000: 0,004 en los dos brazos. Estado del optimizador: 53 MiB (Dsmuon) y 56 MiB
(Fspec) frente a 103 MiB de AdamW en Ds★ y F★ (sin segundo momento en las matrices). VRAM igual
que Ds y F (0,86 y 1,25 GiB). Pasos recortados por la norma global: Fspec 14 % (F★ 99,7 %),
Dsmuon 2 % (Ds★ 6,7 %).

## Lo que dicen las curvas

HECHO. **El optimizador espectral da casi lo mismo a los dos brazos**: a 18 000 pasos, 0,063 al
factorizado y 0,066 al denso; a 6000, 0,102 y 0,103. La ganancia de [R1] (Spectron) sobre AdamW
existe a esta escala y en carácter, pero su ablación ya lo decía [D05]: la mayor parte es la
ortogonalización, que vale igual para el denso. **El hueco a iso-parámetro queda donde estaba**:
+0,051 con AdamW ★ (T20), +0,054 con Muon/Spectron.

HECHO. Medias por paso de las finales (tres semillas):

| Brazo | 6000 | 12 000 | 18 000 |
|---|---|---|---|
| Dsmuon | 1,8283 | 1,6798 | 1,6234 |
| Fspec | 1,9099 | 1,7491 | 1,6770 |
| Ds★ | 1,9000 | 1,7422 | 1,6895 |
| F★ | 1,9851 | 1,8062 | 1,7401 |

Fspec alcanza el BPC final de F★ en el paso 12 500–13 000 (el 70 % del presupuesto) y el de Ds★ en
el 15 500; Dsmuon alcanza el final de Ds★ en el 11 500–12 000 y el final de Fspec en el 12 500. Es
decir, con el optimizador espectral el factorizado llega adonde llegaba el denso AdamW, y el denso
espectral llega adonde llega el factorizado espectral con un 30 % menos de pasos.

HECHO (lecturas en CPU, `diag_t22.json`, anexo). Spectron cambia la geometría de los factores:
el **rango estable** de la matriz efectiva W en las finales de Fspec es 36 / 26 / 41 / 39 (qkv /
proj / fc1 / fc2, de 128) frente a 28 / 24 / 9 / 16 en F★; en fc1 pasa de 9 a 41. Y baja con el
LR: 57 / 52 / 74 / 59 a 1,2e-3, 40 / 27 / 42 / 37 a 4,8e-3. En el denso, Muon hace lo mismo a lo
grande: rango estable de W 106–119 / 72–73 / 114–119 / 112–115 frente a 30–35 / 36 / 16 / 33 en
Ds★. La **parte visible del gradiente** sube (qkv 0,50, proj 0,73, fc1 0,52, fc2 0,54 frente a
0,41 / 0,57 / 0,39 / 0,42 en F★) y la **alineación** escritura→lectura, que en T20 y T21 estaba en
el azar (0,20 frente a 0,167), sube a 0,28. Ortonormalidad: U algo más ortogonal que en F★ (fuera
de diagonal 0,035 frente a 0,048), V menos (0,043 frente a 0,020); normas de las columnas de U
1,33–1,55 (F★ 1,28–1,60). Sin sumidero (rms del flujo final 10–11, contextual 86–87 %), sin
bloques muertos, sin saliencia baja, en los dos brazos.

INFERENCIA. T21 dejó la lectura de que los factores usaban 10–40 direcciones hicieran lo que
hicieran con el rango disponible, y que el espectro, no el rango, era lo que AdamW no movía. E9 lo
mide: con Spectron el espectro se mueve (rango estable ×4 en fc1), el gradiente visible y la
alineación suben, y el factorizado mejora 0,063. Pero el denso, que no tenía ningún cuello de
rango, mejora lo mismo con la misma ortogonalización, y su espectro se aplana todavía más. Lo que
el optimizador espectral corrige no es un defecto de la factorización sino un defecto de AdamW
sobre matrices, común a los dos. El hueco que queda (+0,054) es el que la clase de funciones de
rango 128 a d=768 tiene frente a un denso estrecho a iso-parámetro, bajo cualquiera de los dos
optimizadores. Es una lectura; lo que la mediría es un barrido de rango con d fijo bajo Spectron.

## Predicciones frente a resultado

| Predicción (antes de correr) | P | Resultado |
|---|---|---|
| mejor LR de Fspec 4,8e-3 | 0,20 | ✓ (2,4e-3 tenía 0,35) |
| mejor LR de Dsmuon 1,2e-3 | 0,35 | ✓ |
| alguna caminata llega a 4 LRs | 0,50 | ✓ (Fspec) |
| Fspec pasa la puerta de E1 a 6000 | 0,40 | ✓ (−0,102) |
| RECORTA o mejor | 0,45 | ✓ |
| CIERRA LA MITAD DEL HUECO | 0,25 | no (−6 %) |
| VICTORIA | 0,10 | no |
| MUON AYUDA AL DENSO | 0,50 | ✓ (0,066) |
| 0 NaN y 0 degeneradas | 0,60 | ✓ |
| ortog_U y ortog_V de Fspec por debajo de los de F★ | 0,70 | no (U sí, V no) |
| rango estable de W mayor que en F★ en los cuatro tipos | 0,55 | ✓ |
| reloj: A 1,5–2,5 h; B 3,5–5 h | 0,60 | ✓ (1 h 51; 4 h 55) |
| RFC 2.2: «el hueco es de optimización» (Spectron cierra lo que AdamW no) | — | refutado a esta escala: mejora a los dos por igual |

## Amenazas a la validez

1. Una escala, carácter, fp32, seq 128; una semilla en la calibración; tres en la final.
2. El grupo AdamW (embedding, LayerNorm, S) quedó al LR calibrado en T20 y no se recalibró junto
   al espectral: una interacción con ese LR no se mide.
3. Spectron está adaptado a tres factores y expresado en la escala RMS de Moonshot (S en AdamW,
   Nesterov, weight decay sin el factor 1/(σ_A + σ_B + 1)): no es la receta publicada al pie de la
   letra. **Cuantificado tras el cierre (DESVIACIONES-T22 §3):** como el decaimiento actúa por
   factor, W decae en Fspec 8,2× por paso lo que en Dsmuon y 11× lo que en F★; acumulado en 18 000
   pasos, W se encoge por 0,41 en Fspec frente a 0,90–0,95 en los otros tres brazos. De F★ a
   Fspec cambian a la vez el optimizador y el decaimiento efectivo, y Fspec − Dsmuon compara brazos
   con regularizaciones muy distintas: los 0,063 y el +0,054 son de la receta tal como se corrió;
   su atribución entre optimizador y decaimiento queda para la ablación E9b. Las normas de U y V
   se leen (anexo) y no muestran colapso.
4. La caminata de LR terminó en 4 y 3 LRs sin borde; el óptimo real de Fspec puede estar entre
   2,4e-3 y 9,6e-3 (la curva es plana: 1,826 / 1,815 / 1,835).
5. Las corridas reutilizadas de T20 son del mismo harness y las mismas semillas, en otra sesión.
6. Los minutos llevan el freno térmico y el coste de Newton-Schulz es de lanzamientos de núcleos
   pequeños, no de FLOPs: Fspec tarda 2× y Dsmuon 1,6× que F★ y Ds★ por paso. Sin claims de
   velocidad.
7. El resultado «Fspec < Ds★ en 3/3» compara optimizadores distintos: es descriptivo, no una
   victoria. La comparación justa es contra Dsmuon.

## Qué desbloquea

- **Para la tesis de MZTrain:** E9 era la última vía del RFC para cerrar el hueco. Las cuatro
  explicaciones contrastadas no lo cierran: el montaje (E1: bajo la línea base sana el hueco es
  +0,051 y no desaparece; el +0,086 anterior era de otro corpus y otra programación, no
  comparable pieza a pieza), la forma de la hipérbola (E5: −0,024 como mucho), el denso con la forma de F
  (E5: pierde con el estrecho) y el optimizador (E9: +0,054). **A iso-parámetro y a esta escala,
  el denso estrecho es mejor que el factorizado en BPC bajo los dos optimizadores**, y la métrica
  legítima pasa a ser la de la sección 5.6 del RFC: calidad por GB de VRAM y por GB-hora en el
  techo de la máquina, con el denso con el estado del optimizador comprimido como rival.
- **Lo que E9 sí deja:** (a) Muon sobre la línea base sana mejora al denso 0,066 BPC a 18 000
  pasos con la mitad de estado de optimizador; (b) Spectron adaptado mejora al factorizado 0,063 y
  lo deja por debajo del denso AdamW en 3/3, con el espectro de W cuatro veces más repartido en
  fc1; (c) ninguna de las dos recetas fue inestable: 0 NaN en 13 corridas con LR hasta 9,6e-3.
  Cualquier integración iría como opción desactivada por defecto y después de E10.
- **Siguientes, por coste:** E9b (barato, 2–3 corridas de 6000): ablación del decaimiento de U, V
  (wd = 0 o wd con el mismo factor que el paso) y Muon puro sobre los factores sin la
  renormalización de Spectron, para saber qué parte de los 0,063 es de cada pieza. Barrido de
  rango con d fijo bajo Spectron (la lectura del rango estable lo pide). E10 (3× pasos) con la
  pareja espectral, antes de anunciar nada. E6/E7 siguen abiertos como preguntas de línea base.

## Anexo: tablas generadas por `tablas_t22.py` desde los JSON de T22

### Smoke (no es dato; 500 pasos, coseno sobre 500, semilla 17; Ds y F del smoke de T20 al lado)

| Corrida | BPC paso 500 | parámetros | min/1000 | VRAM pico (GiB) | estado del optimizador (MiB) | reanudación (diff) |
| --- | --- | --- | --- | --- | --- | --- |
| S.Dsmuon.lr0.0012.b0.95.s17 | 2,6817 | 13 472 112 | 1.963 | 0.86 | 53.2 | +0,000000 |
| S.Fspec.lr0.0012.b0.95.s17 | 3,1293 | 13 570 816 | 3.276 | 1.25 | 55.8 | +0,000000 |
| S.Ds.lr0.0006.b0.95.s17 (T20) | 3,1612 | 13 472 112 | 1.230 | 0.91 | 102.8 | +0,000000 |
| S.F.lr0.0006.b0.95.s17 (T20) | 3,2032 | 13 570 816 | 1.556 | 1.30 | 103.5 | +0,000151 |

### Calibración (etapa A; 6000 pasos, semilla 17, coseno sobre 6000); Ds★ y F★ de T20 al lado

| Brazo | LR espectral | LR AdamW | BPC paso 6000 | origen | gn media / máx | pasos recortados | min/1000 | VRAM pico (GiB) | estado opt. (MiB) | brecha val − train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fspec | 0.0012 | 3e-4 | 1,8645 | T22 | 1.92 / 18.6 | 100 % | 2.896 | 1.25 | 55.8 | +0,0176 |
| Fspec | 0.0024 | 3e-4 | 1,8262 | T22 | 1.34 / 18.5 | 100 % | 2.992 | 1.25 | 55.8 | +0,0099 |
| Fspec | 0.0048 | 3e-4 | **1,8149** | T22 | 1.09 / 18.4 | 31 % | 2.985 | 1.25 | 55.8 | +0,0095 |
| Fspec | 0.0096 | 3e-4 | 1,8353 | T22 | 0.97 / 18.4 | 14 % | 3.450 | 1.25 | 55.8 | +0,0141 |
| Dsmuon | 0.0006 | 6e-4 | 1,7780 | T22 | 1.03 / 10.7 | 26 % | 2.020 | 0.86 | 53.2 | +0,0016 |
| Dsmuon | 0.0012 | 6e-4 | **1,7496** | T22 | 0.70 / 10.6 | 7 % | 1.900 | 0.86 | 53.2 | +0,0022 |
| Dsmuon | 0.0024 | 6e-4 | 1,7628 | T22 | 0.57 / 10.6 | 3 % | 2.041 | 0.86 | 53.2 | +0,0043 |
| F | None | 3e-4 (todo) | 1,9172 | T20 | 1.21 / 18.4 | 100 % | 1.590 | 1.30 | 103.5 | +0,0275 |
| Ds | None | 6e-4 (todo) | 1,8526 | T20 | 0.72 / 10.6 | 7 % | 1.299 | 0.91 | 102.8 | +0,0157 |

A 6000 pasos: Fspec − F★ = -0,1023; Fspec − Ds★ = -0,0376; Dsmuon − Ds★ = -0,1030; Fspec pasa la puerta de E1 (≤ F★ − 0,03): sí.

### Finales (etapa B; 18 000 pasos, semillas 17/29/43); Ds★ y F★ de T20

| Brazo | LR | s17 | s29 | s43 | media |
| --- | --- | --- | --- | --- | --- |
| Fspec | 0.0048 | 1,6749 | 1,6774 | 1,6786 | 1,6770 |
| Dsmuon | 0.0012 | 1,6228 | 1,6259 | 1,6216 | 1,6234 |
| Ds_T20 | 0.0006 | 1,6847 | 1,6919 | 1,6919 | 1,6895 |
| F_T20 | 0.0003 | 1,7413 | 1,7368 | 1,7421 | 1,7401 |

Diferencias pareadas por semilla:

| | s17 | s29 | s43 | media |
| --- | --- | --- | --- | --- |
| Fspec − denso de referencia | +0,0521 | +0,0515 | +0,0570 | +0,0535 |
| Fspec − Ds★ | -0,0097 | -0,0145 | -0,0133 | -0,0125 |
| Fspec − Dsmuon | +0,0521 | +0,0515 | +0,0570 | +0,0535 |
| F★ − Fspec | +0,0664 | +0,0594 | +0,0635 | +0,0631 |
| Ds★ − Dsmuon | +0,0619 | +0,0660 | +0,0703 | +0,0660 |
| F★ − Ds★ (hueco de T20) | +0,0567 | +0,0449 | +0,0502 | +0,0506 |

Denso de referencia por semilla: {'17': 'Dsmuon', '29': 'Dsmuon', '43': 'Dsmuon'}. Etiqueta: **RECORTA**. MUON AYUDA AL DENSO: True; PERJUDICA: False; CIERRA LA MITAD DEL HUECO: False; fracción del hueco cerrada: -6 %.

Media de las tres semillas por paso:

| Brazo | 6000 | 12 000 | 18 000 |
| --- | --- | --- | --- |
| Ds_T20 | 1,9000 | 1,7422 | 1,6895 |
| F_T20 | 1,9851 | 1,8062 | 1,7401 |
| Fspec | 1,9099 | 1,7491 | 1,6770 |
| Dsmuon | 1,8283 | 1,6798 | 1,6234 |

### Estado de las corridas

Corridas nuevas: 13; reutilizadas de T20: 8; reintentos: ninguno; lanzamientos: 2; degeneradas: ninguna; sobreajuste marcado: ninguno; baches: 0; checkpoints: 13/13.

### Lecturas en CPU (64 ventanas de validación; `diag_t22.json`)

| Checkpoint | rms flujo final | fracción constante | parte contextual | dim. efectiva | bloques muertos | rango estable qkv / proj / fc1 / fc2 | visible del gradiente (qkv/proj/fc1/fc2) | alineación (azar) | saliencia baja (fc1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C.F.lr0.0003.b0.95.s43 (T20) | 10.08 | 13.3 % | 84.9 % | 13.3 | — | 29 / 26 / 9 / 17 (de 128.0) | 0.42 / 0.56 / 0.39 / 0.42 | 0.206 (0.167) | 0 % |
| C.F.lr0.0003.b0.95.s29 (T20) | 9.04 | 13.3 % | 84.9 % | 14.0 | — | 27 / 23 / 9 / 15 (de 128.0) | 0.41 / 0.58 / 0.39 / 0.43 | 0.203 (0.167) | 0 % |
| C.F.lr0.0003.b0.95.s17 (T20) | 9.68 | 13.0 % | 84.7 % | 13.2 | — | 27 / 24 / 9 / 15 (de 128.0) | 0.41 / 0.57 / 0.39 / 0.42 | 0.205 (0.167) | 0 % |
| C.Ds.lr0.0006.b0.95.s43 (T20) | 7.74 | 10.4 % | 86.1 % | 16.8 | — | denso | — | — | — |
| C.Ds.lr0.0006.b0.95.s29 (T20) | 7.72 | 9.2 % | 86.8 % | 16.9 | — | denso | — | — | — |
| C.Ds.lr0.0006.b0.95.s17 (T20) | 7.59 | 9.4 % | 87.2 % | 17.1 | — | denso | — | — | — |
| A.Ds.lr0.0006.b0.95.s17 (T20) | 4.44 | 9.5 % | 86.0 % | 17.3 | — | denso | — | — | — |
| A.F.lr0.0003.b0.95.s17 (T20) | 4.78 | 11.7 % | 86.2 % | 15.6 | — | 29 / 34 / 9 / 18 (de 128.0) | 0.37 / 0.48 / 0.40 / 0.38 | 0.184 (0.167) | 0 % |
| A.Fspec.lr0.0012.b0.95.s17 | 2.37 | 13.2 % | 85.2 % | 18.2 | — | 57 / 52 / 74 / 59 (de 128.0) | 0.40 / 0.37 / 0.36 / 0.37 | 0.184 (0.167) | 0 % |
| A.Dsmuon.lr0.0012.b0.95.s17 | 5.59 | 12.8 % | 84.4 % | 17.3 | — | denso | — | — | — |
| A.Fspec.lr0.0024.b0.95.s17 | 4.59 | 14.0 % | 84.4 % | 16.1 | — | 50 / 36 / 51 / 47 (de 128.0) | 0.45 / 0.56 / 0.44 / 0.43 | 0.205 (0.167) | 0 % |
| A.Dsmuon.lr0.0024.b0.95.s17 | 19.79 | 13.5 % | 83.8 % | 17.2 | — | denso | — | — | — |
| A.Fspec.lr0.0048.b0.95.s17 | 9.83 | 14.1 % | 83.9 % | 15.0 | — | 40 / 27 / 42 / 37 (de 128.0) | 0.47 / 0.67 / 0.50 / 0.49 | 0.240 (0.167) | 0 % |
| A.Fspec.lr0.0096.b0.95.s17 | 23.85 | 14.6 % | 83.3 % | 15.1 | — | 27 / 23 / 40 / 29 (de 128.0) | 0.53 / 0.71 / 0.56 / 0.59 | 0.292 (0.167) | 0 % |
| A.Dsmuon.lr0.0006.b0.95.s17 | 1.89 | 12.5 % | 84.2 % | 16.4 | — | denso | — | — | — |
| Bf.Fspec.lr0.0048.b0.95.s17 | 10.51 | 12.4 % | 86.3 % | 15.2 | — | 36 / 26 / 41 / 39 (de 128.0) | 0.50 / 0.72 / 0.52 / 0.54 | 0.283 (0.167) | 0 % |
| Bf.Dsmuon.lr0.0012.b0.95.s17 | 11.47 | 11.8 % | 86.5 % | 16.5 | — | denso | — | — | — |
| Bf.Dsmuon.lr0.0012.b0.95.s29 | 11.37 | 12.1 % | 85.4 % | 16.6 | — | denso | — | — | — |
| Bf.Fspec.lr0.0048.b0.95.s29 | 10.33 | 11.6 % | 87.2 % | 15.7 | — | 36 / 26 / 40 / 40 (de 128.0) | 0.50 / 0.73 / 0.51 / 0.54 | 0.278 (0.167) | 0 % |
| Bf.Fspec.lr0.0048.b0.95.s43 | 10.59 | 11.7 % | 86.9 % | 15.1 | — | 38 / 27 / 40 / 39 (de 128.0) | 0.49 / 0.73 / 0.50 / 0.53 | 0.286 (0.167) | 0 % |
| Bf.Dsmuon.lr0.0012.b0.95.s43 | 11.27 | 10.1 % | 87.5 % | 16.6 | — | denso | — | — | — |

Normas espectrales y ortogonalidad (media de los 8 bloques; `normas` de `diag_t22.json`): factorizados σ_A = ‖U·diag(S)‖₂, σ_B = ‖diag(S)·V‖₂; densos ‖W‖₂ y rango estable de W

| Checkpoint | tipo | σ_A o ‖W‖₂ (qkv/proj/fc1/fc2) | σ_B o rango estable W (qkv/proj/fc1/fc2) | ‖U‖₂ / ‖V‖₂ (qkv) | ortog_U / ortog_V (qkv, fuera de diagonal) | normas cols U (min/máx, qkv) |
| --- | --- | --- | --- | --- | --- | --- |
| A.Fspec.lr0.0012.b0.95.s17 | factorizado | 3.18 / 1.74 / 3.69 / 1.77 | 3.00 / 1.74 / 3.36 / 2.18 | 1.28 / 1.20 | 0.014 / 0.013 | 1.00 / 1.06 |
| A.Dsmuon.lr0.0012.b0.95.s17 | denso | 3.52 / 2.10 / 4.51 / 3.96 | 105.9 / 75.0 / 105.4 / 122.9 | — | — | — |
| A.Fspec.lr0.0024.b0.95.s17 | factorizado | 3.63 / 2.01 / 4.43 / 2.02 | 2.98 / 2.04 / 3.76 / 2.92 | 1.48 / 1.21 | 0.022 / 0.021 | 1.07 / 1.18 |
| A.Dsmuon.lr0.0024.b0.95.s17 | denso | 5.75 / 3.81 / 7.46 / 7.69 | 112.8 / 72.6 / 120.6 / 118.1 | — | — | — |
| A.Fspec.lr0.0048.b0.95.s17 | factorizado | 4.35 / 2.45 / 5.62 / 2.54 | 3.11 / 2.45 / 4.28 / 4.12 | 1.81 / 1.29 | 0.030 / 0.032 | 1.21 / 1.38 |
| A.Fspec.lr0.0096.b0.95.s17 | factorizado | 5.58 / 2.99 / 7.45 / 3.43 | 3.58 / 2.72 / 4.94 / 5.78 | 2.34 / 1.50 | 0.041 / 0.051 | 1.40 / 1.68 |
| A.Dsmuon.lr0.0006.b0.95.s17 | denso | 2.46 / 1.33 / 2.75 / 2.12 | 99.4 / 82.8 / 125.5 / 132.9 | — | — | — |
| Bf.Fspec.lr0.0048.b0.95.s17 | factorizado | 4.58 / 2.56 / 6.43 / 3.02 | 3.05 / 2.46 / 4.49 / 4.84 | 1.99 / 1.33 | 0.034 / 0.043 | 1.33 / 1.55 |
| Bf.Dsmuon.lr0.0012.b0.95.s17 | denso | 5.06 / 3.27 / 6.66 / 6.84 | 111.8 / 73.0 / 117.4 / 115.1 | — | — | — |
| Bf.Dsmuon.lr0.0012.b0.95.s29 | denso | 5.21 / 3.27 / 6.77 / 6.89 | 105.9 / 71.6 / 113.9 / 112.0 | — | — | — |
| Bf.Fspec.lr0.0048.b0.95.s29 | factorizado | 4.66 / 2.54 / 6.40 / 2.99 | 3.07 / 2.55 / 4.57 / 4.79 | 2.01 / 1.33 | 0.036 / 0.043 | 1.31 / 1.54 |
| Bf.Fspec.lr0.0048.b0.95.s43 | factorizado | 4.60 / 2.57 / 6.45 / 3.00 | 3.07 / 2.49 / 4.55 / 4.85 | 2.00 / 1.33 | 0.034 / 0.043 | 1.34 / 1.57 |
| Bf.Dsmuon.lr0.0012.b0.95.s43 | denso | 5.12 / 3.26 / 6.59 / 6.84 | 109.2 / 71.9 / 118.8 / 114.3 | — | — | — |
| A.Ds.lr0.0006.b0.95.s17 (T20) | denso | 4.50 / 2.10 / 8.73 / 4.75 | 30.0 / 41.2 / 14.0 / 29.3 | — | — | — |
| A.F.lr0.0003.b0.95.s17 (T20) | factorizado | 4.98 / 2.25 / 9.76 / 2.51 | 3.14 / 2.00 / 5.01 / 3.86 | 2.06 / 1.29 | 0.036 / 0.014 | 1.11 / 1.23 |
| C.Ds.lr0.0006.b0.95.s17 (T20) | denso | 5.42 / 3.00 / 10.82 / 7.67 | 35.3 / 36.5 / 16.0 / 34.1 | — | — | — |
| C.Ds.lr0.0006.b0.95.s29 (T20) | denso | 5.83 / 2.96 / 10.82 / 7.80 | 30.6 / 36.7 / 15.9 / 32.2 | — | — | — |
| C.Ds.lr0.0006.b0.95.s43 (T20) | denso | 5.88 / 2.99 / 10.87 / 7.72 | 30.9 / 36.5 / 15.9 / 33.6 | — | — | — |
| C.F.lr0.0003.b0.95.s17 (T20) | factorizado | 6.14 / 2.71 / 12.30 / 3.57 | 3.22 / 2.25 / 5.82 / 6.40 | 2.64 / 1.38 | 0.048 / 0.020 | 1.28 / 1.60 |
| C.F.lr0.0003.b0.95.s29 (T20) | factorizado | 6.11 / 2.78 / 12.24 / 3.84 | 3.33 / 2.27 / 5.72 / 6.49 | 2.62 / 1.43 | 0.048 / 0.021 | 1.28 / 1.55 |
| C.F.lr0.0003.b0.95.s43 (T20) | factorizado | 6.36 / 2.67 / 12.32 / 3.44 | 3.17 / 2.24 / 5.75 / 6.19 | 2.74 / 1.37 | 0.049 / 0.020 | 1.29 / 1.57 |

