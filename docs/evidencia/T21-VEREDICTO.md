# T21 — Veredicto: la forma importa, pero poco; subir r/d por la hipérbola mejora al factorizado de forma monótona y aun así ninguna forma pasa la puerta, y el denso con la forma de F pierde con el denso estrecho

**Fecha:** 2026-10-07. **Experimento:** E5 de `RFC-CALIDAD-1.md` (sección 5.4; mecanismo en 3.2).

**Preregistro:** `PREREGISTRO-T21.md`, sellado antes del smoke y de cualquier corrida
(`PREREGISTRO-T21.sha256`, bloque v1 de 2026-10-07 04:29, 59 ficheros: harness de T21, harness
sellado de T20 que importa, `t20_results.json` y el sello de T20, tests, dependencias, corpus y
`mztrain` entero). El sello siguió vigente al terminar.

**Desviaciones:** `DESVIACIONES-T21.md`: **ninguna.** Smoke y cadena en un solo lanzamiento cada
uno, 0 cortes, 0 reintentos, 0 NaN, 0 degeneradas, 15/15 checkpoints, stderr vacío, 0 disparos
térmicos (máximo 81 °C).

**Datos:** `t21_smoke.json`, `t21_results.json` (15 corridas nuevas y, dentro, las 10 de T20 que se
reutilizan con su clave y su sello), `t21_decisiones.json` (decisiones de las reglas, sin juicio
humano), `t21_analysis.json`, `diag_t21.json`, `freno_t21.csv`, `vigia_t21.log`, `log_T21_*.txt`,
`ckpt/` (15 checkpoints), `ckpt_smoke/`. Tablas del anexo: `tablas_t21.py` → `t21_tablas.md`.

**Protocolo.** Línea base sana ★, corpus, lote, semilla y harness de T20 (importado, no copiado).
Barrido a 6000 pasos con semilla 17 y coseno sobre 6000: F y las cuatro formas de la hipérbola a
iso-parámetro en {1,5e-4, 3e-4, 6e-4}; el denso de control Dc (d=768, atención 192, FFN 640,
mismos productos por token que F) en {3e-4, 6e-4} y 1,2e-3 por la regla. **Todo es «a presupuesto
fijo de 6000 pasos» y con una semilla.** La final con tres semillas no se corrió porque ninguna
forma pasó la puerta.

## Resultado

| Brazo | (d, r) | r/d | ahorro frente al denso de igual d | mejor LR | BPC paso 6000 | frente a F★ |
|---|---|---|---|---|---|---|
| F★ (T20) | (768, 128) | 0,167 | 4,5× | 3e-4 | 1,9172 | — |
| F640 | (640, 155) | 0,242 | 3,1× | 3e-4 | 1,8986 | −0,019 |
| F576 (T20) | (576, 173) | 0,300 | 2,5× | 3e-4 | 1,8975 | −0,020 |
| F512 | (512, 196) | 0,383 | 2,0× | 3e-4 | 1,8938 | −0,023 |
| F448 | (448, 226) | 0,504 | 1,5× | 3e-4 | **1,8932** | **−0,024** |
| Dc (denso con la forma de F) | 768 / 192 / 640 | — | — | 6e-4 | 1,8674 | −0,050 |
| Ds★ (T20) | denso 368 | — | — | 6e-4 | **1,8526** | −0,065 |

## Reglas aplicadas (`t21_decisiones.json`)

1. **Mejor LR.** 3e-4 en F y en las cuatro formas, dentro de la rejilla (ningún borde). F a
   1,5e-4 da 1,9784, peor que a 3e-4: **el óptimo de F estaba dentro de la rejilla de T20** y la
   regla 3 no se dispara; el hueco F★ − Ds★ de T20 queda con F bien calibrado. Dc prefiere 6e-4
   (1,8674) a 3e-4 (1,8813) y degrada a 1,2e-3 (2,0636).
2. **LA FORMA IMPORTA: SÍ.** Las cuatro formas mejoran a F★ en más de 0,01 (de 0,019 a 0,024) y
   la secuencia por r/d creciente es monótona: 1,8986 ≥ 1,8975 ≥ 1,8938 ≥ 1,8932.
3. **Puerta a la final: NO PASA.** La mejor forma, F448, queda a −0,024 de F★; la puerta pedía
   0,030 (1,8872). Etiqueta: **SIN_FINALISTA**; no hay final de tres semillas.
4. **DC ES LA REFERENCIA: NO.** Dc (1,8674) pierde con Ds★ (1,8526) por 0,015, más que el
   margen de 0,01. Ds sigue siendo el denso de referencia.
5. INCONCLUSO, RECORTA, MUERE, VICTORIA: no aplican (sin final).

**Descriptivos.** Ninguna corrida degenerada, sin baches (saltos > 0,3), sin sobreajuste (brecha
val − train a 6000 entre +0,016 y +0,044; mínimo de validación en el último paso en las 15).
Hueco F − Ds★ a 6000 según el LR de F: +0,065 a 3e-4, +0,126 a 1,5e-4, +0,211 a 6e-4.

## Lo que dice la curva de forma

HECHO. La mejora al subir r/d existe y es monótona, pero **se satura pronto**: de r/d 0,17 a
0,24 se gana 0,019; de 0,24 a 0,50 solo 0,005 más. El salto grande está entre F y la primera
forma, y las cuatro formas están dentro de 0,0054 unas de otras, del orden del σ entre semillas de
F★ a 6000 pasos en T20 (0,006): el **orden** entre formas no está resuelto con una semilla; la
**mejora** frente a F★ sí (3 a 4 veces ese σ).

HECHO. El precio sí crece: el ahorro frente al denso de igual d baja de 4,5× a 1,5× y los productos
por token suben un 3 %. Con r/d = 0,50 el modelo gana 0,024 y conserva la mitad de la tesis de
memoria.

HECHO. El denso con la forma de F, que gasta exactamente los productos por token de F, gana a
todas las formas (0,026 a 0,050) y pierde con el denso estrecho (0,015). Es decir, a iso-parámetro
y a iso-FLOP de lineales, **el orden es Ds < Dc < F448 ≤ F512 ≤ F576 ≤ F640 < F**: el denso
estrecho sigue siendo el techo, el denso ancho-pero-fino va detrás, y los factorizados detrás de
los dos.

HECHO (lecturas en CPU, `diag_t21.json`). En las formas, la parte visible del gradiente crece con
r/d como decía el RFC: en `qkv` pasa de 0,37 (F★) a 0,45, 0,57 y 0,65; en `proj`, de 0,48 a 0,81.
Pero el **rango estable** de las matrices efectivas no lo sigue: en `fc1` se queda en 9–10 con
128, 155, 196 o 226 direcciones disponibles; en `qkv` incluso baja (29, 25, 23, 21). La
**alineación** entre lo que escribe una capa y lo que leen las posteriores está en el azar en las
cuatro formas (0,254 frente a 0,242; 0,390 frente a 0,383; 0,508 frente a 0,504). Ninguna forma
tiene componentes de saliencia baja, ningún bloque muerto y ningún sumidero (rms del flujo final
2,7–4,1; parte contextual 86–87 %).

INFERENCIA. Si el cuello fuera el rango disponible, el rango estable debería crecer con r y la
mejora debería seguir a la fracción visible del gradiente; ocurre lo segundo a medias y lo primero
no. Los factorizados usan unas 10–40 direcciones por matriz hagan lo que hagan con r, y lo que
ganan al subir r/d se parece más a «estrecharse hacia un denso» que a «quitar un cuello». Eso
encaja con que Dc, denso y con los mismos FLOPs, gane a todas las formas, y con que Ds, más
estrecho todavía, gane a Dc. Es una lectura, no una medida: la mide E9 (un optimizador que mueva
el espectro) o un barrido de rango con d fijo.

## Predicciones frente a resultado

| Predicción (antes de correr) | P | Resultado |
|---|---|---|
| LA FORMA IMPORTA | 0,45 | ✓ |
| F448 es la mejor forma | 0,45 | ✓ (por 0,0006 sobre F512) |
| la mejor forma pasa la puerta | 0,35 | no (−0,024) |
| F a 1,5e-4 mejora a 3e-4 en ≥ 0,01 | 0,35 | no (empeora 0,061) |
| mejor LR de cada forma 3e-4 | 0,5 por forma | ✓ en las cuatro |
| DC ES LA REFERENCIA | 0,30 | no (pierde por 0,015; «dentro de ±0,01», P 0,45, tampoco) |
| 0 NaN y 0 degeneradas | 0,7 | ✓ |
| etapa A entre 2 y 3 h con el freno | 0,7 | ✓ (2 h 45 min) |
| RFC 3.2: subir r/d alivia el cuello y agranda la parte visible del gradiente | — | la parte visible sí; el rango estable no |

## Amenazas a la validez

1. Una escala, carácter, fp32, seq 128, **una semilla** y 6000 pasos: la mejora de las formas
   frente a F★ es de 3–4 σ, pero el orden entre formas no está resuelto y nada se sabe a 18 000
   pasos ni con tres semillas (la puerta impidió la final).
2. Rejilla de tres LRs; todos los óptimos cayeron en el centro.
3. Las corridas reutilizadas de T20 son del mismo harness y la misma semilla, en otra sesión; las
   tres formas nuevas reanudaron bit a bit en GPU, Dc no (7e-6).
4. Dc tiene la forma de F pero no su número exacto de parámetros (−4096) ni su init espectral; su
   mejor LR cayó en el centro de su rejilla.
5. Los minutos llevan el freno térmico: sin claims de velocidad.
6. La puerta de 0,03 se fijó en E1 para una criba contra la final; aquí deja fuera una mejora
   real pero pequeña. Es una elección preregistrada, no un hallazgo.

## Qué desbloquea

- **Para la tesis de MZTrain:** ninguna forma de la hipérbola alcanza al denso iso-parámetro;
  la que más se acerca (F448) sigue +0,041 por detrás de Ds★ a 6000 pasos y paga 3× de ahorro.
  El claim de calidad por parámetro no cambia. Y el denso con la forma de F, que es el rival que
  cualquier «ancho y fino» tendría que batir, también pierde con el estrecho.
- **La curva calidad–ahorro queda medida** (una semilla): 4,5× → 1,9172; 3,1× → 1,8986; 2,5× →
  1,8975; 2,0× → 1,8938; 1,5× → 1,8932. Casi toda la ganancia está en el primer paso (de 4,5× a
  3,1×). Si el producto quisiera una forma distinta, F640 es el punto con mejor relación.
- **E9 (optimizador espectral)** es lo que queda del RFC para el hueco: la lectura del rango
  estable apunta a que el espectro de los factores, no el rango disponible, es lo que no se mueve
  con AdamW. Spectron o Muon en F y en el denso, calibrados los dos, a tres semillas.
- **E8 sigue sin condición de entrada** (sin componentes de saliencia baja en ninguna forma).
- Opcional y barato: la final de F640 o F448 con tres semillas a 18 000 pasos (1,5 h) diría si
  la mejora de 0,02 sobre F★ se sostiene; no cambiaría ninguna etiqueta frente a Ds★.

## Anexo: tablas generadas por `tablas_t21.py` desde los JSON de T21

### Smoke (no es dato; 500 pasos, coseno sobre 500, semilla 17)

| Corrida | BPC paso 500 | parámetros | min/1000 | VRAM pico (GiB) | reanudación (diff) |
| --- | --- | --- | --- | --- | --- |
| S.F640.lr0.0003.b0.95.s17 | 3,3195 | 13 522 400 | 1.587 | 1.19 | +0,000000 |
| S.F512.lr0.0003.b0.95.s17 | 3,3868 | 13 507 200 | 1.389 | 1.10 | +0,000000 |
| S.F448.lr0.0003.b0.95.s17 | 3,3516 | 13 540 864 | 1.308 | 1.04 | +0,000000 |
| S.Dc.lr0.0006.b0.95.s17 | 3,0770 | 13 566 720 | 1.050 | 0.80 | +0,000007 |

### Barrido (etapa A; 6000 pasos, semilla 17, coseno sobre 6000). «T20» = corrida reutilizada

| Brazo | LR | BPC paso 6000 | origen | gn media / máx | pasos recortados | min/1000 | VRAM pico (GiB) | brecha val − train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F | 0.00015 | 1,9784 | T21 | 1.72 / 18.4 | 100 % | 1.332 | 1.30 | +0,0300 |
| F | 0.0003 | **1,9172** | T20 | 1.21 / 18.4 | 100 % | 1.590 | 1.30 | — |
| F | 0.0006 | 2,0633 | T20 | 1.05 / 18.4 | 47 % | 1.515 | 1.30 | — |
| F640 | 0.00015 | 1,9787 | T21 | 1.68 / 16.3 | 100 % | 1.488 | 1.19 | +0,0355 |
| F640 | 0.0003 | **1,8986** | T21 | 1.12 / 16.3 | 88 % | 1.546 | 1.19 | +0,0211 |
| F640 | 0.0006 | 1,9757 | T21 | 0.96 / 16.3 | 25 % | 1.750 | 1.19 | +0,0307 |
| F576 | 0.00015 | 1,9827 | T21 | 1.72 / 16.6 | 100 % | 2.279 | 1.14 | +0,0362 |
| F576 | 0.0003 | **1,8975** | T20 | 1.12 / 16.6 | 88 % | 1.546 | 1.14 | — |
| F576 | 0.0006 | 1,9547 | T21 | 0.94 / 16.6 | 17 % | 1.548 | 1.14 | +0,0353 |
| F512 | 0.00015 | 1,9859 | T21 | 1.65 / 13.7 | 100 % | 1.870 | 1.10 | +0,0372 |
| F512 | 0.0003 | **1,8938** | T21 | 1.13 / 13.7 | 88 % | 2.277 | 1.10 | +0,0306 |
| F512 | 0.0006 | 1,9245 | T21 | 0.86 / 13.7 | 9 % | 1.782 | 1.10 | +0,0194 |
| F448 | 0.00015 | 1,9995 | T21 | 1.72 / 11.7 | 100 % | 1.472 | 1.04 | +0,0422 |
| F448 | 0.0003 | **1,8932** | T21 | 1.05 / 11.7 | 45 % | 1.413 | 1.04 | +0,0246 |
| F448 | 0.0006 | 1,9102 | T21 | 0.84 / 11.7 | 9 % | 1.413 | 1.04 | +0,0227 |
| Dc | 0.0003 | 1,8813 | T21 | 1.00 / 17.7 | 26 % | 1.932 | 0.80 | +0,0188 |
| Dc | 0.0006 | **1,8674** | T21 | 0.76 / 17.7 | 7 % | 1.881 | 0.80 | +0,0162 |
| Dc | 0.0012 | 2,0636 | T21 | 0.65 / 17.7 | 8 % | 1.368 | 0.80 | +0,0443 |

### Curva de forma (cada brazo en su mejor LR) y curva calidad–ahorro

| Brazo | (d, r) | r/d | ahorro frente al denso de igual d | LR | BPC paso 6000 | frente a F★ |
| --- | --- | --- | --- | --- | --- | --- |
| F | (768, 128) | 0.167 | 4.5× | 0.0003 | 1,9172 | — |
| F640 | (640, 155) | 0.242 | 3.1× | 0.0003 | 1,8986 | -0,0186 |
| F576 | (576, 173) | 0.300 | 2.5× | 0.0003 | 1,8975 | -0,0197 |
| F512 | (512, 196) | 0.383 | 2.0× | 0.0003 | 1,8938 | -0,0235 |
| F448 | (448, 226) | 0.504 | 1.5× | 0.0003 | 1,8932 | -0,0241 |
| Dc | denso 768 / att 192 / FFN 640 | — | — | 0.0006 | 1,8674 | -0,0499 |
| Ds★ (T20) | denso 368 | — | — | 6e-4 | 1,8526 | -0,0647 |

Diferencia de BPC frente a F★ (3e-4) por paso, cada brazo en su mejor LR:

| Brazo | 500 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F640 | -0,0346 | -0,0577 | -0,0401 | -0,0335 | -0,0216 | -0,0189 | -0,0186 |
| F576 | -0,0021 | -0,0311 | -0,0428 | -0,0359 | -0,0211 | -0,0208 | -0,0197 |
| F512 | +0,0215 | -0,0257 | -0,0526 | -0,0442 | -0,0308 | -0,0259 | -0,0235 |
| F448 | +0,0252 | -0,0549 | -0,0626 | -0,0461 | -0,0311 | -0,0260 | -0,0241 |
| Dc | -0,2235 | -0,2148 | -0,1199 | -0,0863 | -0,0649 | -0,0540 | -0,0499 |

Hueco F − Ds★ a 6000 pasos según el LR de F: 0.00015: +0,1259; 0.0003: +0,0647; 0.0006: +0,2107.

### Estado de las corridas

Corridas nuevas: 15; reutilizadas de T20: 10; reintentos: ninguno; lanzamientos: 2; degeneradas: ninguna; sobreajuste marcado: ninguno; baches: 0; checkpoints: 15/15.

### Lecturas en CPU (64 ventanas de validación; `diag_t21.json`), cada forma en su mejor LR y F

| Checkpoint | rms flujo final | fracción constante | parte contextual | dim. efectiva | bloques muertos | rango estable qkv / proj / fc1 / fc2 | visible del gradiente (qkv/proj/fc1/fc2) | alineación (azar) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A.Ds.lr0.0006.b0.95.s17 (T20) | 4.44 | 9.5 % | 86.0 % | 17.3 | — | denso | — | — |
| A.F.lr0.0003.b0.95.s17 (T20) | 4.78 | 11.7 % | 86.2 % | 15.6 | — | 29 / 34 / 9 / 18 (de 128.0) | 0.37 / 0.48 / 0.40 / 0.38 | 0.184 (0.167) |
| A.F640.lr0.0003.b0.95.s17 | 4.08 | 11.1 % | 86.3 % | 15.1 | — | 25 / 36 / 10 / 25 (de 155.0) | 0.45 / 0.59 / 0.47 / 0.44 | 0.254 (0.242) |
| A.F512.lr0.0003.b0.95.s17 | 3.03 | 10.0 % | 86.5 % | 16.8 | — | 23 / 44 / 9 / 28 (de 196.0) | 0.57 / 0.69 / 0.58 / 0.58 | 0.390 (0.383) |
| A.F448.lr0.0003.b0.95.s17 | 2.74 | 10.0 % | 86.7 % | 17.2 | — | 21 / 43 / 10 / 33 (de 226.0) | 0.65 / 0.81 / 0.67 / 0.66 | 0.508 (0.504) |

