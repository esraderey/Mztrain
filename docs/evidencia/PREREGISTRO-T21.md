# PREREGISTRO — T21: ¿importa la forma (d, r) del factorizado a iso-parámetro? (E5 de RFC-CALIDAD-1)

**Fecha:** 2026-10-07, escrito ANTES de cualquier corrida de T21 (incluido el smoke). El hash queda
en `PREREGISTRO-T21.sha256`, junto al de `analyze_t21.py` (las reglas), `t21.py`, `diag_t21.py`, el
paquete `mztrain_t21/`, los lanzadores, los tests, el harness sellado de T20 que T21 importa
(`t20.py`, `analyze_t20.py`, `diag_t20.py`, `mztrain_t20/`), `t20_results.json` y
`PREREGISTRO-T20.sha256` (las corridas de T20 que se reutilizan y el sello bajo el que se hicieron),
el corpus, las dependencias de T9/T17/T18/T19 y el paquete `mztrain` entero. `t21.py` se niega a
correr si algún fichero del último bloque del sello ha cambiado; la etapa A no arranca sin un smoke
completo bajo el mismo sello; la B no arranca sin la A completa y con `t21_decisiones.json`
coincidente con las reglas recalculadas.

**Origen.** T20 (E1) cerró SIN FINALISTA: con la línea base sana ★ el factorizado F (768, 128) sigue
+0,051 BPC por detrás del denso iso-parámetro Ds (368) en 3/3 a 18 000 pasos, y de las cuatro
candidatas la única por debajo de F★ en la criba fue la forma F576 (576, 173), a −0,020, sin
llegar a la puerta de 0,03. El RFC (sección 5.4) manda E5: la curva completa de la hipérbola
128·r·d + 32·r + 1280·d ≈ 13,5M y un denso de control con la forma de F. T20 dejó además el LR de
F en el borde inferior de su rejilla (3e-4), un cabo suelto que E5 cierra ampliando la rejilla
hacia abajo.

**Preguntas.** (1) A iso-parámetro, ¿mejora la calidad al subir r/d por la hipérbola, y de forma
monótona? (2) ¿Un denso con la forma de F (d=768, atención 192, FFN 640, mismos productos por
token que F) gana al denso estrecho Ds? (3) ¿Estaba el óptimo de F por debajo de 3e-4?

**Hipótesis.**
- H_forma (la forma importa): al menos 3 de las 4 formas mejoran a F★ en ≥ 0,01 y el orden en r/d
  es monótono. Se refuta si no.
- H_Dc (el denso con la forma de F es mejor referencia): Dc < Ds★ en ≥ 0,01 a 6000 pasos. Se refuta
  si no.
- H_X (una forma cierra el hueco): la mejor forma pasa la puerta de E1 (≤ F★ − 0,03 a 6000) y en la
  final queda por debajo del denso de referencia en 3/3.

Métrica: la de T16–T20, sin cambios (BPC de validación completo, `bank.val_bpc_full`, fp32).

## Diseño

Línea base ★, corpus, lote, semilla y harness **idénticos a T20** (`t20.py` importado, no copiado:
T21 añade brazos a `t20.ARMS` y extiende `t20.build` con el denso de control). Las corridas de T20
que cumplen exactamente el mismo protocolo se **reutilizan**, con su clave y el sello v1 de T20
anotados: F a 3e-4 y 6e-4, F576 a 3e-4, Ds a 6e-4 (etapa A de T20, semilla 17) y las seis finales
de Ds★ y F★ (etapa C de T20, 18 000 pasos, semillas 17/29/43).

| Brazo | Forma | r/d | Parámetros (vocab 1119) | Productos por token en las lineales, por bloque | Ahorro frente al denso de igual d (cuenta del RFC) |
|---|---|---|---|---|---|
| F | (768, 128) | 0,167 | 13 570 816 | 1 572 864 | 4,5× |
| F640 | (640, 155) | 0,242 | 13 522 400 | 1 587 200 | 3,1× |
| F576 | (576, 173) | 0,300 | 13 498 336 | 1 594 368 | 2,5× |
| F512 | (512, 196) | 0,383 | 13 507 200 | 1 605 632 | 2,0× |
| F448 | (448, 226) | 0,504 | 13 540 864 | 1 619 968 | 1,5× |
| Dc | denso d=768, atención 192 (8 cabezas de 24), FFN 640 | — | 13 566 720 | 1 572 864 (los de F) | — |
| Ds (referencia, T20) | denso d=368 | — | 13 472 112 | 1 625 088 (12·368²) | — |

Todas las formas son `GPT(fact_lin(r))` de mztrain con otro (d, r). Dc es `GPTForma`
(`mztrain_t21/forma.py`): mismo embedding, LayerNorm y flujo residual de 768 que F; lineales
densas sin bias; init `nn.Linear`.

**Smoke (GPU; no es dato; solo tras el «corre»).** Los cuatro brazos nuevos (F640, F512, F448 a
3e-4; Dc a 6e-4), 500 pasos con coseno sobre 500, semilla 17; reanudación desde el paso 150,
**reportada en los cuatro, ninguna exigible**: en T20 solo Ds (d=368, cabezas de 46) fue
determinista en GPU, y Dc comparte con F el flujo de 768 y cabezas de anchura múltiplo de 4
(24), con las que la atención usa otro kernel. La identidad de la recarga está probada en CPU
(tests) y en GPU con Ds en T20; aquí ningún reintento reanuda desde un checkpoint. Un smoke
anterior se archiva, no se pisa. El anclaje bit a bit con T19 ya lo hizo T20 con este mismo
harness, cuyos hashes forman parte del sello de T21.

**Etapa A, barrido** (6000 pasos, semilla 17, coseno sobre 6000; 15 corridas, ~2,5 h):
- F a 1,5e-4 (3e-4 y 6e-4 vienen de T20);
- F576 a 1,5e-4 y 6e-4 (3e-4 viene de T20);
- F640, F512 y F448 a {1,5e-4, 3e-4, 6e-4};
- Dc a 3e-4 y 6e-4, y un tercer LR por la regla: 1,2e-3 si ganó 6e-4, 1,5e-4 si ganó 3e-4 (en
  empate, el menor).
Orden: F; F640; F576; F512; F448; Dc. Evaluación cada 500 pasos, BPC de train cada 1000,
checkpoint final de todas, freno y vigía como en T20.

**Etapa B, final condicional** (18 000 pasos, semillas 17/29/43): solo si la mejor forma pasa la
puerta. Brazos de la final: la mejor forma a su mejor LR; además F a 1,5e-4 si su óptimo bajó de
3e-4 (regla 3); además Dc a su mejor LR si Dc es la referencia (regla 5). Rotados por semilla.
0, 3, 6 o 9 corridas (0, 1,5, 3 o 4,5 h). Sin final, la etapa B se cierra vacía.

## Reglas de decisión (fijadas antes; código: `analyze_t21.py`)

El BPC de decisión es el de validación en el último paso; no finito, incompleta, ausente **o
degenerada (≥ 3,0)** cuentan +∞ en todas las reglas. Fronteras inclusivas con tolerancia 1e-9.
Las claves de las corridas de la final llevan el prefijo `Bf.` para no coincidir con las de la
criba de T20 (`B.`), que T21 reutiliza como referencia.

1. **Mejor LR de un brazo:** el de menor BPC entre sus LRs corridos (en T21 y en la referencia de
   T20); en empate, el menor. Un óptimo en el borde de su rejilla se declara.
2. **LA FORMA IMPORTA:** al menos 3 de las 4 formas (F640, F576, F512, F448), cada una en su mejor
   LR, quedan ≤ BPC(F en su mejor LR) − 0,01, **y** la secuencia de BPC por r/d creciente
   (F640, F576, F512, F448) es no creciente con tolerancia 0,005. Una forma con BPC no finito
   rompe la monotonía.
3. **F POR DEBAJO DE LA REJILLA DE E1:** si F a 1,5e-4 mejora a F a 3e-4 en ≥ 0,01, el óptimo de
   F estaba fuera de la rejilla de T20; se declara, y la referencia de F para la final (si la hay)
   pasa a ser F a 1,5e-4 con tres semillas corridas en T21. La etiqueta de T20 no cambia (está
   sellada); el hueco F − Ds a 6000 pasos con el nuevo LR se reporta aparte.
4. **Puerta a la final:** la mejor forma (menor BPC en su mejor LR; en empate, la de menor r/d)
   pasa si BPC ≤ BPC(F en su mejor LR) − 0,03, la misma puerta que E1.
5. **DC ES LA REFERENCIA:** si Dc (su mejor LR) ≤ Ds★ (T20, 6e-4: 1,8526) − 0,01, el denso con la
   forma de F pasa a ser el denso de referencia para la final de T21 y para los experimentos
   posteriores; si hay final, Dc corre sus tres semillas.
6. **Final:** VICTORIA si X < denso de referencia (Ds★ de T20 o Dc) en 3/3, pareado por semilla,
   con el denso válido en las tres; RECORTA si X < F de referencia (F★ de T20 a 3e-4, o F a 1,5e-4
   si la regla 3 se disparó) en 3/3, **con F válida en las tres**, y media(F − X) ≥ 0,02, sin
   victoria; MUERE en otro caso; INCONCLUSO sustituye a VICTORIA si el denso de referencia falla
   o degenera en alguna semilla. Se reporta siempre X − Ds★ (T20), aunque Dc sea la referencia.
7. **Sin final:** si ninguna forma pasa la puerta, la etiqueta es SIN_FINALISTA y la conclusión
   es que subir r/d por la hipérbola no cierra el hueco a 6000 pasos.

**Descriptivos (no cambian la etiqueta):** la curva calidad–ahorro (BPC de cada forma frente a
su ahorro de parámetros respecto al denso de igual d); el hueco F − Ds con cada LR; sobreajuste
(subida > 0,02 desde el mínimo o brecha val − train > 0,15) y baches (> 0,3) por corrida; las
lecturas en CPU de `diag_t20` sobre las formas (flujo, factores, alineación, gradiente visible,
saliencia), junto a las de T20.

**Sin exclusiones ni repesca.** NaN o degeneración cuentan como fallo de esa corrida (y +∞ en las
reglas). Una corrida cortada por el freno o el vigilante se repite idéntica desde cero y se anota
en «reintentos» y en `DESVIACIONES-T21.md`. Toda comparación se declara «a presupuesto fijo».

## Predicciones (antes de correr; confianza baja)

| Predicción | P |
|---|---|
| LA FORMA IMPORTA (≥ 3 de 4 mejoran a F en ≥ 0,01 con orden monótono) | 0,45 |
| F448 es la mejor forma | 0,45; F512 0,30; F576 0,15; F640 0,10 |
| la mejor forma pasa la puerta (≤ F − 0,03 a 6000) | 0,35 |
| F a 1,5e-4 mejora a 3e-4 en ≥ 0,01 (regla 3) | 0,35 |
| el mejor LR de cada forma es 3e-4 | 0,5 por forma; 1,5e-4: 0,3; 6e-4: 0,2 |
| DC ES LA REFERENCIA (Dc ≤ Ds★ − 0,01) | 0,30; Dc dentro de ±0,01 de Ds★: 0,45 |
| si hay final: VICTORIA 0,25; RECORTA 0,45; MUERE 0,30 | |
| 0 NaN y 0 degeneradas en la etapa A | 0,7 (F a 1,2e-3 degeneró en T20; aquí ninguna forma pasa de 6e-4) |
| el reloj: etapa A entre 2 y 3 h con el freno activo | 0,7 |

Del RFC (sección 3.2): subir r/d alivia el cuello de rango (M36) y agranda la parte visible del
gradiente (M32), pero gasta más parámetros en gauge (de 4,2 % a 12,7 %) y acerca la clase de
funciones a un denso más estrecho que Ds. Una victoria aquí es «ganar acercándose al denso» y se
reporta con ese precio delante.

## Amenazas a la validez

1. Una escala, carácter, fp32, seq 128; una semilla en el barrido; tres en la final.
2. Rejilla de tres LRs por forma; el óptimo puede quedar en un borde (se declara).
3. Las corridas reutilizadas de T20 son del mismo harness y la misma semilla, pero se hicieron en
   otra sesión; F no es determinista a d=768 (reanudación no idéntica en T20).
4. La monotonía se juzga sobre cuatro puntos con una semilla; el σ entre semillas de F★ a 6000
   en las finales de T20 es 0,006, del orden de la tolerancia (0,005) y de la mejora mínima (0,01).
5. Dc tiene la forma de F pero no su número exacto de parámetros (−4096) ni su init espectral.
6. Los minutos llevan el freno térmico: no hay claims de velocidad.

## Reglas operativas

- **No se lanza nada en GPU hasta que el usuario diga «corre»**, tampoco el smoke.
- No se toca `D:\mztrain` ni las carpetas T9 y T16–T20: se importan. Los checkpoints y resultados
  de T21 van en su carpeta.
- Ningún subagente ejecuta `t21.py`, `t20.py` ni los lanzadores; los revisores trabajan con réplicas.
- Para parar: `parar_t21.ps1`.

## Artefactos

`t21.py`, `analyze_t21.py`, `diag_t21.py`, `mztrain_t21/forma.py`, `vigia_t21.ps1`, `lanzar_t21.ps1`,
`parar_t21.ps1`, `tests/`; `t21_smoke.json`, `t21_results.json` (con la referencia de T20 dentro),
`t21_decisiones.json`, `t21_analysis.json`, `diag_t21.json`, `freno_t21.csv`, `vigia_t21.log`,
`log_T21_*.txt`, `ckpt/`, `ckpt_smoke/`; `DESVIACIONES-T21.md` y `T21-VEREDICTO.md`.
