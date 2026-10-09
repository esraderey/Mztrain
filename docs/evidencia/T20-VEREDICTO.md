# T20 — Veredicto: con una línea base sana el sumidero desaparece en los dos modelos, pero el factorizado sigue por detrás del denso iso-parámetro en 3/3 (+0,051 BPC) y ninguna de las cuatro candidatas pasa la criba

**Fecha:** 2026-10-07. **Experimento:** E1 de `RFC-CALIDAD-1.md` (ejecución según `PROMPT-T20.md`).

**Preregistro:** `PREREGISTRO-T20.md`, sellado antes del smoke y de cualquier corrida
(`PREREGISTRO-T20.sha256`, bloque v1 de 2026-10-06 16:37, 56 ficheros: harness, paquete
`mztrain_t20`, tests, dependencias de T9/T17/T18, referencias de T19, corpus y el paquete `mztrain`
entero). El sello seguía vigente al terminar la cadena; no se tocó ningún fichero sellado.

**Desviaciones:** `DESVIACIONES-T20.md`. §1: un arranque accidental anterior al sello, sin datos ni
GPU (comparación de modos sin distinguir mayúsculas en PowerShell; corregido antes de hashear).
§2: **un corte térmico** en la etapa C a las 21:29 (tres lecturas a 83 °C) con la primera final de
Ds en curso; el vigía enfrió, relanzó la misma etapa y la corrida se repitió idéntica desde cero
(reintento anotado; las 29 evaluaciones de la corrida cortada coinciden con la repetida). Ningún
otro corte, 0 NaN en las finales, 0 degeneradas en las finales, 17/17 checkpoints, stderr vacío.

**Datos:** `t20_smoke.json` (smoke; no es dato), `t20_results.json` (17 corridas: 7 de calibración,
4 de criba, 6 finales), `t20_decisiones.json` (decisiones entre etapas, calculadas por
`analyze_t20.py` sin juicio humano), `t20_analysis.json`, `diag_t20.json` (lecturas en CPU),
`freno_t20.csv`, `vigia_t20.log`, `log_T20_*.txt`, `ckpt/` (17 checkpoints, 2,6 GB) y
`ckpt_smoke/` (1,9 GB, prescindible). Las tablas del anexo las genera `tablas_t20.py` (solo
formato, fuera del sello) en `t20_tablas.md`.

**Protocolo.** Línea base sana («★») igual para todos los brazos: `tok` y `pos` ~ N(0, 0,02²);
AdamW β = (0,9, 0,95), eps 1e-8, wd 0,01; recorte global del gradiente a 1,0; rampa de 200 pasos y
coseno hasta 0 en el último paso; LR pico calibrado por brazo; corpus char-WT103 (primeros 120M
caracteres, vocabulario de WT-2 más `<otro>`), validación de WT-2; lote 16 × 128; fp32; semillas
17, 29 y 43. Brazos: Ds (denso d=368), F (factorizado d=768, r=128), L (carriles con bus), F576
(d=576, r=173), Fcola (no linealidad entre factores), Fpolar (calibre polar de tres relojes).
**Todo es «a presupuesto fijo»**: 6000 pasos en la calibración y la criba, 18 000 en las finales.
Ninguna corrida había convergido.

## Resultado

| BPC de validación, media de 3 semillas | paso 6000 | paso 12 000 | paso 18 000 |
|---|---|---|---|
| Ds★ (denso iso-parámetro, LR 6e-4) | 1,9000 | 1,7422 | **1,6895** |
| F★ (factorizado r=128, LR 3e-4) | 1,9851 | 1,8062 | **1,7401** |
| F★ − Ds★ | +0,0851 | +0,0640 | **+0,0506** |
| arco (T19): Ds / F / hueco | 2,3390 / 2,5086 / +0,1697 | 2,0502 / 2,1877 / +0,1375 | 1,9579 / 2,0441 / +0,0862 |

Por semilla a 18 000 pasos: Ds★ 1,6847 / 1,6919 / 1,6919; F★ 1,7413 / 1,7368 / 1,7421; hueco
pareado +0,0567 / +0,0449 / +0,0502. En el arco, las mismas semillas daban +0,0988 / +0,0754 /
+0,0845.

## Control de anclaje: OK

El harness de T20 configurado como T19 (WT-2, embedding N(0,1), AdamW β2 = 0,999, LR 3e-4
constante tras la rampa, sin recorte) repitió `Ds/s17` de T19 **bit a bit** en los pasos 250 y
500: 4,138897 y 3,822008 (referencias de `t19_smoke.json` y `t19_results.json`). La reanudación
de Ds desde un checkpoint del paso 150 fue idéntica (diferencia 0,0). En CPU, el mismo montaje
reproduce la curva de `t19.run` punto a punto (test del harness). La reanudación de los cinco
brazos factorizados no fue idéntica (diferencias entre 1e-6 y 6,5e-3): no exigible, como preveía
el preregistro.

## Reglas aplicadas (todas preregistradas; `t20_decisiones.json`)

**Calibración (etapa A).**

| | 3e-4 | 6e-4 | 1,2e-3 | LR elegido | β2 |
|---|---|---|---|---|---|
| Ds | 1,9015 | **1,8526** | 1,8862 | 6e-4 | 0,95 (1,8526 frente a 1,8547 con 0,999) |
| F | **1,9172** | 2,0633 | 3,6144 (degenerada) | 3e-4, **en el borde de la rejilla**: se declara, sin más corridas | 0,95 (fijo) |

F degenera a 1,2e-3 pese al recorte y a β2 = 0,95 (norma del gradiente antes del recorte hasta
997; el 78 % de los pasos recortados). A 3e-4 recorta el 99,7 % de los pasos; Ds, a 6e-4, el 7 %.

**Reloj.** Etapa A real 61,0 min; proyección B 38,6 + C 257,1 = 356,7 ≤ 360: ninguna candidata
quitada. (El coste medido de L fue 1,42 veces el de F, no el 1,2 de la regla; la regla no se toca.)

**Criba (etapa B).** Puerta: BPC ≤ 1,9172 − 0,03 = 1,8872.

| Candidata | LR | BPC paso 6000 | frente a F★ | ¿pasa? |
|---|---|---|---|---|
| Fpolar | 6e-4 (el doble) | 1,9657 | +0,0485 | no |
| L | 3e-4 | 1,9182 | +0,0010 | no |
| Fcola | 3e-4 | 2,0323 | +0,1151 | no |
| F576 | 3e-4 | 1,8975 | **−0,0197** | no |

**Sin finalista.** La etapa C corrió solo Ds y F (6 corridas).

**Etiqueta: SIN_FINALISTA.** Ninguna de las cuatro causas (optimización de los factores, rango
completo al coste de F, forma de la hipérbola, cuello lineal) mueve a F★ 0,03 a 6000 pasos.

**EL HUECO ERA DEL MONTAJE: NO.** F★ < Ds★ en 0/3 (Ds★ válido en las tres semillas: finito, sin
degeneración). El hueco iso-parámetro a 18 000 pasos con línea base sana es **+0,0506 de media**,
positivo en las tres semillas.

**INCONCLUSO:** no procede (Ds★ válido en 3/3).

**Sobreajuste: ninguna comparación contaminada.** El mínimo de validación de las seis finales
está en el paso 18 000 (subida 0,0) y la brecha val − train a 18 000 es de +0,002 a +0,011 (en el
arco, con WT-2 multiépoca, llegaba a +0,137). **Baches:** ninguno en las 17 corridas (saltos > 0,3
entre evaluaciones). Degenerada: solo `A.F.lr0.0012` (calibración; cuenta +∞, sin exclusión).

## Descriptivos

- **El hueco se estrecha con los pasos** también bajo ★: +0,085 → +0,064 → +0,051 (arco: +0,170 →
  +0,138 → +0,086). Sobre las curvas medias, F★(t) ≈ Ds★(0,69·t), 0,76·t y 0,67·t en los pasos
  6000, 12 000 y 18 000 (arco: 0,68 desde el 12 000). Con coseno a cero la lectura «ritmo» es
  menos limpia que con LR constante, porque el LR de los dos brazos no es el mismo en el mismo
  paso. Pendiente de los últimos 1000 pasos: F★ −0,0008, Ds★ −0,0006 por 1000 pasos; las dos
  casi planas porque el LR llega a 0.
- **Las dos líneas base mejoran mucho al arco a presupuesto igual** (Ds 1,9579 → 1,6895; F
  2,0441 → 1,7401), pero el cambio no es solo de ★: también cambian el corpus (WT-103, sin
  repetir épocas) y la programación del LR. La comparación directa con las cifras absolutas del
  arco no es válida (lo advertía el preregistro).
- **Candidatas.** F576 va por debajo de F★ en toda la criba (hasta −0,043 en el paso 2000; −0,020
  al final): la única con signo favorable, lejos de la puerta. L va por debajo de F★ hasta el
  paso 4000 (como mucho −0,018 en el 1000) y empata al final; es, además, 1,42 veces más lenta.
  Fpolar al doble de LR queda +0,048 por encima de F★, pero muy por debajo de lo que pierde F al
  doblar su LR (F a 6e-4: +0,146): el calibre tolera el doble de LR, no lo aprovecha. Fcola queda
  +0,115 por encima desde el paso 500.
- **β2 del denso:** 0,95 y 0,999 difieren en 0,002 a 6000 pasos. Sin efecto medible.
- **Térmica:** cadena de 18:54 a 00:52 (5 h 58 min con el enfriado tras el corte), 283 min de
  pared en corridas; GPU a 73,4 °C de media y 84 °C de máximo; freno activo (pausa > 0) en el 97 %
  de las lecturas; 344 avisos ≥ 78 °C, 5 lecturas a 83 °C, 1 disparo. **Los tiempos no valen para
  comparar brazos y este veredicto no hace ningún claim de velocidad.**

## Iso-FLOP y VRAM

Productos por token en las lineales (cuenta del RFC): Ds 14,17M; F 15,01M; L como F; F576
14,58M; Fcola y Fpolar como F. Iso-parámetro e iso-FLOP coinciden dentro de un 6 %. VRAM de pico
medida (fp32, lote 16 × 128): Ds 0,91 GiB; F, Fpolar 1,30; Fcola 1,33; L 1,26; F576 1,14. Estado de
AdamW: 102,8–103,5 MiB en todos. No hay finalista que pierda nada respecto a F.

## Lecturas en CPU (preregistradas; sin efecto en las etiquetas; `diag_t20.json`)

Sobre 64 ventanas de validación, en los 6 checkpoints finales y los 7 de la calibración:

| | F★ finales (3) | Ds★ finales (3) | arco T19, F | arco T19, Ds |
|---|---|---|---|---|
| rms del flujo residual final | 9,0–10,1 | 7,6–7,7 | 273–281 | 66–67 |
| energía del flujo final que es un vector constante | 13,0–13,3 % | 9,2–10,4 % | 96,4–96,8 % | 96,2–97,0 % |
| tras la LN final: energía dependiente del contexto | 84,7–84,9 % | 86,1–87,2 % | 0,10–0,13 % | 0,53–0,55 % |
| f·d² | ≈ 500 000 | ≈ 117 000 | 594–770 | 713–749 |
| dimensión efectiva de esa parte | 13,2–14,0 | 16,8–17,1 | 9,2–12,3 | 14,9–16,6 |
| bloques muertos (rms de escritura < 0,05) | ninguno | ninguno | F_RS: 1–2 | ninguno |
| mayor escritura (rms) | FFN del bloque 8, 3,6–6,2 | FFN del bloque 8, 4,0–4,1 | una FFN, 137–229 | 12–13 |
| rms de `tok` (arranque: 0,02 aquí, 1,0 en el arco) | 0,10 | 0,16–0,21 | 0,95 | 0,95 |
| cos(h final, embedding de entrada) | −0,01 a −0,02 | −0,02 a −0,04 | +0,03 | +0,04 |

**HECHO.** Con ★ el cuadro del arco no se reproduce en ninguna final de F★: rms final 9–10 (criterio
> 100), fracción constante 13 % (> 85 %), parte contextual 85 % (< 0,2 %). **El diagnóstico del
sumidero no queda refutado**: la línea base sana elimina el sumidero en los dos modelos, la
relación f·d² ≈ 700 desaparece y el embedding sí se aprende (su rms crece de 0,02 a 0,10–0,21,
frente al arco, donde se quedaba en el aleatorio del arranque). **Y aun así el hueco persiste**
(+0,051 en 3/3). INFERENCIA: el sumidero era real y dependía del arranque, pero no era la causa
principal del hueco; a lo sumo explica la parte que se recorta (de +0,086 a +0,051), y esa parte
está confundida con el cambio de corpus y de programación.

**Factores de F★ a 18 000 pasos** (media de bloques; arranque: normas 1, ortogonalidad 0, rango
estable 91–105): |S| 1,3–2,8 (baja un 6–10 %, como en el arco); columnas de U a norma 1,1–1,7 y
filas de V a 1,06–1,83 (la deriva de escala sigue); ortogonalidad rms 0,04–0,08, máximo hasta
0,36; **rango estable de W 24–28 en qkv y proj, 9 en fc1, 15–17 en fc2** (arco: 5–16). F ve el
39–59 % de la energía del gradiente de cada matriz (isótropo: 20–31 %; arco 26–65 %). La
alineación escritura → lectura es 0,20–0,21 (azar 0,167; arco 0,18–0,20): sigue siendo casi azar.
**Saliencia:** ninguna componente por debajo de 0,1·mediana en ningún bloque (arco: 2–43 % en los
bloques 5–8): sin el sumidero no hay componentes recicables, y **E8 (rango nómada) pierde su
condición de entrada**. Fpolar mantuvo las normas de U y V a 1 (desviación máxima 1e-7).

## Predicciones frente a resultado

| Predicción (antes de correr) | P | Resultado |
|---|---|---|
| anclaje de `Ds/s17` exacto en los pasos 250 y 500 | 0,9 | ✓ |
| Ds★ < 1,9579 y F★ < 2,0441 | 0,8 | ✓ (1,6895 y 1,7401) |
| hueco F★ − Ds★ a 18k ∈ [0,02, 0,08] | 0,5 | ✓ (+0,0506) |
| EL HUECO ERA DEL MONTAJE | 0,10 | no (F★ gana en 0/3) |
| LR elegido: Ds ∈ {1,2e-3, 2,4e-3}; F ∈ {6e-4, 1,2e-3} | 0,6; 0,65 | **✗ los dos**: Ds 6e-4, F 3e-4 (borde) |
| β2 = 0,95 gana en Ds | 0,6 | ✓ (por 0,002) |
| el reloj no quita ninguna candidata | 0,75 | ✓ |
| alguna candidata pasa la puerta | 0,35 | no (la mejor, F576, a −0,020) |
| Fpolar no diverge al doble de LR y sin saltos > 0,3 | 0,7 | ✓ |
| etiqueta MUERE o SIN_FINALISTA | 0,63 | ✓ (SIN_FINALISTA) |
| 0 NaN y 0 degeneradas en Ds★ | 0,9 | ✓ |
| F★ no reproduce el cuadro del arco | 0,7 | ✓ (en las 3 finales) |
| ninguna final con sobreajuste marcado | 0,8 | ✓ |
| RFC: la curva temprana de L por debajo de la de F★ | — | ✓ en los 6 puntos hasta el 3000, por ≤ 0,018; desaparece en el 5000 |
| RFC: Fpolar tolera y aprovecha un LR doble | — | tolera (no diverge; pierde 0,048 donde F pierde 0,146), no aprovecha |
| innovadores: Fpolar 15 %, L 15 %, F576 20 %; F★ sola 10 % | — | ninguna ganó; la única por debajo de F★ fue F576 |

Lo que falló en mis predicciones fue el LR: con recorte y β2 = 0,95 esperaba picos más altos, y los
dos modelos prefirieron picos bajos; F ni siquiera sobrevive a 1,2e-3.

## Amenazas a la validez

1. Una escala (13,5M), carácter, fp32, seq 128.
2. La criba elige con una semilla a 6000 pasos y una puerta de 0,03. F576 quedó a −0,020: una
   diferencia real a esa escala (el σ entre semillas de F★ a 6000 en la final es 0,006) que la
   puerta no deja pasar; no se sabe si se sostendría a 18 000 pasos ni en 3/3.
3. El LR de F cayó en el borde de la rejilla (3e-4): no se exploró 1,5e-4. El denso se calibró
   solo en LR y β2. F576 y las demás candidatas usan el LR de F★, no el suyo.
4. El LR se eligió a 6000 pasos con coseno sobre 6000 y se transfirió a 18 000 con coseno sobre
   18 000.
5. El cambio de corpus y de programación impide atribuir a ★ sola la mejora absoluta frente al
   arco; solo el anclaje bit a bit garantiza que el harness es el mismo.
6. F no es determinista a d=768 (reanudación no idéntica); Ds sí (repetición idéntica tras el corte).
7. Tres semillas; σ de F★ a 18 000 es 0,003 y la de Ds★ 0,004, muy por debajo del hueco (0,051).
8. Los minutos por 1000 pasos llevan el freno: la regla del reloj se aplicó con una medida
   conservadora, y L costó 1,42× en vez del 1,2× supuesto.

## Qué desbloquea (sección 5.4 del RFC)

- **SIN_FINALISTA → E5 (curva de forma) y E9 (optimizador espectral).** E5 es el siguiente
  natural: F576 fue la única candidata por debajo de F★ y el RFC ya preveía la hipérbola
  completa (640/155, 512/196, 448/226) con un denso de control de la forma de F. Conviene
  calibrar el LR de cada forma, no heredar el de F★ (amenaza 3).
- **La línea base sana se queda:** para cualquier experimento posterior sobre este montaje, ★ es
  la línea base (elimina el sumidero, sin baches, sin sobreajuste a 18 000 pasos sobre WT-103).
- **E8 (rango nómada) pierde su condición de entrada:** sin componentes de saliencia baja no hay
  rango que reciclar.
- **E6 (ablación de ★) sigue teniendo interés** para atribuir la parte del hueco que se recortó
  (+0,086 → +0,051) entre el init del embedding, β2, el recorte, el coseno y el corpus.
- **El claim de calidad por parámetro de MZTrain no cambia:** el factorizado sigue perdiendo
  contra el denso iso-parámetro en 3/3, ahora con una línea base que nadie puede llamar débil.
  Si el objetivo sigue siendo ganar a iso-parámetro, los supuestos que quedan son dos: que el
  rango completo a coste de F con otra forma baste (E5) o que AdamW sea el límite (E9). La
  métrica legítima mientras tanto es la del RFC 5.6: calidad por GB de VRAM frente al denso de
  la misma forma, no frente al iso-parámetro.

## Anexo: tablas generadas por `tablas_t20.py` desde los JSON de T20

### Smoke (no es dato; 500 pasos, coseno sobre 500, semilla 17)

| Corrida | BPC paso 500 | min/1000 pasos | VRAM pico (GiB) | reanudación (diff) | exigible |
| --- | --- | --- | --- | --- | --- |
| S.arco.Ds.lr0.0003.b0.999.s17 | 3,8220 | 1.448 | 0.93 | anclaje: {'250': 4.138897, '500': 3.822008} | — |
| S.Ds.lr0.0006.b0.95.s17 | 3,1612 | 1.230 | 0.91 | +0,0000 | sí |
| S.F.lr0.0006.b0.95.s17 | 3,2032 | 1.556 | 1.30 | +0,0002 | no |
| S.F576.lr0.0006.b0.95.s17 | 3,2955 | 1.417 | 1.14 | -0,0000 | no |
| S.Fcola.lr0.0006.b0.95.s17 | 3,3488 | 1.406 | 1.33 | +0,0012 | no |
| S.L.lr0.0006.b0.95.s17 | 3,1464 | 2.318 | 1.26 | +0,0000 | no |
| S.Fpolar.lr0.0012.b0.95.s17 | 3,4277 | 1.653 | 1.30 | -0,0065 | no |

### Calibración (etapa A; 6000 pasos, semilla 17, coseno sobre 6000)

| Corrida | LR | β2 | BPC paso 6000 | gn media / máx | pasos recortados | min/1000 |
| --- | --- | --- | --- | --- | --- | --- |
| Ds | 0.0006 | 0.95 | 1,8526 | 0.72 / 10.6 | 6.7 % | 1.299 |
| F | 0.0006 | 0.95 | 2,0633 | 1.05 / 18.4 | 47.4 % | 1.515 |
| Ds | 0.0012 | 0.95 | 1,8862 | 0.52 / 10.6 | 1.7 % | 1.406 |
| F | 0.0012 | 0.95 | 3,6144 (degenerada) | 3.60 / 996.8 | 77.9 % | 1.485 |
| Ds | 0.0003 | 0.95 | 1,9015 | 1.13 / 10.6 | 77.6 % | 1.443 |
| F | 0.0003 | 0.95 | 1,9172 | 1.21 / 18.4 | 99.7 % | 1.590 |
| Ds | 0.0006 | 0.999 | 1,8547 | 0.72 / 10.6 | 4.2 % | 1.433 |

Decisiones (reglas 1–3): LR de Ds = 0.0006 (borde: no); LR de F = 0.0003 (borde: sí); β2 de Ds = 0.95 (1,8526 frente a 1,8547). Reloj: A real 61.0 min; proyección B 38.6 + C 257.1 = total 356.7 min ≤ 360: candidatas quitadas: ninguna.

### Criba (etapa B; 6000 pasos, semilla 17)

Puerta: BPC ≤ BPC(F★ en A) − 0,03 = 1,9172 − 0,03 = 1,8872.

| Candidata | LR | BPC paso 6000 | diferencia con F★ | pasa | min/1000 | VRAM pico (GiB) | pasos recortados |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fpolar | 0.0006 | 1,9657 | +0,0485 | no | 1.618 | 1.30 | 97.0 % |
| L | 0.0003 | 1,9182 | +0,0010 | no | 2.173 | 1.26 | 100.0 % |
| Fcola | 0.0003 | 2,0323 | +0,1151 | no | 1.575 | 1.33 | 100.0 % |
| F576 | 0.0003 | 1,8975 | -0,0197 | no | 1.546 | 1.14 | 87.8 % |

Finalista: ninguna.

Curvas de la criba frente a F★ (misma semilla y, salvo Fpolar, mismo LR): diferencia de BPC por paso

| Candidata | 500 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fpolar | +0,0065 | +0,1026 | +0,1485 | +0,1014 | +0,0800 | +0,0572 | +0,0485 |
| L | -0,0018 | -0,0185 | -0,0103 | -0,0114 | -0,0037 | -0,0001 | +0,0010 |
| Fcola | +0,1614 | +0,1919 | +0,1443 | +0,1120 | +0,1139 | +0,1147 | +0,1151 |
| F576 | -0,0021 | -0,0311 | -0,0428 | -0,0359 | -0,0211 | -0,0208 | -0,0197 |

### Finales (etapa C; 18 000 pasos; BPC de validación completo)

| Brazo | Paso | s17 | s29 | s43 | media |
| --- | --- | --- | --- | --- | --- |
| Ds★ | 6000 | 1,8981 | 1,9040 | 1,8979 | 1,9000 |
|  | 12000 | 1,7364 | 1,7451 | 1,7450 | 1,7422 |
|  | 18000 | 1,6847 | 1,6919 | 1,6919 | 1,6895 |
| F★ | 6000 | 1,9909 | 1,9794 | 1,9850 | 1,9851 |
|  | 12000 | 1,8056 | 1,8023 | 1,8107 | 1,8062 |
|  | 18000 | 1,7413 | 1,7368 | 1,7421 | 1,7401 |

| Hueco F★ − Ds★ | s17 | s29 | s43 | media |
| --- | --- | --- | --- | --- |
| paso 6000 | +0,0928 | +0,0754 | +0,0870 | +0,0851 |
| paso 12000 | +0,0692 | +0,0572 | +0,0657 | +0,0640 |
| paso 18000 | +0,0567 | +0,0449 | +0,0502 | +0,0506 |
| arco (T19), paso 18 000 | +0,0988 | +0,0754 | +0,0845 | +0,0862 |

### Hueco medio por paso en las finales (media de 3 semillas)

| Paso | F★ | Ds★ | F★ − Ds★ |
| --- | --- | --- | --- |
| 1500 | 2,4585 | 2,3182 | +0,1402 |
| 3000 | 2,1675 | 2,0633 | +0,1041 |
| 4500 | 2,0595 | 1,9652 | +0,0943 |
| 6000 | 1,9851 | 1,9000 | +0,0851 |
| 7500 | 1,9292 | 1,8513 | +0,0780 |
| 9000 | 1,8845 | 1,8096 | +0,0749 |
| 10500 | 1,8411 | 1,7722 | +0,0689 |
| 12000 | 1,8062 | 1,7422 | +0,0640 |
| 13500 | 1,7775 | 1,7185 | +0,0590 |
| 15000 | 1,7546 | 1,7005 | +0,0540 |
| 16500 | 1,7425 | 1,6913 | +0,0512 |
| 18000 | 1,7401 | 1,6895 | +0,0506 |

### Estado de las corridas de la cadena

| Corrida | estado | pasos | min de pared | min/1000 | VRAM pico (GiB) | AdamW máx (MiB) | gn media / máx | recortados | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A.Ds.lr0.0006.b0.95.s17 | ok | 6000 | 7.8 | 1.299 | 0.91 | 102.8 | 0.72 / 10.6 | 6.7 % | ok |
| A.F.lr0.0006.b0.95.s17 | ok | 6000 | 9.1 | 1.515 | 1.30 | 103.5 | 1.05 / 18.4 | 47.4 % | ok |
| A.Ds.lr0.0012.b0.95.s17 | ok | 6000 | 8.4 | 1.406 | 0.91 | 102.8 | 0.52 / 10.6 | 1.7 % | ok |
| A.F.lr0.0012.b0.95.s17 | ok | 6000 | 8.9 | 1.485 | 1.30 | 103.5 | 3.60 / 996.8 | 77.9 % | ok |
| A.Ds.lr0.0003.b0.95.s17 | ok | 6000 | 8.7 | 1.443 | 0.91 | 102.8 | 1.13 / 10.6 | 77.6 % | ok |
| A.F.lr0.0003.b0.95.s17 | ok | 6000 | 9.5 | 1.590 | 1.30 | 103.5 | 1.21 / 18.4 | 99.7 % | ok |
| A.Ds.lr0.0006.b0.999.s17 | ok | 6000 | 8.6 | 1.433 | 0.91 | 102.8 | 0.72 / 10.6 | 4.2 % | ok |
| B.Fpolar.lr0.0006.b0.95.s17 | ok | 6000 | 9.7 | 1.618 | 1.30 | 103.5 | 1.16 / 18.3 | 97.0 % | ok |
| B.L.lr0.0003.b0.95.s17 | ok | 6000 | 13.0 | 2.173 | 1.26 | 103.5 | 1.27 / 17.8 | 100.0 % | ok |
| B.Fcola.lr0.0003.b0.95.s17 | ok | 6000 | 9.4 | 1.575 | 1.33 | 103.5 | 1.48 / 24.9 | 100.0 % | ok |
| B.F576.lr0.0003.b0.95.s17 | ok | 6000 | 9.3 | 1.546 | 1.14 | 103.0 | 1.12 / 16.6 | 87.8 % | ok |
| C.Ds.lr0.0006.b0.95.s17 | ok | 18000 | 28.1 | 1.562 | 0.91 | 102.8 | 0.54 / 10.6 | 2.2 % | ok |
| C.F.lr0.0003.b0.95.s17 | ok | 18000 | 28.9 | 1.607 | 1.30 | 103.5 | 0.99 / 18.4 | 26.4 % | ok |
| C.F.lr0.0003.b0.95.s29 | ok | 18000 | 35.2 | 1.956 | 1.30 | 103.5 | 0.99 / 20.9 | 25.2 % | ok |
| C.Ds.lr0.0006.b0.95.s29 | ok | 18000 | 25.6 | 1.420 | 0.91 | 102.8 | 0.55 / 12.7 | 1.5 % | ok |
| C.Ds.lr0.0006.b0.95.s43 | ok | 18000 | 28.8 | 1.599 | 0.91 | 102.8 | 0.54 / 10.6 | 1.6 % | ok |
| C.F.lr0.0003.b0.95.s43 | ok | 18000 | 34.3 | 1.903 | 1.30 | 103.5 | 1.00 / 22.3 | 30.8 % | ok |

Reintentos: ['C.Ds.lr0.0006.b0.95.s17']. Lanzamientos: 4. Sello: `# v1 2026-10-06T16:37:22-0600 - antes del smoke y de cualqui…`.

### Sobreajuste y baches (finales; descriptivo)

| Corrida | val mín | paso del mín | val(18k) − mín | train 18k | brecha val − train 18k | brecha paso 1000 | saltos > 0,3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C.Ds.lr0.0006.b0.95.s17 | 1,6847 | 18000 | +0,0000 | 1,6741 | +0,0106 | +0,0548 | 0 |
| C.F.lr0.0003.b0.95.s17 | 1,7413 | 18000 | +0,0000 | 1,7345 | +0,0068 | +0,0543 | 0 |
| C.F.lr0.0003.b0.95.s29 | 1,7368 | 18000 | +0,0000 | 1,7282 | +0,0085 | +0,0483 | 0 |
| C.Ds.lr0.0006.b0.95.s29 | 1,6919 | 18000 | +0,0000 | 1,6858 | +0,0060 | +0,0711 | 0 |
| C.Ds.lr0.0006.b0.95.s43 | 1,6919 | 18000 | +0,0000 | 1,6898 | +0,0021 | +0,0476 | 0 |
| C.F.lr0.0003.b0.95.s43 | 1,7421 | 18000 | +0,0000 | 1,7398 | +0,0023 | +0,0401 | 0 |

Comparación contaminada por sobreajuste: no. Baches en toda la cadena: 0. Degeneradas: ['A.F.lr0.0012.b0.95.s17'].

### Lecturas en CPU (64 ventanas de validación; `diag_t20.json`)

| Checkpoint | rms flujo final | fracción constante | parte contextual tras LN | f·d² | dim. efectiva del contexto | bloques muertos | mayor escritura | rms de tok | cos(h, embedding) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A.Ds.lr0.0006.b0.95.s17 | 4.44 | 9.5 % | 86.0 % | 116436 | 17.3 | — | bloque 8 | 0.150 | +0.007 |
| A.F.lr0.0006.b0.95.s17 | 20.97 | 23.0 % | 75.4 % | 444735 | 13.7 | — | bloque 1 | 0.087 | +0.006 |
| A.Ds.lr0.0012.b0.95.s17 | 12.34 | 16.9 % | 83.2 % | 112706 | 18.9 | — | bloque 1 | 0.159 | -0.017 |
| A.F.lr0.0012.b0.95.s17 | 30180.05 | 76.3 % | 36.0 % | 212527 | 4.2 | — | bloque 4 | 0.072 | -0.002 |
| A.Ds.lr0.0003.b0.95.s17 | 2.02 | 10.2 % | 84.9 % | 115001 | 16.0 | — | bloque 8 | 0.128 | +0.008 |
| A.F.lr0.0003.b0.95.s17 | 4.78 | 11.7 % | 86.2 % | 508225 | 15.6 | — | bloque 8 | 0.081 | -0.004 |
| A.Ds.lr0.0006.b0.999.s17 | 4.70 | 18.7 % | 75.8 % | 102662 | 19.6 | — | bloque 1 | 0.045 | +0.106 |
| C.Ds.lr0.0006.b0.95.s17 | 7.59 | 9.4 % | 87.2 % | 118086 | 17.1 | — | bloque 8 | 0.179 | -0.024 |
| C.F.lr0.0003.b0.95.s17 | 9.68 | 13.0 % | 84.7 % | 499499 | 13.2 | — | bloque 8 | 0.107 | -0.021 |
| C.F.lr0.0003.b0.95.s29 | 9.04 | 13.3 % | 84.9 % | 500679 | 14.0 | — | bloque 8 | 0.103 | -0.010 |
| C.Ds.lr0.0006.b0.95.s29 | 7.72 | 9.2 % | 86.8 % | 117563 | 16.9 | — | bloque 8 | 0.205 | -0.033 |
| C.Ds.lr0.0006.b0.95.s43 | 7.74 | 10.4 % | 86.1 % | 116581 | 16.8 | — | bloque 8 | 0.164 | -0.035 |
| C.F.lr0.0003.b0.95.s43 | 10.08 | 13.3 % | 84.9 % | 500793 | 13.3 | — | bloque 8 | 0.102 | -0.018 |
| arco T19, F (media 3) | 276.6 | 96.6 % | 0.12 % | ~700 | 9–12 | F_RS: sí | FFN bloque 6–7 | 0,95 | +0,03 |
| arco T19, Ds (media 3) | 66.7 | 96.7 % | 0.54 % | ~730 | 15–17 | no | bloques 5–8 | 0,95 | +0,04 |

Factores de F★ en las finales (media de los 8 bloques; arranque: normas 1, ortogonalidad 0, rango estable 91–105):

| Corrida | capa | \|S\| medio | norma cols U | norma filas V | ortog. U rms / máx | rango estable de W | visible del gradiente (isótropo) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C.F.lr0.0003.b0.95.s17 | qkv | 2.30 | 1.35 | 1.06 | 0.048 / 0.36 | 27.5 | 0.41 (0.21) |
| C.F.lr0.0003.b0.95.s17 | proj | 1.33 | 1.12 | 1.09 | 0.042 / 0.16 | 24.4 | 0.57 (0.31) |
| C.F.lr0.0003.b0.95.s17 | fc1 | 2.77 | 1.71 | 1.19 | 0.083 / 0.31 | 8.6 | 0.39 (0.20) |
| C.F.lr0.0003.b0.95.s17 | fc2 | 1.43 | 1.24 | 1.83 | 0.057 / 0.20 | 15.1 | 0.42 (0.20) |
| C.F.lr0.0003.b0.95.s29 | qkv | 2.29 | 1.34 | 1.06 | 0.048 / 0.32 | 27.0 | 0.41 (0.21) |
| C.F.lr0.0003.b0.95.s29 | proj | 1.32 | 1.11 | 1.09 | 0.044 / 0.18 | 23.4 | 0.58 (0.31) |
| C.F.lr0.0003.b0.95.s29 | fc1 | 2.77 | 1.71 | 1.20 | 0.082 / 0.31 | 9.0 | 0.39 (0.20) |
| C.F.lr0.0003.b0.95.s29 | fc2 | 1.43 | 1.25 | 1.83 | 0.060 / 0.21 | 14.6 | 0.43 (0.20) |
| C.F.lr0.0003.b0.95.s43 | qkv | 2.30 | 1.36 | 1.06 | 0.049 / 0.35 | 28.5 | 0.42 (0.21) |
| C.F.lr0.0003.b0.95.s43 | proj | 1.33 | 1.12 | 1.10 | 0.041 / 0.16 | 26.4 | 0.56 (0.31) |
| C.F.lr0.0003.b0.95.s43 | fc1 | 2.77 | 1.71 | 1.19 | 0.083 / 0.31 | 8.8 | 0.39 (0.20) |
| C.F.lr0.0003.b0.95.s43 | fc2 | 1.43 | 1.25 | 1.82 | 0.056 / 0.19 | 16.8 | 0.42 (0.20) |

| Corrida F★ | alineación escritura→lectura (azar 0,167) | máx | mismo bloque proj→fc1 | componentes con saliencia < 0,1·mediana |
| --- | --- | --- | --- | --- |
| C.F.lr0.0003.b0.95.s17 | 0.205 | 0.265 | 0.224 | 0.00 % |
| C.F.lr0.0003.b0.95.s29 | 0.203 | 0.262 | 0.223 | 0.00 % |
| C.F.lr0.0003.b0.95.s43 | 0.206 | 0.251 | 0.226 | 0.02 % |

