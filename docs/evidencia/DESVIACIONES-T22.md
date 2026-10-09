# T22 — Desviaciones e incidentes

Este fichero no entra en el sello y recoge cada corrida cortada, reintento, disparo térmico o
cambio posterior al sello (que exigiría un bloque nuevo con su nota). Antes del sello (v1,
2026-10-07 17:45) no hubo ningún incidente: nada se corrió en GPU. La revisión G4 (opus, réplica de
solo lectura; `.forja/revisor.json`) halló un defecto mayor antes de sellar, corregido: `decidir_A`
contaba como degeneradas también las finales, con lo que una final a ≥ 3,0 habría descuadrado
`t22_decisiones.json` y bloqueado la etapa B (el mismo patrón estaba en T21 y no se disparó).

## 1. Smoke y arranque de la cadena (2026-10-07, 17:48–18:20): el vigía no pudo escribir su log; vigiló igual

- Sesión `smoke-20261007-174833`. `vigia_t22.log` se detiene en la primera vuelta (17:48:34): ni el
  «temp» de los 300 s ni el fin del proceso. Diagnóstico equivocado al principio (que el shell de la
  sesión de Claude había matado al vigía); diagnóstico verificado después: el monitor de la sesión
  seguía el log con `tail -F` (GNU tail de Git), que abre el fichero sin compartir la escritura, y
  `Add-Content` del vigía falla con «siendo utilizado en otro proceso» mientras el tail está encima
  (reproducido sobre una copia). El error es no terminante: el vigía siguió vivo leyendo la GPU y
  terminó cuando terminó el python; solo perdió las líneas del log. En la cadena
  (`cadena-20261007-181359`, 18:13:59) pasó lo mismo desde el primer segundo (el tail del smoke seguía
  vivo y el nuevo monitor añadió otro): se pierden las líneas «inicio», «fase» y «proceso lanzado» de
  la etapa A. A las 18:20 se mataron los dos `tail` y el monitor pasó a leer los logs por sondeo
  (lectura y cierre inmediatos; el python escribe por su propio descriptor y no se ve afectado).
- El smoke corrió entero con el **freno térmico interno** activo (`freno_t22.csv`: 166 lecturas,
  47–74 °C, pausa 0 en todas) y terminó con salida normal: `t22_smoke.json` completo bajo el sello
  v1, las dos corridas finitas (Dsmuon 2,6817; Fspec 3,1293 a 500 pasos) y las dos reanudaciones
  idénticas en GPU (diff 0,0; ninguna exigible). El smoke no es dato; la cadena no se tocó.
- Lección para los próximos experimentos: no seguir `vigia_*.log` con `tail -f/-F`; sondear.

## 2. Ejecución de la cadena (2026-10-07 18:14 – 2026-10-08 01:00): sin desviaciones de protocolo

- Sesión `cadena-20261007-181359`, lanzada desde Bash: etapa A (7 corridas, 18:14–20:05) y etapa B
  (6 finales, 20:05–01:00) en un solo lanzamiento cada una, salida 0 en las dos fases, 0 cortes,
  0 reintentos, 0 NaN, 0 degeneradas, 13/13 checkpoints, stderr vacío. 403 min de pared en corridas.
- Térmica (freno_t22.csv desde las 18:14): 11 604 lecturas, 69,1 °C de media, 81 °C de máximo,
  freno activo en el 42 % de las lecturas, 538 lecturas ≥ 78 °C, 0 lecturas ≥ 83 °C; el vigía anotó
  118 avisos (≥ 78 °C) y ningún disparo. **Los tiempos no valen para comparar brazos.**
- La caminata de LR terminó en 4 LRs en Fspec (el tercero ganó; el cuarto no) y en 3 en Dsmuon;
  ningún óptimo en el borde. Ambos sellos (T22 v1 y T20 v1) vigentes durante toda la ejecución.
- Las líneas «inicio», «fase» y «proceso lanzado» de la etapa A faltan en `vigia_t22.log` por el
  incidente del apartado 1; desde las 18:19 el log está completo.

## 3. Errata documental (2026-10-08, tras la revisión matemática de `mzspectral`): sin cambios en lo sellado

Una revisión independiente del paquete extraído de T22 (`D:/mzspectral/.forja/auditoria-2026-10-08/INFORME-MATEMATICAS.md`)
señala tres imprecisiones de texto que también están en el docstring del fichero sellado
`mztrain_t22/espectral.py` y en el preregistro. No tocan el código que corrió ni ningún número; el
fichero sellado no se modifica (su errata queda aquí):

1. **Decaimiento efectivo de W.** El weight decay desacoplado actúa por factor: en Fspec decaen U
   y V con lr = 4,8e-3 y S con 3e-4, luego W decae por (1 − lr·wd)²(1 − lr_adam·wd) ≈ 9,9e-5 por
   paso al pico, 8,2× el de Dsmuon (1,2e-5) y 11× el de F★. Acumulado con el calendario real de
   18 000 pasos (verificado con `sano.lr_calendario`): Ds★ 0,95, Dsmuon 0,90, F★ 0,92, **Fspec
   0,41** (factor de encogimiento de W). El preregistro lo declaraba como «razón decaimiento/paso
   4–7 veces mayor» (amenaza 8); el enunciado correcto y más fuerte es este. Consecuencia para la
   lectura del veredicto: de F★ a Fspec cambian el optimizador y el decaimiento efectivo (×11), y
   Fspec − Dsmuon compara brazos con 8× de diferencia en regularización; la atribución de los
   0,063 entre las dos piezas requiere la ablación E9b (wd de U y V ÷ 8 o a 0).
2. **El diferencial del producto tiene tres términos**: ΔW = ΔU·diag(S)·V + U·diag(S)·ΔV +
   U·diag(ΔS)·V; el tercero lo mueve AdamW sin el factor 1/(σ_A + σ_B + 1) y escala con ‖U‖₂‖V‖₂
   (en T22 ≈ 5 % de la cota de U y V, por el LR 16 veces menor de S).
3. **Newton-Schulz** deja los valores singulares en [0,68, 1,13] solo para σᵢ/‖G‖_F ≳ 2e-3 (las
   direcciones más pequeñas quedan pequeñas); el RMS por entrada de la actualización escalada es
   0,17–0,20·lr, no 0,2 exacto.
