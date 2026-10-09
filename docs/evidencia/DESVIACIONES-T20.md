# T20 — Desviaciones e incidentes

## 1. Arranque accidental durante los tests del vigía (2026-10-06 15:52; sin datos, sin GPU, antes del sello)

**Qué pasó.** Dos tests que comprueban que un modo mal escrito no lanza nada pasaron `-Modo a` a
`vigia_t20.ps1` y a `lanzar_t20.ps1`. Las comparaciones `-contains` / `-notcontains` de PowerShell
no distinguen mayúsculas: `a` pasó por `A` y los dos scripts lanzaron la fase real
`python -u t20.py --etapa a` en la carpeta del experimento (pids 5536 y 19048).

**Qué hicieron esos procesos.** Salieron con código 2 a los ~10 s en el parser de argumentos
(`argument --etapa: invalid choice: 'a'`), antes de verificar el sello y antes de cargar datos.
**No usaron la GPU, no entrenaron ningún paso y no escribieron resultados:** no se crearon
`t20_results.json`, `t20_smoke.json`, `t20_decisiones.json`, `freno_t20.csv` ni `ckpt/`.
`nvidia-smi` no muestra ningún python. El lanzador había quitado `CUDA_VISIBLE_DEVICES` del
entorno (como hace en los modos reales), así que la GPU era visible para un proceso que murió en
el parser.

**Rastro, conservado tal cual:** `vigia_t20.log` (dos sesiones, `20261006-155211` y
`a-20261006-155228`, ambas «proceso termino exit=2»), `err_T20_20261006-155211_0.txt`,
`err_T20_a-20261006-155228_0.txt` (el mensaje del parser) y los dos `log_T20_*` vacíos.

**Los tests lo detectaron:** los dos fallaron porque aparecieron ficheros nuevos en la carpeta y
porque el lanzador devolvió 1 (vigía lanzado) en vez de 2. Es el mismo tipo de agujero que el
incidente de T19 (argumentos desconocidos ignorados), ahora en el valor del modo.

**Qué se cambió por esto, antes de hashear:**
- `vigia_t20.ps1` y `lanzar_t20.ps1` comparan el modo con `-ccontains`, `-cnotcontains`, `-ceq`,
  `-cne` y `switch -CaseSensitive`: `a`, `b`, `c`, `CADENA`, `Smoke` o `Prueba` ya no son modos;
- tests nuevos para esas variantes en `tests/test_vigia.py` (ninguno usa un modo real válido).

**Efecto sobre el experimento:** ninguno. No hay datos de ese arranque; el preregistro, las reglas
y el análisis no cambian por él, y el sello es posterior.

## 2. Ejecución (2026-10-06, desde las 18:40)

- **Smoke** (sesión `smoke-20261006-184004`, 18:40–18:53, salida 0): anclaje de `Ds/s17` con T19
  exacto en los pasos 250 y 500 (4,138897 y 3,822008); reanudación de Ds idéntica (diff 0,0);
  seis brazos con pérdida finita. La reanudación de F, F576, Fcola, L y Fpolar no fue idéntica
  (diferencias entre 1e-6 y 6,5e-3), como preveía el preregistro: no exigible.
- **Cadena** (sesión `cadena-20261006-185401`, desde las 18:54). Etapas A (18:54–19:57) y B
  (19:57–20:39) completas en un solo lanzamiento, sin cortes ni reintentos; máximo 80 °C.
- **Corte térmico en la etapa C (21:29).** Tres lecturas seguidas a 83 °C con la primera final
  (`C.Ds.lr0.0006.b0.95.s17`) en curso, pasado el paso 12 000. El vigía detuvo el proceso
  (reinicio 1 de 2), enfría a ≤ 60 °C y relanza `--etapa C`; `t20.py` repite esa corrida idéntica
  desde cero y la anota en «reintentos». Ningún dato de la corrida cortada se usa. Con el freno
  adaptativo activo casi todo el tiempo, cada final tarda unos 60–70 min en vez de los 27 del
  coste plano: la regla del reloj solo afecta a la criba y no se toca.
- **Relanzado y cierre.** Enfriada a 59 °C a las 21:49, la etapa C se relanzó (reinicio 1 de 2) y
  terminó a las 00:52 del 2026-10-07 con salida 0: 4 lanzamientos en total (A, B, C y C
  relanzada), 1 reintento (`C.Ds.lr0.0006.b0.95.s17`). La corrida repetida reproduce las 29
  evaluaciones de la cortada (hasta el paso 14 500) a cuatro decimales, que es lo que imprime el
  log: Ds es determinista también bajo ★. Ningún otro corte, 0 NaN, 17/17 checkpoints guardados,
  stderr vacío en los cuatro lanzamientos. Térmica de la cadena: 73,4 °C de media, 84 °C de
  máximo, freno activo (pausa > 0) en el 97 % de las lecturas, 344 avisos ≥ 78 °C, 5 lecturas a
  83 °C y un solo disparo. **Los tiempos no valen para comparar brazos.**
- El sello v1 siguió vigente durante toda la ejecución; no se tocó ningún fichero sellado.

**Observaciones que no son desviaciones** (el preregistro las preveía): la reanudación de los
brazos factorizados en el smoke no fue idéntica (no exigible); `A.F.lr0.0012` degeneró (3,61 BPC):
cuenta como +∞ en la calibración, sin exclusión ni repesca; el LR de F quedó en el borde de la
rejilla (3e-4) y se declara sin más corridas.
