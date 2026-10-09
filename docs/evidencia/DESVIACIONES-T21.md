# T21 — Desviaciones e incidentes

Este fichero no entra en el sello y recoge cada corrida cortada, reintento, disparo térmico o
cambio posterior al sello (que exigiría un bloque nuevo con su nota). Antes del sello (v1,
2026-10-07 04:29) no hubo ningún incidente: nada se corrió en GPU.

## 1. Ejecución (2026-10-07, 04:53–07:46): sin desviaciones

- **Smoke** (sesión `smoke-20261007-045310`, 04:53–04:59, salida 0): los cuatro brazos nuevos con
  pérdida finita y los parámetros exactos de la cuenta (13 522 400, 13 507 200, 13 540 864 y
  13 566 720); reanudación idéntica (diff 0,0) en F640, F512 y F448 y a 7e-6 en Dc, como preveía
  el preregistro tras la revisión (ninguna exigible). Máximo 74 °C.
- **Cadena** (sesión `cadena-20261007-050032`, 05:00–07:46, salida 0 en las dos fases): etapa A
  con sus 15 corridas en un solo lanzamiento, 0 cortes, 0 reintentos, 0 NaN, 0 degeneradas,
  15/15 checkpoints, stderr vacío; etapa B vacía por las reglas (sin finalista). Térmica: 72,6 °C
  de media, 81 °C de máximo, freno activo en el 91 % de las lecturas, 117 avisos ≥ 78 °C,
  0 lecturas a 83 °C, 0 disparos. 152 min de pared en corridas. **Los tiempos no valen para
  comparar brazos.**
- El sello v1 siguió vigente durante toda la ejecución; no se tocó ningún fichero sellado. La
  sesión de Claude que vigilaba se reinició durante la cadena; el vigía y el harness, desacoplados,
  no se enteraron.
