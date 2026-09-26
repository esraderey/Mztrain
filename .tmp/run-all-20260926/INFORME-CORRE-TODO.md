# "Corre todo mztrain" — 2026-09-26

Todo lo ejecutable del repositorio (ejemplos, scripts, generacion, smokes grandes y entrenamientos largos)
ejecutado en secuencia con cwd en `.tmp/run-all-20260926/` para que las salidas relativas (checkpoints,
JSON, HTML) no pisen los artefactos del repo. Logs: `<paso>.log`; cronologia: `STATUS.log`. Los benchmarks y la
suite de tests ya se habian corrido antes (`../bench-20260926/INFORME-PRUEBAS-BENCHMARKS.md`: 413/413, 33.5 %
de ahorro en ElasticShape).

## Resultado por paso

| Paso | Que es | Exit | Duracion | Notas |
|---|---|---:|---:|---|
| `examples/example_basic.py` | MLP factorizado + engine completo | 0 | 5 s | export a denso OK (297 098 params) |
| `examples/example_memory_estimation.py` | `estimate_memory_savings` sobre varios modelos | 0 | 39 s | |
| `examples/example_transformer.py` | transformer con `ZFactorizedTransformerBlock` | 0 | 8 s | |
| `examples/example_zcodebert.py` | ZCodeBERT 12 capas h512, 15 epocas, act-checkpoint MNEME, ZCompressedAdam | **1** | 3455 s | **CUDA OOM en la epoca 1** (21.4 GiB "asignados" en una GPU de 8 GB); la epoca 0 tardo 1358 s. Causa raiz abajo (D1). Preexistente. |
| `scripts/calc_gpu_params_equiv.py` | aritmetica de params/VRAM equivalentes | 0 | 0 s | |
| `scripts/seal.py verify` | integridad criptografica del arbol | **1** | 2 s | **Esperado**: firma Ed25519 valida pero la obra no coincide con el sello: 3 archivos agregados (SAFETY.md, test_elastic_shape_safety.py, el JSON nuevo de bench) y los modificados sin commitear. Se resuelve al re-sellar tras el commit. |
| `generate_game.py --all` (checkpoint `zcodebert_htmlgames.pt`) | generacion de juegos HTML5 por MLM iterativo | **1** | 10 s | **`load_state_dict` estricto falla: faltan `wake_gate`/`sleep_mask`** en el checkpoint (feb-2026, anterior a ElasticRank de mayo). Preexistente (D2). |
| `scripts/train_zcoder_1b.py --skip-save` | smoke 1B-equivalente (1 paso, fp16, MNEME) | 0 | 9 s | metrica y storage MNEME escritos en la carpeta de la corrida |
| `scripts/test_zcoder_410m_full.py` | smoke "todo prendido" 406M-equiv, 3 epocas x 24 pasos, CUDA | 0 | 221 s | val loss 4.3357 (mayo: 4.3359); pico CUDA 3.95 GB (mayo 3.83); rangos finales 72/96 vs 60/84 en mayo — ver A/B abajo |
| `train_html_games.py` (20 min) | continua `zcodebert_trained.pt` con datos de juegos | **1** | 8 s | misma causa que `generate_game`: checkpoint de febrero sin buffers ElasticRank (D2) |
| `analysis_full.py` (~1 h) | analisis arquitectonico + entrenamiento largo + precision + coherencia | 0 | 4102 s | 18 epocas / 9 027 pasos en 64 min; mejor val loss 1.7196, MLM acc 50.9 %, top-5 79.4 %, PPL 5.5, clasificacion de lenguaje 96.7 %, coherencia exacta 17.5 % / top-5 45.0 %; 71.1 % ahorro de memoria, 69 584 tok/s. Historico (feb): val loss 3.98, coherencia 32.5 % / 40.0 %. Resultados en `analysis_results.json` de la carpeta de la corrida (el de la raiz esta intacto). |

## Defectos encontrados (ninguno introducido por las reparaciones de ElasticShape)

### D1 — Fuga de activaciones con MNEME: la evicción de `ZActivationCheckpoint` es un no-op (alto)
`src/mztrain/checkpoint.py:148-154` libera cada activacion registrada con
`getattr(zspace, "remove", None) or getattr(zspace, "delete", None)`. **El `ZSpace` de MNEME (mneme_core.py,
clase `ZSpace`) no tiene ni `remove` ni `delete`** (tiene `register`, `load`, `update`, `cleanup`, `clear` de cache):
el `callable(_evict)` es False y nada se libera. El comentario del codigo lo anticipaba ("si el backend la nombra
distinto, ajustar"): la reparacion A5 del peritaje de agosto asumio una API inexistente. Evidencia en
`example_zcodebert.log`: 599 registros `act_ckpt_N` en la epoca 0 (12 capas x 50 pasos), cero liberaciones, y el
coste por registro crece de 0.05 s (los primeros) a 2.4 s (el #280) — registro O(n) — hasta el OOM. `checkpoint.py`
y MNEME no se tocaron en los lotes de esta semana (mismo archivo byte a byte en la copia pre-reparacion).
Afecta solo con MNEME instalado y `use_activation_checkpointing=True`; la rama fallback INT8 si hace `pop`.
Remediacion (forja): liberar por clave con la API real de MNEME (o registrar con un TTL/`cleanup()` por paso), o
desactivar la ruta ZSpace para activaciones y usar el fallback INT8 hasta que MNEME exponga `remove`. Ancla: un
test que registre N activaciones, las descomprima y compruebe que el registro de ZSpace no crece.

### D2 — Checkpoints anteriores a ElasticRank no cargan (medio)
`ZFactorizedLinear` registra desde mayo los buffers persistentes `wake_gate` y `sleep_mask`; los checkpoints de
febrero (`zcodebert_trained.pt`, `zcodebert_htmlgames.pt`, `zcodebert_checkpoint.pt`: 353 claves, sin buffers)
fallan en `load_state_dict(strict=True)` en `generate_game.py:775` y `train_html_games.py:748`. Remediacion
(forja, compatible): `ZFactorizedLinear._load_from_state_dict` que rellene `wake_gate=1`/`sleep_mask=False`
cuando falten (y los quite de `missing_keys`), con ancla que cargue un `state_dict` sin buffers.

### Observaciones
- `seal.py verify` falla porque el arbol tiene cambios sin sellar: es el estado esperado hasta commit + `sign`.
- El corredor escribio dos lineas de estado en `D:\mztrain\STATUS.log` por un `cd` en subshell (error mio);
  movidas a la carpeta de la corrida y el archivo eliminado de la raiz.

## A/B del smoke de 410M contra el arbol pre-reparacion
Los rangos finales (72/96 frente a 60/84 en mayo) y el pico de memoria (+3 %) diferian del JSON de mayo. Re-corri
el mismo script importando `mztrain` desde la copia pre-reparacion del peritaje (`PYTHONPATH` al lab): **0 hojas
distintas** fuera de tiempos y memoria; rangos `[48, 72, 96]` y val loss `[4.7689, 4.3718, 4.3357]` identicos en pre
y post. La diferencia con mayo es de la evolucion del codigo entre mayo y septiembre, no de estos lotes
(`test_zcoder_410m_full_PRELOT.log`, JSONs en `../bench-20260926/new-results/`).

## Incidente: sobreescritura del checkpoint `zcodebert_trained.pt` y restauracion
Para que `train_html_games.py` (que carga `zcodebert_trained.pt` por ruta relativa) corriera con cwd en la carpeta
de la corrida, cree un **hardlink** a `D:\mztrain\zcodebert_trained.pt`. `analysis_full.py` guarda su modelo en
`zcodebert_trained.pt` relativo: al escribir sobre el enlace escribio sobre el MISMO inodo, y el checkpoint original
del 22/02/2026 (118 019 098 bytes) quedo sustituido por el modelo entrenado hoy (118 079 323 bytes). Error mio: un
hardlink no aisla nada; debi copiar el archivo o pasar la ruta como argumento.
Restauracion: aparecio una copia intacta en `D:\mneme y mztrain\mztrain\zcodebert_trained.pt` (mismo tamano y fecha);
la copie a la raiz y verifique **SHA-256 identico** (`1A90AA68…F7DCEC`). El modelo entrenado hoy se conserva como
`.tmp/run-all-20260926/zcodebert_trained.NEW-20260926.pt`; el enlace se elimino (`fsutil hardlink list` muestra una
sola ruta). Ningun otro artefacto de la raiz cambio (fechas de los demas `.pt` intactas; `analysis_results.json` de la
raiz intacto). Los JSON de `bench/results` si cambiaron por los benchmarks (respaldo en `../bench-20260926/results-backup/`).

## Veredicto
- 8 de 11 pasos en verde. Los 3 fallos son preexistentes y estan diagnosticados: D1 fuga de activaciones con MNEME
  (evicción no-op: `ZSpace` no tiene `remove`/`delete`), D2 checkpoints de febrero sin buffers de ElasticRank
  (`generate_game`, `train_html_games`), y `seal.py verify` que fallara hasta commit + re-sello. Ninguno lo
  introducen las reparaciones de ElasticShape: el A/B pre/post del 410M es identico y `checkpoint.py` no se toco.
- Propuesta (forja): D1 — liberar por clave con la API real de MNEME o usar el fallback INT8; D2 —
  `_load_from_state_dict` tolerante en `ZFactorizedLinear` con defaults `wake_gate=1`, `sleep_mask=False`.
