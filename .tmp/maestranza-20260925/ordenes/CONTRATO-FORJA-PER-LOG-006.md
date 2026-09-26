# Contrato forja (R2): guard ruidoso de migración pendiente — PER-LOG-006

## Objetivo observable
Un `GPT` de `mztrain.elastic_shape` con migración de optimizer pendiente (tras `widen_gpt` o `deepen_gpt` directos, antes de `migrate_optimizer`) debe **fallar ruidosamente en `forward`** con `RuntimeError` que nombre el remedio, en vez de dejar entrenar en silencio con un optimizer que ya no referencia los parámetros vivos (hoy: 20/20 parámetros nunca se actualizan tras `widen_gpt`; 16/16 del bloque nuevo tras `deepen_gpt`).

## Decisión de diseño (tomada por el director; no reabrir)
- Guard en `GPT.forward` (clase de referencia del propio módulo): al inicio,
  `if getattr(self, "_mzshape_pending_recs", False) or getattr(self, "_mzshape_new_param_sources", None): raise RuntimeError("migracion de optimizer pendiente: llama migrate_optimizer (o usa apply_event) antes de forward/entrenar")`.
- NO usar hooks (`register_forward_pre_hook`): `_forward_pre_hooks` es un dict compartido que la transacción de rollback no restaura.
- `apply_event` no se ve afectado: su sonda (`_probe_logits`) corre DESPUÉS de `migrate_optimizer`, que limpia ambos atributos. Verifícalo leyendo el orden en `apply_event`; si no fuera así, detente y reporta `bloqueado`.
- `rel_drift` (función pública legado) sobre un modelo pendiente pasará a lanzar el mismo `RuntimeError`: aceptado; documentarlo en su docstring (una línea).

## Archivos permitidos
- `D:/mztrain/src/mztrain/elastic_shape.py` (solo `GPT.forward`, docstring de `widen_gpt`, `deepen_gpt` y `rel_drift`).
- `D:/mztrain/tests/test_elastic_shape_safety.py` (nuevo test `test_forward_falla_con_migracion_pendiente`: tras `widen_gpt` → `forward` lanza `RuntimeError`; tras `deepen_gpt` → igual; tras `migrate_optimizer` → forward funciona; `apply_event` completo con `probe_x` → `accepted=True` sin error).
- `D:/mztrain/docs/ELASTICSHAPE-SAFETY.md`: añadir en la sección "Rollback y manejo de fallos" (o donde encaje) UN párrafo: las primitivas dejan el modelo en estado "migración pendiente" y `forward` falla con `RuntimeError` hasta `migrate_optimizer`; además documentar (a) que `LrWarmup` libera la base al terminar y por tanto respeta cambios externos del LR entre growths, pero un scheduler externo que escriba `lr` en cada paso compite con un warmup activo; (b) `_require_adamw` exige exactamente `torch.optim.AdamW` (no subclases); (c) `widen_embedding` con tabla de norma cero produce columnas nuevas cero; (d) el piso de KL en bf16 (~3e-5 nats) por cuantización de logits; (e) `dense_to_factorized` en bf16/fp16 mantiene S en ese dtype y avisa con `RuntimeWarning`. Prosa en español sin tildes como el resto del documento.
- Prohibido todo lo demás. Sin git que modifique, sin pip, sin red.

## Baseline
Banco verde de la pieza anterior: `D:/mztrain/.tmp/maestranza-20260925/banco_elastic_final.json` (ruta exacta en la conversación). El árbol tiene cambios preexistentes sin commitear (revisión 2026-09-07 + reparaciones de maestranza): preservarlos; registrar `git status --short` antes y después.

## Criterio de aceptación (ejecutable)
1. Test nuevo rojo antes del cambio (ejecuta primero solo el test con `D:/mztrain/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -o addopts="" D:/mztrain/tests/test_elastic_shape_safety.py -k migracion_pendiente` → debe fallar), verde después.
2. Banco completo verde:
```
cd D:/mztrain/.tmp/maestranza-20260925 && D:/mztrain/.venv/Scripts/python.exe sync_staging.py && PYTHONIOENCODING=utf-8 D:/mztrain/.venv/Scripts/python.exe "C:/Users/Raul/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/cf2f2be4-9257-4b47-81a2-ad2de01ada51/104b01cf-234c-409d-8bb0-6e8e706b4583/skills/maestranza/scripts/banco.py" D:/mztrain/.tmp/maestranza-20260925/staging --sin-venv --timeout 900 --ambito "src/mztrain/shape_ops.py,src/mztrain/elastic_shape.py,tests/test_shape_ops.py,tests/test_elastic_shape.py,tests/test_elastic_shape_safety.py" --json D:/mztrain/.tmp/maestranza-20260925/banco_forja_1.json
```
3. `D:/mztrain/.venv/Scripts/python.exe -m ruff check --select E9,F,B,PLE D:/mztrain/src/mztrain/elastic_shape.py D:/mztrain/tests/test_elastic_shape_safety.py` limpio.

## Condición de parada
Dos intentos del mismo enfoque como máximo; al tercero, `parcial` con diagnóstico. Reporte JSON del implementador (schema_version 1) en `D:/mztrain/.tmp/maestranza-20260925/reporte_FORJA-006.json`.
