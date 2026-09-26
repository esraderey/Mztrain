# Orden de trabajo: pieza `elastic_shape.py` (T2, 6 ciclos)

## Pieza y ámbito
- Árbol REAL (donde editas): `D:/mztrain`. Archivos permitidos: `D:/mztrain/src/mztrain/elastic_shape.py` y `D:/mztrain/tests/test_elastic_shape_safety.py`. **Nada más.** `shape_ops.py` ya fue reparado por otra pieza (lee su versión actual del árbol real: expone `NoiseError(ValueError)`); no lo toques. Sin docs, sin pyproject, sin git.
- Línea base: el banco de la pieza anterior en VERDE (`D:/mztrain/.tmp/maestranza-20260925/banco_shape_final.json`, ruta exacta en la conversación). El gate `--cov-fail-under=85` es deuda preexistente excluida en el staging (aceptado por el jefe). Modo `--sin-venv` aceptado por escrito.
- Un rojo fuera de tu ámbito → `fuera_de_ambito`, se devuelve al jefe.

## Banco (comando exacto de cada corrida, N = número de ciclo)
```
cd D:/mztrain/.tmp/maestranza-20260925 && D:/mztrain/.venv/Scripts/python.exe sync_staging.py && PYTHONIOENCODING=utf-8 D:/mztrain/.venv/Scripts/python.exe "C:/Users/Raul/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/cf2f2be4-9257-4b47-81a2-ad2de01ada51/104b01cf-234c-409d-8bb0-6e8e706b4583/skills/maestranza/scripts/banco.py" D:/mztrain/.tmp/maestranza-20260925/staging --sin-venv --timeout 900 --ambito "src/mztrain/shape_ops.py,src/mztrain/elastic_shape.py,tests/test_shape_ops.py,tests/test_elastic_shape.py,tests/test_elastic_shape_safety.py" --json D:/mztrain/.tmp/maestranza-20260925/banco_elastic_N.json
```
Edita SIEMPRE el árbol real; verifica SIEMPRE con sync + banco nuevo. Reporte: `json.load(open(ruta, encoding="utf-8"))`. Exploración puntual: `D:/mztrain/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -o addopts="" D:/mztrain/tests/test_elastic_shape_safety.py -k nombre` (no es evidencia).

## Hallazgos a reparar (arreglo mínimo prescrito; diseño ya decidido)
PoCs en `D:/mztrain/.tmp/peritaje-elasticshape-20260925/peritos/<PERITO>/poc_<ID>.py`. Conviértelos en anclas `test_ancla_<id>` en `tests/test_elastic_shape_safety.py` (usa los helpers/fixtures ya presentes en ese archivo). Anclas primero (banco rojo), arreglo, re-banco.

1. **PER-LOG-001 (ALTO)** `LrWarmup` (l.646-676): en `step()`, cuando `self.t` alcanza `self.steps` (warmup terminado), tras aplicar `s=1.0`, hacer `g.pop("_mzshape_base_lr", None)` en cada grupo. Así el caso G4-B (warmup activo sin terminar → base persistente) se conserva y el caso "warmup terminado + LR cambiado externamente" toma el LR vigente como base. Ancla: el PoC `PER-LOG/poc_PER-LOG-001.py` (LR bajado a 1e-4 tras un warmup completo → el siguiente warmup, tras un `apply_event`, debe terminar en 1e-4, no en 1e-3) + control (sin cambio externo termina en el LR base) + el caso G4-B existente debe seguir verde (`test_warmup_base_survives_optimizer_rebuild`).
2. **PER-LOG-002** `apply_event`: si `lr is not None or weight_decay is not None` y NO hubo rebuild del optimizer en el evento (ni conversión con `n_conv > 0`, ni `recs`, ni `ev.add_layers`), entonces `opt = migrate_optimizer(model, opt, [], lr, weight_decay)` y `report["optimizer"].append("overrides lr/weight_decay aplicados sin cirugia")`. Ancla: PoC `PER-LOG/poc_PER-LOG-002.py` (factorize redundante y `new_d == d` con `lr=5e-4, weight_decay=0.05` → grupos con esos valores; `original_opt` intacto).
3. **PER-LOG-005** `migrate_optimizer` (l.411-413): si `recs` no está vacío y NO hay ledger pendiente (`not pending`) → `raise ValueError("ledger ya consumido o no emitido por widen_gpt: nada que migrar")`. `recs == []` sigue permitido (deepen). Ancla: PoC `PER-LOG/poc_PER-LOG-005.py`.
4. **PER-LOG-003** `apply_event`: dentro del `try`, envolver la llamada `widen_gpt(...)` en `try/except NoiseError as e: raise _GrowthRejected(f"ruido no representable: {e}") from e` (importa `NoiseError` de `.shape_ops`). Resultado: `accepted=False, rolled_back=True`, optimizer original. Ancla: PoC `PER-LOG/poc_PER-LOG-003.py`.
5. **PER-LOG-004** `apply_event` (l.565-566): tras validar la forma de `probe_targets`, validar también `probe_targets.dtype` entero (`torch.int64/int32/int16/int8/uint8`) y valores en `{-100} ∪ [0, model.vocab)`; si no → `ValueError("probe_targets debe ser entero con valores en [0, vocab) o -100")` ANTES de la transacción. Ancla: PoC `PER-LOG/poc_PER-LOG-004.py` (fuera de rango y dtype float → ValueError sin que `model.d` ni la topología cambien).
6. **PER-API-006 / PER-LOG-007** `LrWarmup.__init__`: validar `math.isfinite(floor) and 0.0 <= floor <= 1.0` → `ValueError("floor debe estar en [0, 1]")`. Anclas: PoCs `PER-API/poc_PER-API-006.py` y `PER-LOG/poc_PER-LOG-007.py`.
7. **PER-API-010** `apply_event` (l.573) y `widen_gpt` (l.226): antes de `math.isfinite(noise_scale)`, comprobar `isinstance(noise_scale, (int, float)) and not isinstance(noise_scale, bool)`; si no → el mismo `ValueError("noise_scale debe ser finito y >= 0")`. Ancla: PoC `PER-API/poc_PER-API-010.py`.
8. **PER-MAT-001** `widen_gpt` (l.277): `scale_output_rows(blk.qkv, omap_qkv[:d], math.sqrt(hd2 / hd))` — escalar SOLO las filas q viejas (las nuevas son ruido; la exactitud SDPA no las necesita y así el presupuesto `||dW||=ns·||W||` de qkv se cumple). `_rescale_q_state` ya usa exactamente esas filas. Actualiza el comentario. Ancla: PoC `PER-MAT/poc_PER-MAT-001.py` (ratio realizado/ns en qkv tras `widen_gpt` con noise 1e-2 y growth 4× debe ser 1.0 ± 1e-6 en fp64) + los tests existentes de exactitud SDPA deben seguir verdes.
9. **PER-MAT-006** `_probe_metrics` (l.525): corregir el comentario: "FP64 evita que el redondeo del CALCULO de KL/loss domine; si los logits ya son bf16, su cuantizacion fija un piso de ~3e-5 nats en la KL". Sin ancla.

## Restricciones
- Vetos absolutos del obrero. No cambiar firmas públicas ni semántica documentada más allá de lo prescrito. No tocar `_shape_transaction`, `_new_adamw` ni `_rescale_q_state` salvo lo indicado.
- Estilo del módulo (español sin tildes, ≤ 120 columnas). Sin red, sin pip, sin git que modifique.
- Presupuesto: 6 ciclos de banco. Sin verde → RECHAZADO con diagnóstico.

## Reporte final
JSON del obrero en `D:/mztrain/.tmp/maestranza-20260925/reporte_PER-ELASTIC.json` (pieza, veredicto, ciclos_usados, banco_final, fallos[...], diff_resumen verificado con `git diff --stat -- src/mztrain/elastic_shape.py tests/test_elastic_shape_safety.py`, deuda, escalacion). Responde con la ruta y una tabla hallazgo → ancla → estado.
