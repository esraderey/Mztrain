# Orden de trabajo: pieza `shape_ops.py` (T1, 3 ciclos)

## Pieza y ámbito
- Árbol REAL (donde editas): `D:/mztrain`. Archivos permitidos: `D:/mztrain/src/mztrain/shape_ops.py` y `D:/mztrain/tests/test_shape_ops.py`. **Nada más.** No tocar `elastic_shape.py`, otros tests, docs, pyproject, ni nada de git.
- Línea base: banco VERDE (`D:/mztrain/.tmp/maestranza-20260925/banco_baseline.json`, hash `f6fbad8b4fae8aee`, suite completa). El gate `--cov-fail-under=85` del pyproject es deuda preexistente (81.7 %) y está excluido en el staging por `pytest.ini`; aceptado por escrito por el jefe. Modo degradado `--sin-venv` aceptado por escrito (torch 2.11 cu128 del venv del proyecto; instalarlo en un venv nuevo no es viable).
- Un rojo fuera de tu ámbito se reporta como `fuera_de_ambito` y se devuelve.

## Banco (comando exacto de cada corrida, N = número de ciclo)
```
cd D:/mztrain/.tmp/maestranza-20260925 && D:/mztrain/.venv/Scripts/python.exe sync_staging.py && PYTHONIOENCODING=utf-8 D:/mztrain/.venv/Scripts/python.exe "C:/Users/Raul/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/cf2f2be4-9257-4b47-81a2-ad2de01ada51/104b01cf-234c-409d-8bb0-6e8e706b4583/skills/maestranza/scripts/banco.py" D:/mztrain/.tmp/maestranza-20260925/staging --sin-venv --timeout 900 --ambito "src/mztrain/shape_ops.py,src/mztrain/elastic_shape.py,tests/test_shape_ops.py,tests/test_elastic_shape.py,tests/test_elastic_shape_safety.py" --json D:/mztrain/.tmp/maestranza-20260925/banco_shape_N.json
```
`sync_staging.py` regenera la copia desde el árbol real: edita SIEMPRE el árbol real, verifica SIEMPRE con sync + banco nuevo. El reporte se lee con `json.load(open(ruta, encoding="utf-8"))`. Para reproducir un test suelto sin el banco puedes usar `D:/mztrain/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -o addopts="" D:/mztrain/tests/test_shape_ops.py -k nombre` — solo para explorar; la evidencia es el JSON del banco.

## Hallazgos a reparar (arreglo mínimo prescrito; la decisión de diseño ya está tomada)
Los PoC del peritaje están en `D:/mztrain/.tmp/peritaje-elasticshape-20260925/peritos/<PERITO>/poc_<ID>.py`; conviértelos en anclas (tests pytest cortos en `tests/test_shape_ops.py`, nombre `test_ancla_<id>`), que deben FALLAR antes del arreglo y pasar después. Añade todas las anclas, corre el banco (rojo esperado), arregla, re-banco.

1. **PER-API-001** `_validate_map` (l.29): rechazar mapas no enteros ANTES del cast. Tensores: `if torch.is_tensor(m) and not m.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8): raise ValueError(f"{name}: indices deben ser enteros")`; secuencias Python: rechazar si algún elemento es `bool` o no es `int` (usa `numbers.Integral` y excluye `bool`). PoC: `PER-API/poc_PER-API-001.py`.
2. **PER-API-002** `scale_output_rows` (l.246): exigir `factor > 0` además de finito (`ValueError("factor debe ser finito y > 0")`). PoC: `PER-API/poc_PER-API-002.py`.
3. **PER-API-004 / 005 / 009** (validación de tipos): `pad_state_tensor`: `new_shape` debe ser secuencia de `int` (no tensor) → `ValueError`; `pad_adam_entry`: si falta `"step"` → `ValueError("entrada AdamW sin 'step'")` en vez de `KeyError`; `widen_layernorm`/`widen_embedding`/`widen_factorized_linear`: `new_dim`/`new_in`/`new_out` deben ser `int` (no bool, no float) → `ValueError`. PoCs: `PER-API/poc_PER-API-004.py`, `-005`, `-009`.
4. **PER-API-011** `pad_adam_entry` (l.343-347): sustituir la heurística por forma por la lista explícita `("exp_avg", "exp_avg_sq", "max_exp_avg_sq")`: solo esas claves se paddean; el resto se deep-copia. PoC: `PER-API/poc_PER-API-011.py`.
5. **PER-MAT-002** `_functional_noise` (l.85-87): tras `noise = (noise * (target / size)).to(dt)`, si `target > 0` y `noise.norm() == 0` (el cast al dtype aplastó todo el ruido) → `raise NoiseError("ruido funcional colapsa a cero en el dtype de U")`. Define `class NoiseError(ValueError)` a nivel de módulo (pública, sin guion bajo) y úsala TAMBIÉN en el `raise` existente de la l.87 ("no representable"). Motivo: `elastic_shape.apply_event` la convertirá en rechazo recuperable en la siguiente pieza. PoC: `PER-MAT/poc_PER-MAT-002.py` (caso fp16 gauge 1e-3, ns 1e-5 → hoy ratio 0.0 sin aviso).
6. **PER-MAT-003** `dense_to_factorized` (l.266-271): NO cambiar el dtype de S (con U/V bf16 y S fp32 el forward sin autocast falla: `expected m1 and m2 to have the same dtype`, verificado por el jefe). Arreglo: si `dt in (torch.float16, torch.bfloat16)`, emitir `warnings.warn("dense_to_factorized en baja precision: S queda en <dt> (no fp32) y la reconstruccion tiene error ~1e-3-1e-2; la ruta recomendada es parametros fp32", RuntimeWarning, stacklevel=2)` y ampliar el docstring con esa nota. Ancla: `pytest.warns(RuntimeWarning)` con una Linear bf16, y ausencia de warning con fp32 (`warnings.catch_warnings(record=True)`).
7. **PER-MAT-004** `widen_layernorm` (l.225): cuando `variance_compensation=True`, construir el LN nuevo con `eps=ln.eps * old_dim / new_dim` (cancela exactamente el término de eps; derivación: LN([x,0]) con γ·√(d/d') = γ·x/√(σ²+eps·d'/d)). Con `variance_compensation=False` no cambia nada. Docstring: una línea. PoC: `PER-MAT/poc_PER-MAT-004.py` (caso σ²=eps: error 0.18 → ~1e-16).
8. **PER-API-003** `widen_embedding`: solo docstring, una frase: "Con ||W||_F = 0 las columnas nuevas quedan en cero (calibración relativa)". Sin ancla.

## Restricciones
- Vetos absolutos del obrero (no relajar asserts, no skip/xfail, no `noqa` sin justificación, no cambiar firmas públicas ni semántica documentada más allá de lo prescrito).
- Estilo del módulo: docstrings en español sin tildes como el resto del archivo, líneas ≤ 120.
- No ejecutes nada fuera del banco/pytest/python del venv; sin red; sin `pip`; sin comandos git que modifiquen (solo `git diff`/`git status` para inspeccionar).
- Presupuesto: 3 ciclos de banco. Si no llegas a verde, RECHAZADO con diagnóstico.

## Reporte final
JSON del obrero (pieza, veredicto, ciclos_usados, banco_final, fallos[{paso, causa_raiz, ancla, arreglo, lineas_tocadas}], diff_resumen verificado con `git diff --stat -- src/mztrain/shape_ops.py tests/test_shape_ops.py`, deuda, escalacion) escrito en `D:/mztrain/.tmp/maestranza-20260925/reporte_PER-SHAPE.json`. Responde con la ruta y una tabla hallazgo → ancla → estado.
