# Contrato forja (R2): residuos del peritaje ElasticShape — 2026-09-26

## Objetivo observable
Cerrar los cinco residuos reproducidos por el director (`../residuos/poc_residuos.py`, 7/7 reproducen):
R1 export a denso ≠ forward con `wake_gate` parcial; R2 `grow_rank(preserve_weights=False)` reaplica gates
viejos a una base SVD nueva; R3/R3b `LrWarmup` vuelve a una base caduca si el warmup se abandona o se reanuda
de checkpoint y el LR cambio fuera; R4 el `eps` reescalado de LayerNorm no persiste al reconstruir el GPT
desde su configuracion; R5/R5b `dense_to_factorized` deja S en bf16 porque el forward no tolera dtypes mixtos.

## Decisiones de diseno (tomadas por el director; no reabrir)

### D1 — `src/mztrain/layers.py` (nucleo; cambios minimos, semantica identica cuando gate=1 y dtypes iguales)
1. `ZFactorizedLinear.reconstruct_weight`: devolver la **W efectiva del forward**:
   `s = self._gated_s(); dt = torch.promote_types(self.U.dtype, s.dtype); W = (self.U.to(dt) * s.to(dt).unsqueeze(0)) @ self.V.to(dt); return W.to(self.U.dtype)`.
   Docstring: "W efectiva = U diag(S*wake_gate) V (la misma funcion que forward)". Con gate=1 y dtype unico
   es bit a bit lo de antes (los `.to` son no-op).
2. `ZSparseFactorizedLinear.reconstruct_weight`: mismo cambio en la parte low-rank (`_gated_s()`, promocion).
3. `ZFactorizedLinear.grow_rank` rama `preserve_weights=False` (re-SVD via `_init_from_weight`): la base cambia,
   asi que tras `self.rank = new_rank` NO llamar `_resync_elastic_buffers` sino resetear:
   `self.wake_gate = torch.ones(new_rank, device=device); self.sleep_mask = torch.zeros(new_rank, dtype=torch.bool, device=device)`.
   La rama `preserve_weights=True` sigue usando `_resync_elastic_buffers` (indices preservados). Esto cubre
   tambien `ZSparseFactorizedLinear.grow_rank` (llama a super con `preserve_weights=False`).
4. `ZFactorizedLinear.forward` (rama estandar) y `ZSparseFactorizedLinear.forward`: tras `h = h * s_eff...`,
   `if h.dtype != self.U.dtype and not torch.is_autocast_enabled(h.device.type): h = h.to(self.U.dtype)`
   antes del segundo `F.linear`. Bajo autocast no cambia nada (autocast ya castea); con dtype unico es no-op;
   solo actua en el caso hoy imposible (U/V bf16 con S fp32 sin autocast).
   NO tocar la rama fp8, `_gated_s`, `elastic_replace_factors` ni `_resync_elastic_buffers`.

### D2 — `src/mztrain/shape_ops.py::dense_to_factorized`
- S se conserva en fp32 cuando `dt in (torch.float16, torch.bfloat16)` (`new.S.data = S_svd.to(torch.float32)`,
  reasignando `.data` como hace `widen_factorized_linear`); en fp32/fp64 S sigue en `dt`.
- El `RuntimeWarning` se mantiene con texto actualizado: "U/V quedan en {dt} (S en fp32): la reconstruccion
  tiene error ~1e-3-1e-2; la ruta recomendada es parametros fp32". Docstring acorde.
- `_reconstruction_error` se sigue calculando con `reconstruct_weight()` (ya tolera dtypes mixtos por D1).

### D3 — `src/mztrain/elastic_shape.py::LrWarmup` — regla "la escritura externa gana"
- `_apply(s)`: ademas de `g["lr"] = b * s`, guardar `g["_mzshape_warmup_lr"] = g["lr"]` (ultimo LR escrito
  por un warmup).
- `__init__` (tras validar `steps` y `floor`): por grupo,
  `if "_mzshape_base_lr" in g and g.get("_mzshape_warmup_lr") == g["lr"]: base = g["_mzshape_base_lr"]`
  (warmup activo sin que nadie tocara el LR: caso G4-B) `else: base = g["lr"]` (sin warmup activo, o alguien
  escribio el LR despues: gana la escritura externa); `g["_mzshape_base_lr"] = base`. Sustituye al `setdefault`.
- `step()`: al llegar a `done`, `pop` de AMBAS claves (`_mzshape_base_lr`, `_mzshape_warmup_lr`).
- `_new_adamw` no cambia (deep-copia las claves del grupo; con `lr=` override el LR ya no coincide con
  `_mzshape_warmup_lr` y por tanto el siguiente warmup toma el override como base: correcto).
- Documentar en el docstring la coincidencia benigna: si una escritura externa deja exactamente el mismo valor
  que escribio el warmup, se trata como no tocada.

### D4 — `src/mztrain/elastic_shape.py` — `eps` de LayerNorm como parte de la configuracion del GPT
- `Block.__init__(self, d, heads, lin, eps: float = 1e-5)` → `nn.LayerNorm(d, eps=eps)` en ln1/ln2.
- `GPT.__init__(..., lin, ln_eps: float = 1e-5)` → `self.ln_eps = ln_eps`; bloques con `eps=ln_eps`;
  `self.lnf = nn.LayerNorm(d, eps=ln_eps)`.
- `widen_gpt`: tras reemplazar `model.lnf`, `model.ln_eps = model.lnf.eps` (la transaccion lo restaura en
  rollback porque vive en `__dict__`).
- `deepen_gpt`: construir `Block(model.d, model.heads, lin, eps=first.ln1.eps)` y quitar el bucle que copiaba
  `eps` a mano si queda redundante.
- Firmas: `ln_eps`/`eps` son parametros nuevos con default identico al comportamiento actual (compatibles).

### Documentacion — `docs/ELASTICSHAPE-SAFETY.md`
Actualizar (a) del bloque "Ademas:" con la regla "la escritura externa gana" (y la coincidencia benigna);
en "Nuevo contrato del ruido" sustituir la frase sobre `eps` y `state_dict` por: reconstruir con
`GPT(..., ln_eps=model.ln_eps)`; en "Optimizador y precision" indicar que `dense_to_factorized` conserva S en
fp32 tambien en bf16/fp16 y que `reconstruct_weight` devuelve la W efectiva (incluye `wake_gate`), lo que
hace exacta la exportacion a denso durante un mini-warmup de ElasticRank. Espanol sin tildes.

## Archivos permitidos
`src/mztrain/layers.py` (solo las zonas de D1), `src/mztrain/shape_ops.py` (solo `dense_to_factorized`),
`src/mztrain/elastic_shape.py` (solo `Block.__init__`, `GPT.__init__`, `widen_gpt`, `deepen_gpt`, `LrWarmup`),
`tests/test_layers.py`, `tests/test_shape_ops.py` (solo `test_ancla_PER_MAT_003`: anadir la asercion
`S.dtype == torch.float32` para bf16; el warning sigue), `tests/test_elastic_shape_safety.py`,
`docs/ELASTICSHAPE-SAFETY.md`. Prohibido todo lo demas (incluidos `refactorize.py`, `engine.py`,
`elastic_rank.py`, `CHANGELOG.md`). Sin git que modifique, sin pip, sin red.

## Tests ancla (rojos antes, verdes despues; nombres `test_residuo_R*`)
- `tests/test_layers.py`: R1 `reconstruct_weight` con `wake_gate=[1,.5,0]` reproduce `forward` (fp64, atol 1e-12)
  y con gate=1 es identico a `(U*S)@V`; R2 tras `grow_rank(5, preserve_weights=False)` con gates parciales,
  `wake_gate` es todo 1 y `sleep_mask` todo False, y con `preserve_weights=True` el prefijo se conserva
  (comportamiento actual); R5b forward con U/V bf16 y S fp32 sin autocast funciona y coincide con el forward
  fp32 de referencia a tolerancia bf16 (rel < 5e-2); `reconstruct_weight` mixto devuelve dtype de U.
- `tests/test_shape_ops.py`: R5 `dense_to_factorized(Linear bf16)` → `S.dtype == float32`, warning presente,
  forward de la capa coincide con la Linear bf16 a rel < 1e-2 (ajustar `test_ancla_PER_MAT_003`).
- `tests/test_elastic_shape_safety.py`: R3 (warmup abandonado a 3/10 pasos, LR externo 1e-5, `apply_event`,
  nuevo `LrWarmup(2)` → LR final 1e-5), R3b (mismo con `state_dict`/`load_state_dict` en medio), G4-B sigue:
  warmup activo SIN cambio externo + `apply_event` + nuevo warmup → LR final = base original (el test
  `test_warmup_base_survives_optimizer_rebuild` y `test_ancla_per_log_001_*` deben seguir verdes); R4
  `widen_gpt` 8→16 en fp64, `GPT(..., ln_eps=m.ln_eps)` + `load_state_dict` → logits identicos (atol 1e-12) y
  `model.ln_eps == model.lnf.eps == blocks[i].ln1.eps`; rollback de `widen_gpt` con fallo inyectado restaura
  `model.ln_eps`.
Ejecuta primero los anclas solos (`D:/mztrain/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -o addopts="" <archivo> -k residuo`) → deben fallar; tras implementar → verdes.

## Baseline y criterio de aceptacion
- Baseline: `D:/mztrain/.tmp/maestranza-20260925/banco_final2_jefe.json` VERDE, hash `ced9a2541f6c7423`. Arbol con
  cambios preexistentes sin commitear: preservarlos; `git -C D:/mztrain status --short` antes y despues.
- Banco completo VERDE (suite entera del repo: incluye test_layers, test_elastic_rank, test_engine,
  test_refactorize, test_zcodebert — cualquier rojo ahi es regresion NUEVA y bloquea):
```
cd D:/mztrain/.tmp/maestranza-20260925 && D:/mztrain/.venv/Scripts/python.exe sync_staging.py && PYTHONIOENCODING=utf-8 D:/mztrain/.venv/Scripts/python.exe "C:/Users/Raul/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/cf2f2be4-9257-4b47-81a2-ad2de01ada51/104b01cf-234c-409d-8bb0-6e8e706b4583/skills/maestranza/scripts/banco.py" D:/mztrain/.tmp/maestranza-20260925/staging --sin-venv --timeout 900 --ambito "src/mztrain/layers.py,src/mztrain/shape_ops.py,src/mztrain/elastic_shape.py,tests/test_layers.py,tests/test_shape_ops.py,tests/test_elastic_shape.py,tests/test_elastic_shape_safety.py" --json D:/mztrain/.tmp/maestranza-20260925/banco_residuos_N.json
```
- `D:/mztrain/.venv/Scripts/python.exe -m ruff check --select E9,F,B,PLE` sobre los archivos tocados: limpio.
- El PoC del director `D:/mztrain/.tmp/maestranza-20260925/residuos/poc_residuos.py` debe imprimir `0 / 7 reproducen`
  (ejecutalo al final; su salida literal va al reporte).

## Condicion de parada
Dos intentos del mismo enfoque como maximo; al tercero, `parcial` con diagnostico. Si D1.4 rompe algun test
de `test_layers`/`test_elastic_rank`/`test_engine`, detente y reporta `bloqueado` con el test y la causa: no
"arregles" tests ajenos. Reporte JSON del implementador (schema_version 1) en
`D:/mztrain/.tmp/maestranza-20260925/reporte_FORJA-RESIDUOS.json`, con `hash_pieza` del banco final en el
`resumen` del comando del banco y el comando de los anclas en rojo con `fallo_esperado`.
