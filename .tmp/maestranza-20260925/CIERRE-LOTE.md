# Cierre del lote: reparacion de los hallazgos del peritaje ElasticShape — 2026-09-25

Origen: `../peritaje-elasticshape-20260925/PERITAJE-elasticshape.md` (gate BLOCKED: 1 alto, 2 medio, 15 bajo).
Metodo: maestranza (arreglo minimo + ancla roja→verde, banco por pieza, verificacion independiente del
jefe) para 17 hallazgos; forja (decision de diseno + revisor ciego) para PER-LOG-006.

## Laboratorio
- Banco: `scripts/banco.py` de maestranza sobre un **staging** regenerado desde el arbol real por
  `sync_staging.py` (src/, tests/, scripts/, pyproject, SEAL.json; el repo completo pesa 2.3 GB en
  checkpoints ajenos al modulo). Modo degradado `--sin-venv` aceptado por escrito (torch 2.11 cu128 del
  venv del proyecto). `pytest.ini` del staging: `pythonpath = src` (verificado por smoke que se importa la
  copia, no el editable) y sin `--cov-fail-under=85` (deuda preexistente 81.7 %, ortogonal).
- Linea base: `banco_baseline.json` VERDE, hash `f6fbad8b4fae8aee` (suite completa del repo).

## Piezas

| Pieza | Talla | Veredicto | Ciclos | Banco del jefe | Hash jefe = hash reparador |
|---|---|---|---|---|---|
| `src/mztrain/shape_ops.py` (+ `tests/test_shape_ops.py`) | T1 | **REPARADO-2** | 2 | `banco_shape_jefe.json` verde | `6218700c3496394f` ✔ |
| `src/mztrain/elastic_shape.py` (+ `tests/test_elastic_shape_safety.py`) | T2 | **REPARADO-2** | 2 | `banco_elastic_jefe.json` verde | `5aa84b40251f29d9` ✔ |
| forja PER-LOG-006 (`GPT.forward`, docs) | R2 | **COMPLETA** (tras 1 bloqueo y cambio de diseno) | 2 intentos | `banco_final_jefe.json` verde | `ac8bab673e693d49` ✔ |

Control de salida del jefe en cada pieza: re-ejecucion del banco (hash identico), diff real contra la
copia pre-reparacion del peritaje (no contra HEAD: el arbol ya tenia la revision 2026-09-07 sin commitear),
inspeccion contra los vetos (ningun assert/test relajado o borrado; 0 lineas eliminadas en los tests; sin
skip/xfail/noqa), anclas verificadas rojas contra el codigo pre-reparacion (12/12 de elastic_shape las
ejecute yo contra `elastic_shape.pre.py`, porque el obrero las habia escrito junto con los arreglos).

## Hallazgo → arreglo → ancla

| ID | Sev. | Arreglo (archivo) | Ancla | PoC original contra el arbol reparado |
|---|---|---|---|---|
| PER-LOG-001 | alto | `LrWarmup.step` libera `_mzshape_base_lr` al terminar (G4-B intacto: warmup ACTIVO sigue reutilizando la base) | `test_ancla_per_log_001_*` | no reproduce |
| PER-LOG-002 | medio | `apply_event`: overrides lr/wd aplicados via `migrate_optimizer(model, opt, [], ...)` cuando no hubo rebuild; anotado en el reporte | `test_ancla_per_log_002_*` | no reproduce |
| PER-LOG-006 | medio | **forja**: `GPT.forward` emite `RuntimeWarning` si hay migracion pendiente Y `torch.is_grad_enabled()`; inferencia/sondas bajo `no_grad` no avisan; docstrings + SAFETY.md | `test_forward_avisa_con_migracion_pendiente` | reproduce el sintoma (los parametros siguen sin actualizarse) — **mitigado, no eliminado**: el fallo ya es ruidoso; ver decision abajo |
| PER-LOG-005 | bajo | `migrate_optimizer` rechaza `recs` sin ledger pendiente | `test_ancla_per_log_005_*` | no reproduce |
| PER-LOG-003 | bajo | `NoiseError` (nueva, `shape_ops`) convertida en `_GrowthRejected` en `apply_event` | `test_ancla_per_log_003_*` | no reproduce |
| PER-LOG-004 | bajo | `probe_targets`: dtype entero y valores en {-100} ∪ [0, vocab) antes de la transaccion | `test_ancla_per_log_004_*[2]` | no reproduce |
| PER-API-001 | bajo | `_validate_map` rechaza indices no enteros (tensor y secuencia) | `test_ancla_PER_API_001` | no reproduce |
| PER-API-002 | bajo | `scale_output_rows`: `factor > 0` | `test_ancla_PER_API_002` | ValueError (arreglado) |
| PER-API-006 / LOG-007 | bajo | `LrWarmup`: `0 <= floor <= 1` finito | `test_ancla_per_api_006_*`, `test_ancla_per_log_007_*[3]` | ValueError (arreglado) |
| PER-API-004/005/009 | bajo | `_require_int` en widen_*; `new_shape` secuencia de int; `'step'` → ValueError | `test_ancla_PER_API_004/005/009` | no reproduce |
| PER-API-010 | bajo | `noise_scale` debe ser numero real (no bool/None) en `apply_event` y `widen_gpt` | `test_ancla_per_api_010_*` | no reproduce |
| PER-API-011 | bajo | `pad_adam_entry` con lista explicita `exp_avg/exp_avg_sq/max_exp_avg_sq` | `test_ancla_PER_API_011` | no reproduce |
| PER-MAT-001 | bajo | `widen_gpt` escala solo las filas q VIEJAS (`omap_qkv[:d]`): presupuesto qkv = 1.000·ns (medido) | `test_ancla_per_mat_001_*` | ratio 1.000 (el PoC falla solo contra su propia prediccion del defecto) |
| PER-MAT-002 | bajo | `_functional_noise` lanza `NoiseError` si el cast al dtype colapsa el ruido a cero | `test_ancla_PER_MAT_002` | NoiseError (arreglado) |
| PER-MAT-003 | bajo | `dense_to_factorized`: `RuntimeWarning` + docstring en fp16/bf16. **S NO pasa a fp32**: verificado que forward y `reconstruct_weight` sin autocast fallan con dtypes mixtos | `test_ancla_PER_MAT_003` | reproduce (decision documentada) |
| PER-MAT-004 | bajo | `widen_layernorm`: `eps' = eps·old/new` con `variance_compensation=True` (exacto) | `test_ancla_PER_MAT_004` | no reproduce |
| PER-MAT-006 | bajo | comentario corregido + piso de KL bf16 en SAFETY.md | — | reproduce (inherente al dtype; documentado) |
| PER-API-003 | bajo | docstring `widen_embedding` + SAFETY.md | — | reproduce (documentado) |
| PER-API-008 | bajo | SAFETY.md: "exactamente `torch.optim.AdamW`" | — | reproduce (documentado) |

No tocados (por decision): PER-MAT-005 (momentos de γ, limitacion declarada), PER-MAT-007 (fuera de
alcance: ElasticRank/export, pendiente de su propio peritaje), PER-MAT-008 (redaccion de la SPEC).

## Decision de diseno registrada (PER-LOG-006)
El contrato inicial prescribia `RuntimeError` en `forward`. El implementador se bloqueo correctamente: 4
tests preexistentes de `tests/test_elastic_shape.py` miden deriva tras `widen_gpt`/`deepen_gpt` sin
optimizer (uso legitimo: `rel_drift` es API publica y el propio director midio gradientes asi durante el
peritaje). Un raise habria roto la inferencia y las sondas tras crecer sin optimizer. Se cambio a aviso
`RuntimeWarning` condicionado a `torch.is_grad_enabled()`: cubre exactamente el caso danino (paso de
entrenamiento con optimizer caduco) sin romper usos legitimos. Residual aceptado: el aviso puede
filtrarse; el remedio completo sigue siendo `apply_event`.

## Estado final
- `banco_final_jefe.json`: VERDE (compilar, importar, lint E9/F/B/PLE, suite completa, smoke = los tres
  archivos de tests de ElasticShape). Hash `ac8bab673e693d49`, identico al del implementador.
- Tests de ElasticShape: 74 → 74 + 9 anclas shape_ops + 12 anclas elastic_shape + 1 test forja (algunas
  parametrizadas). Cero tests existentes modificados o eliminados.
- Ruff limpio en los cuatro archivos tocados. Finales de linea: `shape_ops.py` devuelto a CRLF (como en
  HEAD; la revision 2026-09-07 lo habia pasado a LF y el diff contra HEAD baja de 617 a 207 lineas);
  `elastic_shape.py` LF como en HEAD.
- Archivos tocados por el lote: `src/mztrain/shape_ops.py`, `src/mztrain/elastic_shape.py`,
  `tests/test_shape_ops.py`, `tests/test_elastic_shape_safety.py`, `docs/ELASTICSHAPE-SAFETY.md`,
  `CHANGELOG.md` (entrada del director). Nada commiteado; el sello firmado sigue sin cubrir esta revision.
- Revision G4 ciega del lote: `revision/reporte_REVISOR.json` (ver seccion al final).

## Revision G4 ciega del lote (forja) y contra-arreglos
Revisor independiente (solo peritaje + ordenes + diff; sin los reportes de los reparadores):
**aprobar_con_cambios**, 7 hallazgos menores, 0 mayores/bloqueantes. Verifico por su cuenta: 96/96 tests;
las 22 anclas fallan contra el arbol previo y pasan contra el actual; ningun test previo tocado; G4-B
conservado; PER-LOG-002 correcto en 7 tipos de evento x 3 combinaciones; SDPA exacta (2.8e-16/4.4e-16)
y coherente con `_rescale_q_state`; guard de `forward` no se dispara dentro de `apply_event` ni bajo
`simplefilter("error")` ni con autocast bf16. Reporte: `revision/reporte_REVISOR.json`.

Contra-arreglos del jefe (mostrador, 1 linea cada uno, banco como juez):
- PER-MAT-002: criterio de colapso `(noise == 0).all()` en vez de `norm() == 0` en dtype bajo (falso
  positivo con `noise_scale <= 1e-22` en fp32; verificado: 48/48 elementos no nulos aceptados).
- PER-LOG-004: `probe_targets` solo `int64` (`cross_entropy` rechaza int32/int16/int8; ahora ValueError
  temprano con `model.d` intacto).
- PER-API-010: `noise_scale` acepta `numbers.Real` (np.float32 vuelve a funcionar en `widen_gpt` y
  `apply_event`; tensores 0-d siguen rechazados, coherente con `_require_int`).
- SAFETY.md: (a) precisa que la base de warmup solo se libera al llegar a `done` y persiste en el
  `state_dict` (residuo aceptado: warmup abandonado o reanudado); documenta `NoiseError` -> rechazo, el
  reescalado de `eps` y su no-persistencia en `state_dict`, y que un override sin cirugia devuelve un
  optimizer nuevo; corrige la frase sobre la correccion de Q.
- Aceptados sin cambio: `stacklevel=2` del aviso (una vez por proceso, atribuido a torch) y el residuo
  de PER-LOG-001 (indistinguible de G4-B por diseno).

Banco final del jefe tras los contra-arreglos: `banco_final2_jefe.json` **VERDE**, hash `ced9a2541f6c7423`.
Ruff limpio. Suite ElasticShape: 96 passed.

## Veredicto del lote
- `shape_ops.py`: REPARADO-2. `elastic_shape.py`: REPARADO-2. Forja PER-LOG-006: COMPLETA (G4 aprobado con
  cambios, cambios aplicados y re-verificados).
- Gate del peritaje tras el lote: el alto (PER-LOG-001) y los dos medios estan reparados y anclados; los
  bajos reparados o documentados; residuo declarado. Pendiente del usuario: revisar el diff, commitear y
  re-sellar (`scripts/seal.py sign`) — el sello firmado sigue describiendo el snapshot anterior.

## Lote 2 (2026-09-26): auditoria y cierre de los residuos
Ver `residuos/AUDITORIA-RESIDUOS.md`. 7/7 residuos reproducidos → 0/7 tras forja (implementador opus +
revisor ciego opus: aprobar, 5 menores, 4 contra-arreglos + 2 anclas). Banco final del jefe
`banco_residuos2_jefe.json` VERDE hash `080c70e9a97a3885` (suite completa; `layers.py` en ambito).
Cambios de diseno: `reconstruct_weight` = W efectiva (con `wake_gate`); reset de gates en
`grow_rank(preserve=False)`; forward tolera U/V bf16 + S fp32; `dense_to_factorized` conserva S fp32;
`LrWarmup` con regla "la escritura externa gana"; `GPT(..., ln_eps=)` para reconstruir el eps compensado.
No reparado por diseno: PER-MAT-005 (no existe reescalado exacto de los momentos de γ).
