# Auditoría: ElasticShape (shape_ops.py + elastic_shape.py) — 2026-09-25

## Alcance y modelo de amenazas

- **Objetivo:** `src/mztrain/shape_ops.py` (348 líneas, 8 primitivas públicas) y `src/mztrain/elastic_shape.py` (676 líneas: GPT de referencia, transacción de rollback, cirugías, migración AdamW, `apply_event`, `ShapeSchedule`, `LrWarmup`), en el **árbol de trabajo** sobre `e095ac0` con la revisión "cirugía reversible" (2026-09-07) **sin commitear**. `docs/ELASTICSHAPE-SAFETY.md` se tomó como contrato declarado.
- **Fuera de alcance:** engine, ElasticRank, refactorize, checkpoint, precision; la evidencia empírica T8–T14 (no se re-corre); DDP, `torch.compile`, optimizadores distintos de `torch.optim.AdamW`.
- **Nivel A2.** API pública en PyPI, refactor de >1000 líneas, fallo = corrupción silenciosa del entrenamiento. Sin entrada externa no confiable, dinero, cripto ni concurrencia: no A3.
- **Activos:** preservación funcional prometida; integridad del estado AdamW migrado; atomicidad del rollback (modelo + optimizer + RNG bit a bit); fallos ruidosos; fidelidad de SAFETY.md.
- **Actor:** caller honesto con entradas de borde y secuencias fuera del camino feliz; el propio código. No hay adversario externo.
- **Límites de confianza:** API exportada en `mztrain.__init__`; formato interno del estado AdamW de torch; dtypes mixtos (U/V bf16 con S fp32); CPU/CUDA; generator explícito.
- **Laboratorio:** Z0 (lectura, 3 peritos solo-lectura) + Z1 (copia desechable en scratchpad, venv del proyecto, sin red, sin instalar nada; ejecuta únicamente el director). Sin contenedor: aislamiento por copia, no por kernel. Runtime: Python 3.13.2, torch 2.11.0+cu128, RTX 4060.
- **Línea base:** 74/74 tests de ElasticShape verdes antes de auditar (`test_shape_ops`, `test_elastic_shape`, `test_elastic_shape_safety`).

## Resumen ejecutivo

| Severidad | Confirmados | Descartados |
|---|---:|---:|
| crítico | 0 | 0 |
| alto | 1 | 1 |
| medio | 2 | 0 |
| bajo | 15 | 0 |

**Gate: BLOCKED** (un alto confirmado) · cobertura 82 % (autodeclarada con soporte AST: los 51 símbolos de ambos módulos leídos por ≥2 peritos) · confianza alta · eficacia **no calibrada** (sin ensayo crisol de esta cuadrilla). Score secundario: 89/100 (correctitud 64, robustez 78, resto ≥96).

Lo que importa:

1. **PER-LOG-001 (alto).** `LrWarmup` fija `_mzshape_base_lr` una sola vez y nunca lo borra; `_new_adamw` lo copia al optimizer migrado. Si el LR cambia entre growths (scheduler, decaimiento manual), el siguiente warmup rampa hasta el **pico viejo** (10× en el PoC, 30× en la sonda del director) y se queda ahí, sin aviso. Hallado de forma independiente por el director y por el perito de lógica.
2. **PER-LOG-002 (medio).** `apply_event(lr=, weight_decay=)` ignora el override cuando el evento no migra (factorize redundante, `new_d == d`) y devuelve `accepted=True`. Encadena con el anterior: pasar `lr=` como mitigación no sirve en eventos no-op.
3. **PER-LOG-006 (medio).** `widen_gpt`/`deepen_gpt` directos sin `migrate_optimizer` dejan entrenar con el optimizer viejo sin error: 20/20 parámetros nunca se actualizan (widen) o el bloque nuevo queda congelado (deepen).

**Lo que NO se rompió** (y se atacó en serio): la transacción de rollback es bit a bit correcta con fallo inyectado en cada etapa y con rechazo por calidad; la migración AdamW (grupos, excluidos, `param_names`, AMSGrad, sin alias, `step` por valor) es correcta; la corrección SDPA y el re-escalado del estado q (1/c, 1/c²) son **exactos** elemento a elemento; la calibración funcional del ruido por QR es exacta en fp32/fp64; la identidad de `deepen` es exacta incluso con autocast/fp8; los mapas `qkv_out_map`/`attn_in_map` son correctos índice a índice; el único "alto" propuesto por un perito (targets todo `-100` aceptados) **se descartó** al ejecutarlo.

## Hallazgos

Orden estricto de severidad. Ubicaciones en `src/mztrain/...` sobre el árbol de trabajo (`e095ac0+wt`). Todo PoC vive en `peritos/<PERITO>/poc_<ID>.py` del expediente y falla con `AssertionError` si el defecto existe; todos fueron re-disparados por el director en Z1.

### PER-LOG-001 · alto · correctitud · `elastic_shape.py::LrWarmup.__init__` (650-661) y `_new_adamw` (351-395)
- **Causa raíz (ROOT-WARMUP-BASE):** `g.setdefault("_mzshape_base_lr", g["lr"])` solo escribe la primera vez; `done` nunca limpia la clave; `_new_adamw` la deep-copia al optimizer nuevo.
- **Precondiciones / actor:** un warmup anterior terminó; el LR del grupo cambió después; nuevo growth + `LrWarmup`. Caller honesto con cualquier LR schedule.
- **Invariante rota:** docstring "rampa lineal floor→1.0 **del LR** tras un growth" (el LR vigente).
- **PoC:** `poc_PER-LOG-001.py`; también `poc/director_probe.py` bloque D4, escrito antes de recibir el reporte del perito.
- **Salida observada:** `lr previo=0.0001 lr tras warmup=0.001 _mzshape_base_lr=0.001` → 10×; D4: LR vigente 1e-5, final 3e-4 → 30×.
- **Control negativo:** sin cambio externo del LR, el warmup termina exactamente en el LR vigente.
- **Impacto:** salto silencioso de LR de un orden de magnitud en el segundo growth. T10 mostró que a 1.2e-3 4/6 referencias degeneran: riesgo real de divergencia de un run largo.
- **Remediación (maestranza, arreglo mínimo + ancla):** borrar `_mzshape_base_lr` en `step()` al alcanzar `done` (y en `__init__` si no hay warmup activo, tomar `g["lr"]` como base). Documentar en SAFETY.md la interacción con schedulers externos. El PoC pasa a `tests/`.
- **Verificación:** re-disparo del director ✔; hallazgo independiente del director (D4) ✔.

### PER-LOG-002 · medio · correctitud · `elastic_shape.py::apply_event` (596-625)
- Overrides `lr`/`weight_decay` solo llegan a `_new_adamw`, que no se invoca sin conversión ni migración. Reporte: `accepted=True`, grupos intactos `(0.001, 0.01)` cuando se pidió `(0.0005, 0.05)`.
- Contrato roto: SAFETY.md "un override explícito afecta a todos". Control negativo: con `add_layers=1` sí se aplica.
- Remediación: aplicar overrides siempre (o rechazar con `ValueError` si no hay nada que hacer) y anotarlo en `report["optimizer"]`.

### PER-LOG-006 · medio · robustez · `elastic_shape.py::widen_gpt` (221-303), `deepen_gpt` (305-343)
- El guard de ledger pendiente solo dispara en el **siguiente** widen/apply_event. Entre medias `opt.step()` con el optimizer viejo no falla y no entrena nada (`congelados=20/20`; deepen `16/16`).
- Mitigación existente: SAFETY.md recomienda `apply_event` de principio a fin; pero las primitivas son API pública exportada.
- Remediación (forja, decisión de diseño): flag en el modelo que haga fallar `forward` o emita warning mientras haya ledger/sources pendientes; como mínimo, docstring explícito.

### Bajos (15)

| ID | Ubicación | Qué | Salida observada | Remediación |
|---|---|---|---|---|
| PER-LOG-005 | `migrate_optimizer` 400-438 | Re-migrar un ledger ya consumido con el opt previo se acepta | `step 3.0 vs 7.0` del opt vivo (4 pasos perdidos) | marcar recs consumidos / exigir ledger pendiente |
| PER-LOG-003 | `shape_ops::_functional_noise` 86-87 | Ruido no representable → `ValueError` en vez de `accepted=False` (embedding sí rechaza); modelo restaurado igual | `ValueError(...) modelo restaurado=True` | lanzar `_GrowthRejected` |
| PER-LOG-004 | `apply_event` 565-566 | `probe_targets` validado solo por forma; dtype/rango se detectan tras la cirugía (rollback OK en CPU) | `IndexError ... cirugias_ejecutadas=1 model.d=24` | validar dtype/rango antes de la transacción |
| PER-API-001 | `shape_ops::_validate_map` 29 | Mapas float truncados en silencio | `[0.1,1.1,2.1,3.1]` aceptado | rechazar no enteros |
| PER-API-002 | `scale_output_rows` 246-247 | factor 0 / negativo aceptado | `factor=-2.5` invierte filas | exigir `factor > 0` |
| PER-API-006 (+PER-LOG-007, ROOT-WARMUP-FLOOR) | `LrWarmup.__init__` 650-653 | `floor` sin validar: NaN, negativo, >1 | `lr=nan` → pesos NaN en 1 paso; `floor=2.0` pico 2× | `0 <= floor <= 1` finito |
| PER-MAT-003 (=D3 director) | `dense_to_factorized` 266-271 | Linear bf16 → **S en bf16** (regla "S siempre FP32" que `widen` sí respeta); error 2.9e-3 > noise 1e-3 | `S.dtype=bfloat16 rel=2.92e-03` vs fp32 `1.08e-06` | S en fp32 siempre; documentar error bf16 |
| PER-MAT-001 (=D7) | `widen_gpt` 274-277 | La corrección SDPA multiplica también las filas q **nuevas** con ruido: presupuesto qkv excedido hasta ×c | 4×: `1.386×` predicho y medido; 2×: `1.1606e-2` vs `1.1547e-2` | escalar solo `omap[:d]` o cuantificar en SAFETY |
| PER-MAT-004 | `widen_layernorm` 225, 233-234 | eps no compensado: eps efectivo `eps·d'/d` | σ²=eps: err 0.1835; σ²=1: 5e-6; con `eps'=eps·d/d'`: 2e-16 | pasar `eps*old/new` al LN nuevo |
| PER-MAT-002 | `_functional_noise` 85-87 | fp16: subnormales → ruido cero silencioso (solo `isfinite`) | `ns=1e-5: ratio=0.000 fracción cero=1.00` | rechazar/avisar si castea a cero |
| PER-MAT-006 | `_probe_metrics` 525-526 | En bf16 la KL la domina el redondeo de logits; comentario "FP64 evita…" falso | bf16 `2.78e-5` vs real `3.12e-7` (×89) | corregir comentario; piso por dtype en SAFETY |
| PER-API-003 | `widen_embedding` 186-190 | Tabla de norma 0 → ruido 0 con `noise_scale>0`, no documentado | columnas nuevas exactamente 0 | documentar |
| PER-API-004 (ROOT-TYPE-VALIDATION; agrupa 004/005/009/010) | `pad_state_tensor` 312, `pad_adam_entry` 341, `widen_layernorm` 225, `apply_event` 573 | Errores internos (`KeyError('step')`, `TypeError` de torch) en vez de `ValueError` propio | ver PoCs | validar tipos al inicio |
| PER-API-011 | `pad_adam_entry` 343-347 | Heurística "misma forma que exp_avg" expande claves ajenas | `custom_adapter_counter` → (6,) | lista explícita de claves |
| PER-API-008 | `_require_adamw` 346-348 | `type() is` rechaza subclases; SAFETY dice "AdamW estándar" | subclase trivial → `TypeError` | documentar "exactamente AdamW" |

### Descartado

- **PER-API-007** (propuesto alto): "apply_event acepta `probe_targets` todo `-100`". El perito asumió `cross_entropy(mean)` = 0.0 con todo enmascarado. Verificado: en torch 2.11 devuelve **NaN**, la métrica no es finita y el evento se **rechaza** con el optimizer original (`accepted: False | reason: metricas no finitas en la sonda`). SAFETY.md cumple lo que promete.

### Cadenas

- PER-LOG-001 + PER-LOG-002: la mitigación natural del primero (pasar `lr=` al evento) no se aplica en eventos no-op. Severidad de la cadena = la del primero (alto).
- PER-LOG-006 + PER-LOG-005: ambos son huecos de la misma máquina de estados del ledger fuera de `apply_event`; raíces distintas, misma dirección de arreglo (el ledger como objeto con estado consumido/pendiente).

## Superficie examinada sin hallazgos confirmados

- **Transacción/rollback (`_shape_transaction`, `_atomic_shape`).** Instantánea superficial de `__dict__` + copia de `_modules`; verificado por lectura (PER-LOG, PER-API) que toda mutación sobre objetos preexistentes es un rebind en `_modules`/`__dict__`, nunca escritura in-place en `.data`, `_parameters`, `_buffers` ni `.grad`. Verificado por ejecución (director, D1): fallo inyectado en `widen_gpt`, `deepen_gpt`, `migrate_optimizer` y `_rescale_q_state`, y rechazo por calidad → `state_dict`, ids de módulos y parámetros, estado y grupos del optimizer, `model.d`, atributos `_mzshape_*` y RNG **bit a bit** iguales. Transacciones anidadas y modos `training` de la sonda restaurados. Límite: no se probó `KeyboardInterrupt` real ni CUDA device-side assert.
- **Migración AdamW (`_new_adamw`, `migrate_optimizer`, `pad_*`).** Mapeo old→new coherente con los mapas de widen para embeddings, LN, U/S/V/bias; 2 grupos, excluidos, `param_names`, `initial_lr`, `_mzshape_base_lr`, AMSGrad, estado perezoso sin crear; ningún tensor comparte storage; `step` por valor; orden `qkv, proj, fc1, fc2` del closure igual al de `Block.__init__`. `capturable`/`fused` cubiertos por el test CUDA existente, no re-auditados.
- **Corrección SDPA y `_rescale_q_state`.** Derivación (PER-MAT) y medida aislada a nivel de capa (director, `director_d6_layer.py`): forward exacto en dims viejas (diff 0 / 4e-16 con bias), gradiente de U_q y b_q escala **exactamente 1/c** elemento a elemento, gradientes de V, S, U_k, U_v invariantes; `q_rows` correctos; `max_exp_avg_sq/c²` coherente. La desviación vista a nivel de red (0.728 vs 0.707) es deriva de LayerNorm, documentada.
- **Calibración funcional del ruido (QR).** Exacta para cualquier relación in/r, S con signo, gate con ceros, rango 1, count 1; en fp32/fp64 ratio 1.00000000 con gauges de 1e-3 a 1. Usa `_gated_s` (la W del forward).
- **LayerNorm con compensación.** Factor √(d'/d) cancelado; residuo de primer orden en μ tal como declara la SPEC (bandas medidas en G4 no re-verificadas).
- **`deepen_gpt`.** Identidad exacta también bajo autocast y fp8 (clamp de amax → 0); U_proj nace con gradiente ≠ 0, V/S con 0; S, bias, dtype por parámetro, modos y `requires_grad` heredados.
- **`dense_to_factorized` fp32:** error 1e-6. **Embeddings:** calibración correcta, padding excluido, `tok`/`pos` con ruido independiente. **`qkv_out_map`/`attn_in_map`:** correctos índice a índice (d 4→8, 2 cabezas). **`_probe_metrics`:** dirección de la KL, `clamp_min`, `ignore_index` coherentes con `GPT.forward`. **`ShapeSchedule`:** catch-up y consumo correctos; duplicados en el mismo paso preservan orden. **Nota 3.16·lr:** correcta como límite t→∞ del primer paso.
- **Fidelidad documental:** "34 regresiones nuevas" = 24 funciones × parametrizaciones = 34 casos, coherente. Cobertura 93 %/94 %, Ruff/Black: no verificados (ni refutados).
- **Seguridad / concurrencia / recursos / supply-chain:** sin sinks (eval/pickle/subprocess/red), sin estado compartido ni hilos, sin dependencias nuevas. Pico de memoria 2× documentado, no estresado. CI y sello fuera de este alcance (el sello firmado **no cubre** estos cambios sin commitear; SAFETY.md lo declara).

## Sospechas sin confirmar

- **PER-LOG-004b:** en CUDA, `probe_targets` fuera de rango → device-side assert → contexto CUDA inválido → rollback no garantizado. No ejecutado en CUDA en esta sesión (el PoC lo aísla con `--cuda` en subproceso). Qué falta: correrlo.
- **Fuera de alcance, escalación (PER-MAT-007, medio):** `ZFactorizedLinear.reconstruct_weight` ignora `wake_gate`; con gate ≠ 1 no es la W del forward (gap relativo 1.28 en el PoC). Lo usan export (`engine.py:1557`) y `refactorize.py:83`. Dentro de ElasticShape solo se usa con gate=1 (sin efecto). Merece un peritaje de ElasticRank/export.

## Observaciones (no son hallazgos)

- Gradiente de γ (LN) escala √(d'/d) y sus momentos no se re-escalan (medido ×1.414): SAFETY.md lo declara como no abordado; incoherente con la política de q, pero limitación documentada.
- Tras `factorize`, S hereda el `weight_decay` del grupo del peso denso (0.1 sobre valores singulares): política no declarada.
- SPEC §4 "nunca ambos": escalar q y k por √c cada uno también es exacto; la SPEC quiere decir "no c en ambos". El widen fp32 con noise=0 del toy fue bit a bit exacto (más fuerte que lo prometido).
- La nota 3.16·lr describe el primer paso; PER-MAT deriva un pico posterior ≈6.6·lr (k≈12) y cota 7.27·lr. SAFETY ya avisa de que es "límite del primer paso".
- Los ejemplos de README/SAFETY no son ejecutables tal cual.

## Laboratorio

- Expediente: `audit-manifest.json`, `auditoria.json` (validado por `evaluar_auditoria.py --nivel A2`: exit 2 = BLOCKED, JSON válido), `inv/inventario_ast.json`, `peritos/PER-{LOG,MAT,API}/reporte.json` + PoCs, `poc/director_probe.py`, `poc/director_d6_layer.py`.
- Reproducir: copiar `src/`, `tests/` del árbol de trabajo a una carpeta limpia; `python peritos/<X>/poc_<ID>.py` con el venv del proyecto (Python 3.13, torch 2.11 cu128); ningún PoC necesita red, GPU ni datos.
- Presupuesto consumido: PER-LOG ≈217k tokens, PER-MAT ≈196k, PER-API ≈205k, director (recon + sondas + re-disparo + triaje) sin medir. Suficiencia alcanzada: las 3 clases con cobertura por símbolo; toda sospecha alta con veredicto.
- Destino de los confirmados: PER-LOG-001/002/005/003/004, PER-API-001/002/006/004/011, PER-MAT-003/004/002 → **maestranza** (arreglo mínimo + PoC como test ancla). PER-LOG-006 → **forja** (decisión de diseño). PER-MAT-001/006, PER-API-003/008 → documentación. Tras arreglar, re-verificar solo los hallazgos, no el módulo.
