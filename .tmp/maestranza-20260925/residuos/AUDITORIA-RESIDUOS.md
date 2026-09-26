# Auditoria de los residuos del peritaje ElasticShape y su cierre — 2026-09-26

Origen: residuos declarados en `../CIERRE-LOTE.md` (lote 1) y escalacion PER-MAT-007 del peritaje.
Nivel A1 (superficie acotada, sin entrada externa): reconocimiento y PoC del director; reparacion con forja
(un implementador, revisor ciego) y banco de maestranza como juez. Todo re-ejecutado por el director.

## Reconocimiento (Z0)
- `reconstruct_weight` lo consumen: exportacion a denso (`engine.py:1557`, `models/zcodebert.py:518`),
  `refactorize.py:83` (SVD fresca) y `grow_rank(preserve_weights=False)` (solo via `ZSparse.grow_rank`).
- Los gates parciales existen de verdad: tras un revive, `tick_wake` (`elastic_rank.py:552-566`) rampa
  `wake_gate` 0→1 durante `elastic_rank_wake_warmup_steps` (100 por defecto). Un export o refactorize en ese
  lapso ve una W distinta de la funcion entrenada. `reset_after_refactorize` (engine) ya asume que la base
  cambia y pone gate=1; `grow_rank(preserve=False)` no lo hacia.
- `LrWarmup`: la base persistente solo se liberaba al llegar a `done`; abandonos y reanudaciones desde
  checkpoint (claves en `param_groups`, serializadas) conservaban la base vieja.
- `widen_layernorm` reescala `eps` pero `eps` no es buffer: `GPT` reconstruido desde su configuracion no lo
  recupera.
- `ZFactorizedLinear.forward` y `reconstruct_weight` fallaban con U/V bf16 y S fp32 sin autocast: por eso
  `dense_to_factorized` dejaba S en bf16.

## Hallazgos reproducidos (PoC `poc_residuos.py`, Z1, 7/7 antes → 0/7 despues)

| ID | Sev. | Hallazgo | Medida antes | Arreglo (forja, decision del director) | Ancla |
|---|---|---|---|---|---|
| R1 | medio | export a denso ≠ forward con `wake_gate` parcial | gap rel 0.867 | `reconstruct_weight` = W efectiva `U diag(S·gate) V` (ZFactorizedLinear y ZSparse), promocion de dtype, salida en dtype de U | `test_residuo_R1_*` ×3 (incl. continuidad de `refactorize`: salto 3.5e-7 vs 0.92 antes) |
| R2 | bajo | `grow_rank(preserve=False)` reaplica gates viejos a la base SVD nueva | `wake_gate=[1,.5,0,1,1]` | reset de `wake_gate`/`sleep_mask` en esa rama (la re-SVD cambia la base) | `test_residuo_R2_*` |
| R3/R3b | medio | warmup abandonado o reanudado + LR externo → siguiente warmup vuelve al pico viejo | LR final 1e-3 con vigente 1e-5 (100×) | regla "la escritura externa gana": `_mzshape_warmup_lr` (copia escalar del ultimo LR escrito); base heredada solo si nadie toco el LR; ambas claves limpiadas al terminar | `test_residuo_R3_*`, `R3b`, G4-B y PER-LOG-001 siguen verdes |
| R4 | bajo | `eps` compensado no reconstruible desde la configuracion | eps 5e-6 vs 1e-5; drift 5e-6 | `GPT(..., ln_eps=)`, `Block(..., eps=)`, `model.ln_eps` fijado por `widen_gpt` (restaurado por rollback) | `test_residuo_R4_*` ×2 (drift 0.0) |
| R5/R5b | bajo | S bf16 tras `dense_to_factorized`; forward mixto imposible | `S.dtype=bf16`; `RuntimeError` | forward tolera dtypes mixtos sin autocast (guard `is_autocast_enabled(device_type)`, excepto `meta`); S fp32 en la conversion; aviso mantenido | `test_ancla_PER_MAT_003` (S fp32), `test_residuo_R5b_*` ×2 (incl. invariancia bajo autocast: `torch.equal`) |

Descartado como reparacion: PER-MAT-005 (momentos de γ no reescalados). El factor √(d'/d) del gradiente de γ
solo es exacto con media cero; con el residuo de media de LN no lo es, asi que un reescalado "exacto" no
existe. Sigue como limitacion documentada (SAFETY.md) pendiente de experimento.

## Verificacion
- Implementador (opus): `parcial` por dos motivos del director (mi PoC R4 reconstruia sin `ln_eps`; `F401
  Tuple` preexistente en `layers.py` destapado por el ambito nuevo). Ambos resueltos por el director.
- Banco del jefe (suite completa del repo, ambito con `layers.py`): `banco_residuos_jefe.json` VERDE hash
  `4f36f6fd435ba026`; tras los contra-arreglos del revisor: `banco_residuos2_jefe.json` VERDE hash
  `080c70e9a97a3885`. Smoke = suite entera (411+ tests). Ruff limpio.
- Revisor ciego G4 (opus, solo contrato + diffs): **aprobar**, 5 menores. Sondas propias: con gate=1 forward y
  `reconstruct_weight` bit a bit iguales a HEAD en fp32/fp64/bf16, tambien bajo autocast bf16 (CPU y CUDA);
  `is_autocast_enabled(device_type)` confirmado en torch 2.11 y tratado como constante por Dynamo.
  Contra-arreglos aplicados (mostrador, banco como juez): copia escalar del LR (alias con LR tensor:
  verificado `fill_` externo → 1e-5), guard `meta`, nota de migracion de checkpoints antiguos, "exacta salvo
  redondeo bf16" en SAFETY, dos anclas nuevas (invariancia bajo autocast; continuidad de refactorize).
- Objecion mas fuerte del revisor (semantica de `reconstruct_weight` para consumidores no tocados) evaluada y
  no bloqueante: la funcion del modelo es la gated; el unico llamador de `refactorize` resetea gates justo
  despues (engine); llamar sin reset ya era incorrecto en HEAD.

## Estado final
Archivos tocados en este lote: `src/mztrain/layers.py` (nucleo), `src/mztrain/shape_ops.py`,
`src/mztrain/elastic_shape.py`, `tests/test_layers.py`, `tests/test_shape_ops.py`,
`tests/test_elastic_shape_safety.py`, `docs/ELASTICSHAPE-SAFETY.md`, `CHANGELOG.md`. Tests de las cuatro
suites de ElasticShape/layers: 136 passed. Nada commiteado; el sello firmado sigue describiendo el snapshot
anterior: revisar, commitear y `scripts/seal.py sign`.
