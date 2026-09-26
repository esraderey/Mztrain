# Release 1.4.0: MNEME desde PyPI (`mnemosys`) — 2026-09-26

## Lo que se midio antes de decidir (mnemosys 1.0.1, PyPI, mismos autores; import `mneme`)
- Exporta todo lo que mztrain importa (ZSpace, MnemeConfig, CompressionLevel, CompressionConfig, compress_model,
  get_compression_stats, SecureStorageBackend, StorageConfig, create_secure_config). Dependencia extra `mscs` (PyPI).
- API distinta al MNEME 2.0.1 local: `load(name)` sin kwarg `device`; SI tiene `delete(name)` (el local no).
- Fidelidad del round-trip de activaciones N(0,1) con `register` por defecto: **error relativo 0.89-0.99** en
  tensores >= 256x512 (su enrutado automatico descompone tensorialmente); con `quantization_type="int8"` (cadena; el
  enum revienta en `"int4" in quant_type`): **0.0058**, comparable al 2.0.1 local (0.0065).
- Coste por activacion 16x256x512 en CUDA: register ~350 ms / delete ~240 ms con `storage_path` por defecto
  (`./mneme_storage`, persiste a disco); 172 / 75 ms con `storage_path` temporal. INT8 interno de mztrain: ms.
  Registro acotado (0 entradas tras 200 register/load/delete).

## Cambios (forja R2; contrato `../ordenes/CONTRATO-FORJA-MNEMOSYS.md`)
- `src/mztrain/checkpoint.py`: adaptador de backend en `ZActivationCheckpoint`: `prefer_zspace` (opt-in, default
  False); `storage_path` temporal y `enable_encryption=False` para el store efimero; exige `delete`/`remove` (si no,
  `RuntimeWarning` + fallback INT8: cierra la fuga D1 del informe "corre todo"); `register(..., quantization_type="int8")`
  con fallback a `register` plano; `load` con o sin kwarg `device` y restauracion del device original; sonda de
  fidelidad en el primer uso (rel > 0.05 -> aviso + fallback). Limpieza de dos avisos de lint preexistentes.
- `src/mztrain/config.py`: `ZTrainConfig.mneme_activation_store: bool = False`. `src/mztrain/engine.py`: lo propaga a
  `ZActivationCheckpoint.prefer_zspace` al aplicar el checkpointing.
- Empaquetado: extra `mneme = ["mnemosys>=1.0.1"]` (pyproject + setup.py), `requirements.txt`, version **1.4.0** en
  pyproject/setup/`__init__`. README: seccion "MNEME / MNEMOSYS" y "Relacionados". CHANGELOG `[1.4.0] - 2026-09-26`.
- Tests: 7 nuevos en `tests/test_checkpoint.py` (opt-in por defecto, backend sin eviccion, kwargs INT8, register
  plano, device sin kwarg [CUDA], baja fidelidad, reset). El ancla A5 del peritaje de agosto sigue verde.

## Evidencia
- Banco de maestranza (suite completa, ambito con checkpoint/layers/shape_ops/elastic_shape y sus tests): VERDE,
  hash `fa66f4f9c0b21fbb` (`../banco_mnemosys2_jefe.json`). Ruff limpio en los archivos tocados.
- Smoke con mnemosys REAL (`D:/mztrain/.tmp/mnemosys-smoke/smoke_mnemosys.py`, PYTHONPATH al site aislado): por defecto
  no se construye ZSpace; opt-in: rel_err 0.0059, device y dtype conservados, sin avisos de mztrain, registro 0 tras
  31 activaciones, 267 ms/activacion, storage en `%TEMP%\mztrain_act_ckpt_*`. Con el MNEME 2.0.1 local (sin `delete`):
  aviso y fallback INT8, `_zspace=None`.
- `examples/example_zcodebert.py` (antes 1358 s la epoca 0 y OOM en la 1): ahora **~22 s/epoca**, 0 registros ZSpace
  (`../../run-all-20260926/example_zcodebert_INT8.log`, cortado por limite de tiempo en la epoca 8).
- `python -m build` -> `.tmp/dist-1.4.0/mztrain-1.4.0-py3-none-any.whl` + `.tar.gz`; `twine check`: PASSED x2; wheel
  instalado con `--no-deps` en un target aislado: importa, `__version__ == "1.4.0"`, `Provides-Extra: mneme`,
  `Requires-Dist: mnemosys>=1.0.1; extra == "mneme"`.
- Revisor ciego G4: ver `../revision3/reporte_REVISOR.json` (seccion al final).

## Lo que queda en manos del titular (clave privada y credenciales)
```bash
cd D:\mztrain
git add -A && git commit -m "v1.4.0 - MNEME desde PyPI (mnemosys) con store de activaciones opt-in y adaptador de backend"
python scripts/seal.py sign --key C:\Users\Raul\.mztrain-keys\mztrain-priv.pem
python scripts/seal.py verify
git add SEAL.json MANIFEST.sha256 && git commit -m "sello: v1.4.0" && git tag v1.4.0
python -m build --outdir dist
python -m twine check dist/mztrain-1.4.0*
python -m twine upload dist/mztrain-1.4.0*
```
Recomendacion previa: sacar `.tmp/` del indice (`git rm -r --cached .tmp` + `.gitignore`): el ultimo commit incluyo
252 archivos de expedientes y logs (161 MB en disco) que el sello ya excluye.

## Residuos declarados
- Con `mneme_activation_store=True` y mnemosys 1.0.1 el store escribe cada activacion a disco y no borra el directorio
  temporal al terminar el proceso (crece durante la corrida; se libera al reiniciar la maquina o limpiando `%TEMP%`).
- mnemosys 1.0.1 emite `ResourceWarning: unclosed database` (sqlite) por operacion: defecto suyo, no de mztrain.
- Checkpoints de febrero sin `wake_gate`/`sleep_mask` (D2 del informe "corre todo") siguen sin cargar: fuera de este lote.

## Revision G4 ciega y contra-arreglos (cierre)
Revisor independiente (opus, solo contrato + `git diff HEAD`): **aprobar_con_cambios**, 1 mayor + 8 menores, 0
bloqueantes (`../revision3/reporte_REVISOR.json`). Verifico por su cuenta: 63 tests, smoke real OK (0.00587), twine
PASSED, wheel con `Requires-Dist: mnemosys>=1.0.1; extra == "mneme"`, los 7 tests nuevos fallarian contra HEAD.
Contra-arreglos aplicados (todos con test nuevo):
- (mayor) la sonda de fidelidad usa ahora la FORMA, dtype y device de la primera activacion real (el enrutado que
  destruye tensores grandes solo se manifiesta por tamano) y un generador propio (no consume el RNG del run);
  `_register` solo cae al `register` plano si el `TypeError` es por el kwarg `quantization_type`; extra acotado a
  `mnemosys>=1.0.1,<2` (pyproject, setup.py, requirements).
- `ZSpace(config)` que lanza -> store deshabilitado con aviso y fallback (antes la excepcion salia en cada compress).
- Directorio temporal del store: se borra en `reset()` y al salir del proceso (`atexit`); limpiados los 48 huerfanos
  de las corridas anteriores en `%TEMP%`.
- `mneme_activation_store=True` sin mnemosys instalado -> `RuntimeWarning` explicito (antes caia en silencio).
- Activacion comprimida en INT8 antes de habilitar el store -> `decompress_activation` la busca en el fallback en vez
  de fallar con `not found` (cambio del flag entre forward y backward).
- `MnemeConfig(enable_encryption=False)` en el constructor (mnemosys avisa en `__post_init__`); docstring de config
  documenta que el ajuste es global del proceso; README documenta el opt-in fuera del engine
  (`ZActivationCheckpoint.prefer_zspace = True`); CHANGELOG sin encabezado duplicado; README con sus finales de linea
  originales (diff limpio: +25/-7).
Aceptado sin cambio: `prefer_zspace` es un flag de clase (documentado); mnemosys emite `ResourceWarning` sqlite por
operacion (defecto suyo).

## Estado final verificado por el director
- Banco (suite completa del repo, ambito con checkpoint/layers/shape_ops/elastic_shape y sus tests): **VERDE**, hash
  `1a4c8111d740f2e4` (`../banco_mnemosys3_jefe.json`). `tests/test_checkpoint.py` 10 -> 23 tests.
- Smoke con mnemosys 1.0.1 real: OK (rel_err 0.00587, device/dtype conservados, registro 0, 275 ms/activacion).
- `python -m build`: `mztrain-1.4.0-py3-none-any.whl` + `.tar.gz` en `D:\mztrain\.tmp\dist-1.4.0\`; `twine check`
  PASSED x2; wheel instalado aislado importa 1.4.0 con `prefer_zspace=False`; sdist sin `.tmp/` (111 archivos).
- Arbol: 10 archivos modificados del lote + 4 JSON de `bench/results` (benchmarks de hoy) + `.tmp/` (untracked nuevos
  y staging regenerado). Nada commiteado.
