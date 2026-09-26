# Contrato forja (R2): soporte de MNEMOSYS (PyPI) como backend MNEME y release 1.4.0 — 2026-09-26

## Hechos medidos (base del diseno)
- `mnemosys` 1.0.1 (PyPI, mismos autores) conserva el import `mneme`; exporta todo lo que mztrain importa
  (ZSpace, MnemeConfig, CompressionLevel, CompressionConfig, compress_model, get_compression_stats,
  SecureStorageBackend, StorageConfig, create_secure_config). `ZSpace` tiene `delete(name)`, `exists`, `load(name)`
  (SIN kwarg `device`) y `register(name, tensor, **kwargs)`.
- Enrutado automatico de `register` para tensores grandes = descomposicion tensorial: round-trip de activaciones
  N(0,1) con error relativo 0.89-0.99 (destruidas). Con `quantization_type="int8"` (cadena; el enum rompe con
  `"int4" in quant_type`): error 0.0058, comparable al MNEME 2.0.1 local (0.0065).
- Coste por activacion (16x256x512, CUDA): register ~350 ms / delete ~240 ms con `storage_path` por defecto
  (`./mneme_storage`, persiste a disco); 172 / 75 ms con `storage_path` temporal. El INT8 interno cuesta ~ms.
  Registro acotado tras 200 register/load/delete (0 entradas).
- MNEME 2.0.1 local (editable): sin `delete`/`remove` -> evicción imposible (fuga D1 del informe "corre todo").

## Decisiones (director)
D1 `src/mztrain/checkpoint.py` — adaptador de backend en `ZActivationCheckpoint`:
  1. Atributo de clase `prefer_zspace: bool = False` (OPT-IN). `_get_zspace` solo construye ZSpace si
     `HAS_MNEME and prefer_zspace` y no esta deshabilitado. Motivo: con mnemosys instalado por defecto, el store
     multiplicaria por 5-10 el coste de cada activacion.
  2. `MnemeConfig`: `compression_level=ULTRA_FAST` (como hoy) y, si el config tiene `storage_path`, un
     `tempfile.mkdtemp(prefix="mztrain_act_ckpt_")` (no escribir `./mneme_storage` en el cwd del usuario).
  3. Backend sin `remove`/`delete` callable -> se deshabilita el store con `warnings.warn(RuntimeWarning)` (una vez,
     razon en `_zspace_disabled_reason`) y se usa el fallback INT8. Nunca mas una fuga silenciosa.
  4. Sonda de fidelidad en el PRIMER `compress_activation` (device real del tensor): register/load/evict de
     `randn(64,64)`; si falla o el error relativo > 0.05 -> deshabilitar con warning y fallback INT8.
  5. `_register(z, key, t)`: `z.register(key, t, quantization_type="int8")`, y si el backend rechaza el kwarg
     (`TypeError`) -> `z.register(key, t)`. `_load(z, key, device)`: `z.load(key, device=device)` y ante `TypeError`
     `z.load(key)`; siempre `.to(device)` si difiere. `_evict`: `remove` o `delete`.
  6. `reset()` limpia `_zspace`, la razon de deshabilitado y la marca de sonda; NO toca `prefer_zspace`.
  7. El ancla existente `TestA5ActivationStoreFrees` (FakeZSpace con `register(k,t)`, `load(k, device=None)`,
     `remove`) debe seguir verde sin tocarla (parchea `_get_zspace`, asi que la sonda debe tolerar un fake que
     devuelve el mismo tensor).
D2 `src/mztrain/config.py`: `mneme_activation_store: bool = False` (docstring con el coste medido) junto a
   `mneme_compression_level`. `src/mztrain/engine.py::_apply_activation_checkpointing`: al inicio,
   `ZActivationCheckpoint.prefer_zspace = bool(getattr(self.config, "mneme_activation_store", False))`.
D3 Empaquetado: extra `mneme = ["mnemosys>=1.0.1"]` en `pyproject.toml` (sustituye la NOTA "no se distribuye por
   PyPI") y en `setup.py` `extras_require`; `requirements.txt` comenta `# mnemosys>=1.0.1  -> pip install mztrain[mneme]`.
   Version 1.4.0 en `pyproject.toml`, `setup.py`, `src/mztrain/__init__.py`.
D4 Docs: README seccion "MNEME" -> "MNEME / MNEMOSYS": `pip install "mztrain[mneme]"`, import `mneme`, que aporta
   (compress_model/GPTQ, SecureStorageBackend, store de activaciones opt-in con su coste medido), enlace a
   https://pypi.org/project/mnemosys/ y al repo; "Relacionados" enlaza mnemosys. CHANGELOG: `[Sin publicar] -
   2026-09-07` pasa a `[1.4.0] - 2026-09-26` y anade subseccion "MNEMOSYS desde PyPI" + "Compatibilidad" (store opt-in).
D5 Tests (`tests/test_checkpoint.py`): fakes por monkeypatch de `checkpoint.HAS_MNEME/ZSpace/MnemeConfig/CompressionLevel`:
   (a) por defecto (`prefer_zspace=False`) no se construye ZSpace aunque HAS_MNEME; (b) backend sin evicción ->
   RuntimeWarning + fallback + `_zspace` None; (c) backend con kwargs recibe `quantization_type="int8"`; backend que
   rechaza kwargs -> register plano; (d) `load` sin `device` -> tensor devuelto en el device original; (e) backend
   de baja fidelidad (devuelve ruido) -> RuntimeWarning + fallback; (f) tras N compress/decompress el registro del
   fake queda vacio (evicción por clave); (g) `reset()` conserva `prefer_zspace`.
   Smoke fuera de la suite (no depende de red en CI): `.tmp/mnemosys-smoke/smoke_mnemosys.py` con
   `PYTHONPATH=<site de mnemosys 1.0.1>`: registro acotado, fidelidad, dispositivo, warning ausente.

## Fuera de alcance
D2-viejo (checkpoints de febrero sin `wake_gate`/`sleep_mask`): no se toca. Publicar en PyPI y firmar el sello:
los ejecuta el titular (clave privada y credenciales).

## Criterio de aceptacion
Suite completa verde (banco de maestranza, ambito ampliado a checkpoint.py/config.py/engine.py/test_checkpoint.py);
ruff limpio; smoke con mnemosys real verde; `python -m build` + `twine check dist/*` OK; wheel instalable con
`--no-deps` en un target aislado e importable con `mztrain.__version__ == "1.4.0"`.
