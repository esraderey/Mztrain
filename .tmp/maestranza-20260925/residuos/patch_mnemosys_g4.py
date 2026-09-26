"""Contra-arreglos del G4 ciego (lote MNEMOSYS)."""
import pathlib, re

def patch(path, pairs):
    p = pathlib.Path(path); raw = p.read_bytes(); crlf = b"\r\n" in raw
    s = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old)); s = s.replace(old, new)
    if crlf: s = s.replace("\n", "\r\n")
    p.write_bytes(s.encode("utf-8")); print("ok", path)

R = "D:/mztrain/"
patch(R + "src/mztrain/checkpoint.py", [
 ("import logging\nimport math\nimport tempfile\nimport warnings\n",
  "import atexit\nimport logging\nimport math\nimport shutil\nimport tempfile\nimport warnings\n"),
 ("""    _zspace_disabled_reason: Optional[str] = None
    _zspace_probed: bool = False
""",
  """    _zspace_disabled_reason: Optional[str] = None
    _zspace_probed: bool = False
    _zspace_storage_dir: Optional[str] = None
"""),
 ("""        if cls._zspace is None and HAS_MNEME and cls.prefer_zspace and cls._zspace_disabled_reason is None:
            config = MnemeConfig()
            config.compression_level = CompressionLevel.ULTRA_FAST
            if hasattr(config, "storage_path"):
                # mnemosys persiste cada tensor en storage_path (por defecto ./mneme_storage
                # en el cwd del usuario): usar un directorio temporal propio.
                config.storage_path = tempfile.mkdtemp(prefix="mztrain_act_ckpt_")
            if hasattr(config, "enable_encryption"):
                # activaciones efimeras en un directorio temporal: cifrarlas en reposo solo
                # anade coste (y mnemosys avisa si no hay secret_key).
                config.enable_encryption = False
            zspace = ZSpace(config)
            if not callable(cls._evict_fn(zspace)):
                cls._disable_zspace(
                    "el backend MNEME no expone remove()/delete(): sin eviccion por clave el store "
                    "creceria con cada paso (fuga de memoria)"
                )
                return None
            cls._zspace = zspace
        return cls._zspace
""",
  """        if cls._zspace is None and cls.prefer_zspace and cls._zspace_disabled_reason is None:
            if not HAS_MNEME:
                cls._disable_zspace("MNEME no esta instalado (pip install \\"mztrain[mneme]\\")")
                return None
            storage_dir = tempfile.mkdtemp(prefix="mztrain_act_ckpt_")
            try:
                # activaciones efimeras: sin cifrado en reposo (mnemosys avisa en __post_init__
                # si el cifrado esta activo sin secret_key, por eso va en el constructor).
                try:
                    config = MnemeConfig(enable_encryption=False)
                except TypeError:
                    config = MnemeConfig()
                config.compression_level = CompressionLevel.ULTRA_FAST
                if hasattr(config, "storage_path"):
                    # mnemosys persiste cada tensor en storage_path (por defecto ./mneme_storage
                    # en el cwd del usuario): usar un directorio temporal propio.
                    config.storage_path = storage_dir
                zspace = ZSpace(config)
            except Exception as error:  # noqa: BLE001 - backend externo: cualquier fallo deshabilita el store
                shutil.rmtree(storage_dir, ignore_errors=True)
                cls._disable_zspace(f"no se pudo construir ZSpace: {type(error).__name__}: {error}")
                return None
            if not callable(cls._evict_fn(zspace)):
                shutil.rmtree(storage_dir, ignore_errors=True)
                cls._disable_zspace(
                    "el backend MNEME no expone remove()/delete(): sin eviccion por clave el store "
                    "creceria con cada paso (fuga de memoria)"
                )
                return None
            cls._zspace = zspace
            cls._zspace_storage_dir = storage_dir
            atexit.register(shutil.rmtree, storage_dir, ignore_errors=True)
        return cls._zspace
"""),
 ("""    @staticmethod
    def _register(zspace, key: str, tensor: torch.Tensor) -> None:
        try:
            # mnemosys >= 1.0: sin esto el enrutado automatico descompone tensorialmente
            # las activaciones grandes (error relativo ~0.9 medido); INT8 da ~6e-3.
            zspace.register(key, tensor, quantization_type="int8")
        except TypeError:
            zspace.register(key, tensor)
""",
  """    @staticmethod
    def _register(zspace, key: str, tensor: torch.Tensor) -> None:
        try:
            # mnemosys >= 1.0: sin esto el enrutado automatico descompone tensorialmente
            # las activaciones grandes (error relativo ~0.9 medido); INT8 da ~6e-3.
            zspace.register(key, tensor, quantization_type="int8")
        except TypeError as error:
            # solo el rechazo del kwarg cae al register plano; otro TypeError es un fallo real
            if "quantization_type" not in str(error) and "unexpected keyword" not in str(error):
                raise
            zspace.register(key, tensor)
"""),
 ("""    @classmethod
    def _probe_backend(cls, zspace, device) -> bool:
        \"\"\"Sonda de fidelidad en el primer uso: register/load/evict de un tensor de prueba.
        Un backend que corrompe el round-trip (o falla) se deshabilita con aviso.\"\"\"
        if cls._zspace_probed:
            return True
        cls._zspace_probed = True
        probe = torch.randn(64, 64, device=device)
        key = f"__mztrain_probe_{id(probe)}"
""",
  """    @classmethod
    def _probe_backend(cls, zspace, like: torch.Tensor) -> bool:
        \"\"\"Sonda de fidelidad en el primer uso: register/load/evict de un tensor de prueba con
        la MISMA forma, dtype y device que la primera activacion real (el enrutado de algunos
        backends depende del tamano). Generador propio: no consume el RNG global del run.
        Un backend que corrompe el round-trip (o falla) se deshabilita con aviso.\"\"\"
        if cls._zspace_probed:
            return True
        cls._zspace_probed = True
        device = like.device
        gen = torch.Generator(device=device).manual_seed(0)
        probe = torch.randn(like.shape, generator=gen, device=device, dtype=torch.float32)
        if like.dtype.is_floating_point:
            probe = probe.to(like.dtype)
        key = f"__mztrain_probe_{id(probe)}"
"""),
 ("""        zspace = cls._get_zspace()
        if zspace is not None and not cls._probe_backend(zspace, tensor.device):
            zspace = None
""",
  """        zspace = cls._get_zspace()
        if zspace is not None and not cls._probe_backend(zspace, tensor):
            zspace = None
"""),
 ("""            try:
                # P12: respetar el device del tensor original (con o sin kwarg device)
                tensor = cls._load(zspace, key, original_device)
            except KeyError:
                # Re-raise como RuntimeError para contrato estable del API
                raise RuntimeError(f"Activation checkpoint not found: {key}") from None
""",
  """            try:
                # P12: respetar el device del tensor original (con o sin kwarg device)
                tensor = cls._load(zspace, key, original_device)
            except KeyError:
                fallback = getattr(cls, "_fallback_store", {})
                if key in fallback:
                    # la activacion se comprimio con INT8 antes de habilitar el store
                    cls._device_map[key] = original_device
                    return cls._decompress_fallback(key, shape, dtype)
                # Re-raise como RuntimeError para contrato estable del API
                raise RuntimeError(f"Activation checkpoint not found: {key}") from None
"""),
 ("""        else:
            if hasattr(cls, "_fallback_store") and key in cls._fallback_store:
                q, scales, orig_numel = cls._fallback_store.pop(key)
                # Descomprimir block-wise y reshape a forma original
                blocks = q.float() * scales.float().unsqueeze(1)
                tensor = blocks.flatten()[:orig_numel].reshape(shape).to(dtype)
                if original_device is not None and tensor.device != original_device:
                    tensor = tensor.to(original_device)
                cls._stats["total_decompressed"] += 1
                return tensor
            raise RuntimeError(f"Activation checkpoint not found: {key}")
""",
  """        else:
            if hasattr(cls, "_fallback_store") and key in cls._fallback_store:
                cls._device_map[key] = original_device
                return cls._decompress_fallback(key, shape, dtype)
            raise RuntimeError(f"Activation checkpoint not found: {key}")

    @classmethod
    def _decompress_fallback(cls, key: str, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        original_device = cls._device_map.pop(key, None)
        q, scales, orig_numel = cls._fallback_store.pop(key)
        # Descomprimir block-wise y reshape a forma original
        blocks = q.float() * scales.float().unsqueeze(1)
        tensor = blocks.flatten()[:orig_numel].reshape(shape).to(dtype)
        if original_device is not None and tensor.device != original_device:
            tensor = tensor.to(original_device)
        cls._stats["total_decompressed"] += 1
        return tensor
"""),
 ("""        cls._device_map.clear()
        cls._zspace = None
        cls._zspace_disabled_reason = None
        cls._zspace_probed = False
""",
  """        cls._device_map.clear()
        cls._zspace = None
        cls._zspace_disabled_reason = None
        cls._zspace_probed = False
        if cls._zspace_storage_dir:
            shutil.rmtree(cls._zspace_storage_dir, ignore_errors=True)
            cls._zspace_storage_dir = None
"""),
])

patch(R + "src/mztrain/config.py", [
 ("""    mneme_activation_store: bool = False
    \"\"\"Usar el ZSpace de MNEME/MNEMOSYS como store de activaciones (opt-in).
    Requiere un backend con eviccion por clave (mnemosys>=1.0.1: pip install mztrain[mneme]).
    Coste medido con mnemosys 1.0.1: ~0.2-0.6 s por activacion (persiste a disco); el
    fallback INT8 interno (False) cuesta milisegundos.\"\"\"
""",
  """    mneme_activation_store: bool = False
    \"\"\"Usar el ZSpace de MNEME/MNEMOSYS como store de activaciones (opt-in).
    Requiere un backend con eviccion por clave (mnemosys>=1.0.1: pip install mztrain[mneme]);
    sin el se avisa y se usa el INT8 interno. Coste medido con mnemosys 1.0.1: ~0.2-0.6 s por
    activacion (persiste a disco en un directorio temporal que se limpia al salir); el fallback
    INT8 interno (False) cuesta milisegundos. Es un ajuste GLOBAL del proceso
    (ZActivationCheckpoint.prefer_zspace): el ultimo engine construido manda.\"\"\"
"""),
])

patch(R + "pyproject.toml", [('    "mnemosys>=1.0.1",\n', '    "mnemosys>=1.0.1,<2",\n')])
patch(R + "setup.py", [('            "mnemosys>=1.0.1",\n', '            "mnemosys>=1.0.1,<2",\n')])
patch(R + "requirements.txt", [("# mnemosys>=1.0.1\n", "# mnemosys>=1.0.1,<2\n")])

# CHANGELOG: encabezado duplicado de ElasticShape
cl = pathlib.Path(R + "CHANGELOG.md"); raw = cl.read_bytes(); s = raw.decode("utf-8").replace("\r\n", "\n")
dup = "### ElasticShape: cirugia reversible y migracion AdamW\n\n### ElasticShape: cirugia reversible y migracion AdamW\n"
if dup in s:
    s = s.replace(dup, "### ElasticShape: cirugia reversible y migracion AdamW\n"); print("CHANGELOG: encabezado duplicado eliminado")
else:
    m = re.findall(r"### ElasticShape: cirugia reversible y migracion AdamW", s); print("CHANGELOG: apariciones del encabezado:", len(m))
s = s.replace("  temporal propio y desactiva el cifrado en reposo para ese store efimero). Cierra la fuga de activaciones\n",
              "  temporal propio, limpiado al salir, y sin cifrado en reposo para ese store efimero; extra acotado a\n  `mnemosys<2`). Cierra la fuga de activaciones\n")
cl.write_bytes(s.replace("\n", "\r\n").encode("utf-8")); print("ok CHANGELOG")

# README: restaurar los finales de linea originales (HEAD tenia CRLF y LF mezclados) aplicando los cambios sobre HEAD
import subprocess
head = subprocess.run(["git", "-C", R, "show", "HEAD:README.md"], capture_output=True).stdout.decode("utf-8")
cur = pathlib.Path(R + "README.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
head_norm = head.replace("\r\n", "\n")
# diff por bloques: sustituir en HEAD (con sus EOL) los tres pasajes editados, usando el EOL de cada pasaje
def eol_replace(text, old_norm, new_norm):
    pat = re.compile(re.escape(old_norm).replace(r"\n", r"\r?\n"))
    m = pat.search(text); assert m, old_norm[:60]
    eol = "\r\n" if "\r\n" in m.group(0) else "\n"
    return text[:m.start()] + new_norm.replace("\n", eol) + text[m.end():]
pairs = [
 ("```bash\npip install mztrain            # desde PyPI\n```\n",
  "```bash\npip install mztrain            # desde PyPI\npip install \"mztrain[mneme]\"   # + MNEME (paquete PyPI `mnemosys`, import `mneme`)\n```\n"),
]
# los otros dos pasajes: tomarlos del README actual normalizado
i0 = cur.index("### MNEME / MNEMOSYS"); i1 = cur.index("## Inicio r", i0); nuevo_mneme = cur[i0:i1]
j0 = head_norm.index("### MNEME (backend opcional"); j1 = head_norm.index("## Inicio r", j0); viejo_mneme = head_norm[j0:j1]
pairs.append((viejo_mneme, nuevo_mneme))
k0 = cur.index("- [MNEME / MNEMOSYS"); k1 = cur.index("MNEME comprime por bits", k0) + len("MNEME comprime por bits"); nuevo_rel = cur[k0:k1]
l0 = head_norm.index("- [MNEME "); l1 = head_norm.index("MNEME comprime por bits", l0) + len("MNEME comprime por bits"); viejo_rel = head_norm[l0:l1]
pairs.append((viejo_rel, nuevo_rel))
out = head
for old, new in pairs: out = eol_replace(out, old, new)
# frase extra: ZCodeBERT fuera del engine
out = eol_replace(out, "  y hace una sonda de fidelidad en el primer uso; si algo falla, avisa y cae al INT8\n  interno en vez de fugar memoria.\n",
  "  y hace una sonda de fidelidad en el primer uso; si algo falla, avisa y cae al INT8\n  interno en vez de fugar memoria. Fuera de `ZTrainEngine` (p. ej. `ZCodeBERT` con\n  `z_checkpoint`), el opt-in se activa con `ZActivationCheckpoint.prefer_zspace = True`.\n")
pathlib.Path(R + "README.md").write_bytes(out.encode("utf-8")); print("ok README (EOL originales)")
