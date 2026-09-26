import pathlib

def patch(path, pairs):
    p = pathlib.Path(path); raw = p.read_bytes(); crlf = b"\r\n" in raw
    s = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old)); s = s.replace(old, new)
    if crlf: s = s.replace("\n", "\r\n")
    p.write_bytes(s.encode("utf-8")); print("ok", path)

ROOT = "D:/mztrain/"
# ---------------- checkpoint.py ----------------
patch(ROOT + "src/mztrain/checkpoint.py", [
 ("import logging\nfrom typing import Dict, Any, Optional, Tuple\n",
  "import logging\nimport math\nimport tempfile\nimport warnings\nfrom typing import Dict, Any, Optional, Tuple\n"),
 ("""    _zspace: Optional['ZSpace'] = None
    _counter: int = 0
""",
  """    # OPT-IN: el store ZSpace de MNEME/MNEMOSYS solo se usa si el caller lo pide
    # (ZTrainConfig.mneme_activation_store). Medido con mnemosys 1.0.1: ~0.2-0.6 s
    # por activacion (persiste a disco) frente a milisegundos del INT8 interno.
    prefer_zspace: bool = False
    _zspace: Optional['ZSpace'] = None
    _zspace_disabled_reason: Optional[str] = None
    _zspace_probed: bool = False
    _FIDELITY_TOL: float = 0.05
    _counter: int = 0
"""),
 ("""        if cls._zspace is None and HAS_MNEME:
            config = MnemeConfig()
            config.compression_level = CompressionLevel.ULTRA_FAST
            cls._zspace = ZSpace(config)
        return cls._zspace
""",
  """        if cls._zspace is None and HAS_MNEME and cls.prefer_zspace and cls._zspace_disabled_reason is None:
            config = MnemeConfig()
            config.compression_level = CompressionLevel.ULTRA_FAST
            if hasattr(config, "storage_path"):
                # mnemosys persiste cada tensor en storage_path (por defecto ./mneme_storage
                # en el cwd del usuario): usar un directorio temporal propio.
                config.storage_path = tempfile.mkdtemp(prefix="mztrain_act_ckpt_")
            zspace = ZSpace(config)
            if not callable(cls._evict_fn(zspace)):
                cls._disable_zspace(
                    "el backend MNEME no expone remove()/delete(): sin eviccion por clave el store "
                    "creceria con cada paso (fuga de memoria)"
                )
                return None
            cls._zspace = zspace
        return cls._zspace

    @staticmethod
    def _evict_fn(zspace):
        return getattr(zspace, "remove", None) or getattr(zspace, "delete", None)

    @classmethod
    def _disable_zspace(cls, reason: str) -> None:
        cls._zspace = None
        cls._zspace_disabled_reason = reason
        warnings.warn(
            f"ZActivationCheckpoint: store ZSpace deshabilitado, se usa el fallback INT8 ({reason})",
            RuntimeWarning,
            stacklevel=3,
        )

    @staticmethod
    def _register(zspace, key: str, tensor: torch.Tensor) -> None:
        try:
            # mnemosys >= 1.0: sin esto el enrutado automatico descompone tensorialmente
            # las activaciones grandes (error relativo ~0.9 medido); INT8 da ~6e-3.
            zspace.register(key, tensor, quantization_type="int8")
        except TypeError:
            zspace.register(key, tensor)

    @staticmethod
    def _load(zspace, key: str, device) -> torch.Tensor:
        try:
            tensor = zspace.load(key, device=device) if device is not None else zspace.load(key)
        except TypeError:
            tensor = zspace.load(key)  # backends sin kwarg device (mnemosys 1.0.x)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def _probe_backend(cls, zspace, device) -> bool:
        \"\"\"Sonda de fidelidad en el primer uso: register/load/evict de un tensor de prueba.
        Un backend que corrompe el round-trip (o falla) se deshabilita con aviso.\"\"\"
        if cls._zspace_probed:
            return True
        cls._zspace_probed = True
        probe = torch.randn(64, 64, device=device)
        key = f"__mztrain_probe_{id(probe)}"
        try:
            cls._register(zspace, key, probe)
            back = cls._load(zspace, key, device)
            cls._evict_fn(zspace)(key)
            rel = float((back.to(probe.device).float() - probe).norm() / probe.norm())
        except Exception as error:  # noqa: BLE001 - cualquier fallo del backend externo deshabilita el store
            cls._disable_zspace(f"el round-trip de prueba fallo: {type(error).__name__}: {error}")
            return False
        if not math.isfinite(rel) or rel > cls._FIDELITY_TOL:
            cls._disable_zspace(f"fidelidad insuficiente en el round-trip de prueba: error relativo {rel:.3g}")
            return False
        return True
"""),
 ("""        zspace = cls._get_zspace()
        if zspace is not None:
            zspace.register(key, tensor.detach().contiguous())
            cls._stats["total_compressed"] += 1
""",
  """        zspace = cls._get_zspace()
        if zspace is not None and not cls._probe_backend(zspace, tensor.device):
            zspace = None
        if zspace is not None:
            cls._register(zspace, key, tensor.detach().contiguous())
            cls._stats["total_compressed"] += 1
"""),
 ("""            try:
                # P12: pasar device= para respetar el device del tensor original
                if original_device is not None:
                    tensor = zspace.load(key, device=original_device)
                else:
                    tensor = zspace.load(key)
            except KeyError:
                # Re-raise como RuntimeError para contrato estable del API
                raise RuntimeError(f"Activation checkpoint not found: {key}")
            cls._stats["total_decompressed"] += 1
            result = tensor.to(dtype)
            # Liberar la entrada tras cargarla, simetrico al fallback (.pop):
            # sin esto el store retiene toda activacion del run -> fuga de
            # memoria proporcional a steps. Se asume la API de eviction
            # simetrica a register(); si el backend la nombra distinto, ajustar
            # aqui (el fallo seria visible, no una fuga silenciosa).
            _evict = getattr(zspace, "remove", None) or getattr(zspace, "delete", None)
            if callable(_evict):
                _evict(key)
            return result
""",
  """            try:
                # P12: respetar el device del tensor original (con o sin kwarg device)
                tensor = cls._load(zspace, key, original_device)
            except KeyError:
                # Re-raise como RuntimeError para contrato estable del API
                raise RuntimeError(f"Activation checkpoint not found: {key}")
            cls._stats["total_decompressed"] += 1
            result = tensor.to(dtype)
            # Liberar la entrada tras cargarla, simetrico al fallback (.pop): sin esto el
            # store retiene toda activacion del run -> fuga proporcional a steps. La
            # existencia de remove()/delete() se exige al construir el ZSpace.
            cls._evict_fn(zspace)(key)
            return result
"""),
 ("""        cls._device_map.clear()
        if cls._zspace is not None:
            cls._zspace = None
""",
  """        cls._device_map.clear()
        cls._zspace = None
        cls._zspace_disabled_reason = None
        cls._zspace_probed = False
"""),
])

# ---------------- config.py ----------------
patch(ROOT + "src/mztrain/config.py", [
 ("""    mneme_compression_level: CompressionLevel = CompressionLevel.BALANCED
    \"\"\"Nivel de compresion MNEME para activaciones.\"\"\"
""",
  """    mneme_compression_level: CompressionLevel = CompressionLevel.BALANCED
    \"\"\"Nivel de compresion MNEME para activaciones.\"\"\"

    mneme_activation_store: bool = False
    \"\"\"Usar el ZSpace de MNEME/MNEMOSYS como store de activaciones (opt-in).
    Requiere un backend con eviccion por clave (mnemosys>=1.0.1: pip install mztrain[mneme]).
    Coste medido con mnemosys 1.0.1: ~0.2-0.6 s por activacion (persiste a disco); el
    fallback INT8 interno (False) cuesta milisegundos.\"\"\"
"""),
])

# ---------------- engine.py ----------------
patch(ROOT + "src/mztrain/engine.py", [
 ("""        checkpointed = 0
        layer_idx = 0

        # Buscar bloques transformer o modulos Sequential significativos
""",
  """        checkpointed = 0
        layer_idx = 0
        # Store ZSpace (MNEME/MNEMOSYS) solo si el usuario lo pide; por defecto INT8 interno.
        ZActivationCheckpoint.prefer_zspace = bool(getattr(self.config, "mneme_activation_store", False))

        # Buscar bloques transformer o modulos Sequential significativos
"""),
])
