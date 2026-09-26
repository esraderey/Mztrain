"""
MZTrain - Checkpointing de activaciones comprimido.

Implementa compresion de activaciones intermedias durante el forward pass,
descomprimiendolas on-demand durante el backward, ahorrando ~50-60% de
memoria de activaciones.
"""

import atexit
import logging
import math
import shutil
import tempfile
import warnings
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("mztrain")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from mneme import ZSpace, MnemeConfig, CompressionLevel
    HAS_MNEME = True
except ImportError:
    HAS_MNEME = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.checkpoint. "
        "Instalar con: pip install torch>=2.0.0"
    )


class ZActivationCheckpoint:
    """Checkpoint de activaciones comprimido via MNEME.

    Durante forward: comprime activaciones intermedias con INT8/LZ4.
    Durante backward: descomprime on-demand.

    Ahorro estimado: ~50-60% de memoria de activaciones.

    Usa ``torch.autograd.Function`` para integrarse con autograd.

    Example:
        >>> key, shape, dtype = ZActivationCheckpoint.compress_activation(tensor)
        >>> recovered = ZActivationCheckpoint.decompress_activation(key, shape, dtype)
    """

    # OPT-IN: el store ZSpace de MNEME/MNEMOSYS solo se usa si el caller lo pide
    # (ZTrainConfig.mneme_activation_store). Medido con mnemosys 1.0.1: ~0.2-0.6 s
    # por activacion (persiste a disco) frente a milisegundos del INT8 interno.
    prefer_zspace: bool = False
    _zspace: Optional['ZSpace'] = None
    _zspace_disabled_reason: Optional[str] = None
    _zspace_probed: bool = False
    _zspace_storage_dir: Optional[str] = None
    _FIDELITY_TOL: float = 0.05
    _counter: int = 0
    _stats: Dict[str, Any] = {
        "total_compressed": 0,
        "total_decompressed": 0,
        "bytes_saved": 0,
    }
    # Preserva device del tensor original para round-trip fiel
    # (MNEME ZSpace.load por defecto retorna en zspace.device)
    _device_map: Dict[str, "torch.device"] = {}

    @classmethod
    def _get_zspace(cls) -> Optional['ZSpace']:
        """Obtener o crear instancia de ZSpace para compresion."""
        if cls._zspace is None and cls.prefer_zspace and cls._zspace_disabled_reason is None:
            if not HAS_MNEME:
                cls._disable_zspace("MNEME no esta instalado (pip install \"mztrain[mneme]\")")
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
        except TypeError as error:
            # solo el rechazo del kwarg cae al register plano; otro TypeError es un fallo real
            if "quantization_type" not in str(error) and "unexpected keyword" not in str(error):
                raise
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
    def _probe_backend(cls, zspace, like: torch.Tensor) -> bool:
        """Sonda de fidelidad en el primer uso: register/load/evict de un tensor de prueba con
        la MISMA forma, dtype y device que la primera activacion real (el enrutado de algunos
        backends depende del tamano). Generador propio: no consume el RNG global del run.
        Un backend que corrompe el round-trip (o falla) se deshabilita con aviso."""
        if cls._zspace_probed:
            return True
        cls._zspace_probed = True
        device = like.device
        gen = torch.Generator(device=device).manual_seed(0)
        probe = torch.randn(like.shape, generator=gen, device=device, dtype=torch.float32)
        if like.dtype.is_floating_point:
            probe = probe.to(like.dtype)
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

    @classmethod
    def compress_activation(
        cls, tensor: torch.Tensor
    ) -> Tuple[str, torch.Size, torch.dtype]:
        """Comprimir activacion y registrar en ZSpace.

        Args:
            tensor: Tensor de activacion a comprimir.

        Returns:
            Tupla (clave, forma, dtype) para recuperar la activacion.
        """
        cls._counter += 1
        key = f"act_ckpt_{cls._counter}_{id(tensor)}"
        cls._device_map[key] = tensor.device  # preservar device original

        zspace = cls._get_zspace()
        if zspace is not None and not cls._probe_backend(zspace, tensor):
            zspace = None
        if zspace is not None:
            cls._register(zspace, key, tensor.detach().contiguous())
            cls._stats["total_compressed"] += 1
            cls._stats["bytes_saved"] += tensor.numel() * tensor.element_size()
            return key, tensor.shape, tensor.dtype
        else:
            # Fallback: cuantizacion INT8 block-wise (aisla outliers por bloque)
            block_size = 256
            flat = tensor.detach().contiguous().flatten()
            n = flat.numel()
            # Pad a multiplo de block_size
            n_blocks = (n + block_size - 1) // block_size
            if n % block_size != 0:
                padded = torch.zeros(n_blocks * block_size, device=flat.device, dtype=flat.dtype)
                padded[:n] = flat
            else:
                padded = flat
            blocks = padded.reshape(n_blocks, block_size)
            abs_maxes = blocks.abs().amax(dim=1).clamp(min=1e-12)  # (n_blocks,)
            scales = abs_maxes / 127.0  # Escala para usar rango completo INT8 [-127, 127]
            q = (blocks / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)

            if not hasattr(cls, "_fallback_store"):
                cls._fallback_store = {}
            cls._fallback_store[key] = (q, scales.to(torch.float16), n)
            cls._stats["total_compressed"] += 1
            # INT8 + FP16 scales vs original: ~30% del tamano original en FP32
            cls._stats["bytes_saved"] += tensor.numel() * tensor.element_size()
            return key, tensor.shape, tensor.dtype

    @classmethod
    def decompress_activation(
        cls, key: str, shape: torch.Size, dtype: torch.dtype
    ) -> torch.Tensor:
        """Descomprimir activacion desde ZSpace o fallback INT8.

        Args:
            key: Clave de la activacion comprimida.
            shape: Forma original del tensor.
            dtype: Tipo de dato original.

        Returns:
            Tensor descomprimido.

        Raises:
            RuntimeError: Si la clave no se encuentra.
        """
        original_device = cls._device_map.pop(key, None)
        zspace = cls._get_zspace()
        if zspace is not None:
            try:
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
            cls._stats["total_decompressed"] += 1
            result = tensor.to(dtype)
            # Liberar la entrada tras cargarla, simetrico al fallback (.pop): sin esto el
            # store retiene toda activacion del run -> fuga proporcional a steps. La
            # existencia de remove()/delete() se exige al construir el ZSpace.
            cls._evict_fn(zspace)(key)
            return result
        else:
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

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Obtener estadisticas de checkpointing."""
        return dict(cls._stats)

    @classmethod
    def reset(cls) -> None:
        """Resetear todo el estado del checkpoint."""
        cls._counter = 0
        cls._stats = {
            "total_compressed": 0,
            "total_decompressed": 0,
            "bytes_saved": 0,
        }
        cls._device_map.clear()
        cls._zspace = None
        cls._zspace_disabled_reason = None
        cls._zspace_probed = False
        if cls._zspace_storage_dir:
            shutil.rmtree(cls._zspace_storage_dir, ignore_errors=True)
            cls._zspace_storage_dir = None
        if hasattr(cls, "_fallback_store"):
            cls._fallback_store.clear()


class ZCheckpointFunction(torch.autograd.Function):
    """Autograd Function que comprime activaciones durante forward
    y las descomprime durante backward."""

    @staticmethod
    def forward(ctx, run_function, *args):
        with torch.no_grad():
            outputs = run_function(*args)

        arg_refs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.requires_grad:
                    key, shape, dtype = ZActivationCheckpoint.compress_activation(arg)
                    arg_refs.append(("compressed_tensor", key, shape, dtype))
                else:
                    # Preservar argumentos tensoriales auxiliares, como masks,
                    # en su posicion original para la recomputacion del backward.
                    arg_refs.append(("tensor", arg.detach()))
            else:
                arg_refs.append(("object", arg))

        ctx.run_function = run_function
        ctx.arg_refs = arg_refs

        return outputs

    @staticmethod
    def backward(ctx, *output_grads):
        inputs = []
        grad_positions = []

        for idx, ref in enumerate(ctx.arg_refs):
            kind = ref[0]
            if kind == "compressed_tensor":
                _, key, shape, dtype = ref
                tensor = ZActivationCheckpoint.decompress_activation(key, shape, dtype)
                tensor.requires_grad_(True)
                inputs.append(tensor)
                grad_positions.append(idx)
            elif kind == "tensor":
                inputs.append(ref[1])
            else:
                inputs.append(ref[1])

        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        torch.autograd.backward(outputs, output_grads)

        grads = [None] * len(ctx.arg_refs)
        for idx in grad_positions:
            inp = inputs[idx]
            grads[idx] = inp.grad

        return (None,) + tuple(grads)


def z_checkpoint(function, *args):
    """Wrapper conveniente para activation checkpointing comprimido.

    Args:
        function: Funcion a ejecutar con checkpointing.
        *args: Argumentos para la funcion.

    Returns:
        Resultado de la funcion con activaciones comprimidas.

    Example:
        >>> out = z_checkpoint(lambda x: model.layer(x), input_tensor)
    """
    return ZCheckpointFunction.apply(function, *args)
