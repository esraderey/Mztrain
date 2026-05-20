"""
MZTrain - Multi-Precision Manager (T8).

Gestion de precision mixta multi-nivel que va mas alla de AMP estandar:

- STANDARD: FP16/BF16 forward via AMP, FP32 backward (lo que ZTrain hace hoy)
- AGGRESSIVE: FP16 forward + INT8 activaciones + INT8 optimizer states
- FP8: E4M3 forward, E5M2 backward, FP32 master weights (solo Hopper SM90+)

LIMITACION IMPORTANTE:
- RTX 4060 es SM89 (Ada Lovelace). FP8 nativo requiere SM90+ (Hopper).
- En SM89, FP8 se emula en software (PEOR rendimiento que FP16).
- El nivel "aggressive" es el mas beneficioso para Ada Lovelace.
- FP8 se implementa para forward-compatibility con H100/H200.

REGLA CRITICA: Singular values S NUNCA en baja precision.
Los valores singulares controlan la escala de toda la capa factorizada.
Cuantizarlos causa inestabilidad catastrofica. SIEMPRE FP32.

Referencias:
- COAT: ICLR 2025 — FP8 optimizer + FP8 activaciones simultaneos
- MOSS: arXiv:2511.05811 — Two-level microscaling, 34% throughput
- Quartet: NeurIPS 2025 (arXiv:2505.14669) — FP4 end-to-end training
- NVIDIA Transformer Engine: FP8 E4M3/E5M2 para Hopper
- 8-bit Adam (bitsandbytes): Dettmers et al., ICLR 2022
"""

import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger("mztrain")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.precision. "
        "Instalar con: pip install torch>=2.0.0"
    )


# ============================================================================
# Constantes de FP8
# ============================================================================

# E4M3: 4 bits exponente, 3 bits mantissa. Rango [-448, 448]. Precision alta.
FP8_E4M3_MAX = 448.0
# E5M2: 5 bits exponente, 2 bits mantissa. Rango [-57344, 57344]. Rango amplio.
FP8_E5M2_MAX = 57344.0


class ZMultiPrecisionManager:
    """Gestion de precision mixta multi-nivel para MZTrain.

    Encapsula la logica de precision para forward/backward pass,
    cuantizacion de activaciones, y cuantizacion de gradientes.

    Niveles:
        "standard": AMP FP16 estandar. Sin cuantizacion adicional.
        "aggressive": AMP FP16 + INT8 activaciones + INT8 optimizer states.
            La agresividad esta en optimizer/activaciones, no en compute.
        "fp8": FP8 E4M3 forward + E5M2 backward. Solo Hopper+ (SM90+).
            Fallback automatico a "aggressive" si hardware no soporta.

    Args:
        precision_level: Nivel de precision ("standard", "aggressive", "fp8").
        device: Dispositivo de computo.

    Example:
        >>> manager = ZMultiPrecisionManager("aggressive", torch.device("cuda"))
        >>> with manager.forward_context():
        ...     output = model(input)
        >>> print(manager.get_stats())
    """

    # Niveles validos
    VALID_LEVELS = ("standard", "aggressive", "fp8")

    def __init__(
        self,
        precision_level: str = "standard",
        device: Optional[torch.device] = None,
    ):
        if precision_level not in self.VALID_LEVELS:
            raise ValueError(
                f"precision_level debe ser uno de {self.VALID_LEVELS}, "
                f"recibido: {precision_level}"
            )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._requested_level = precision_level

        # Detectar capacidad del hardware.
        #
        # FP8 GEMM via torch._scaled_mm requiere FP8 tensor cores. Disponibles:
        #   - SM89 (Ada Lovelace): RTX 4060/4070/4080/4090, L4, L40 — FP8 OK,
        #     speedup limitado vs Hopper (no full FP8 accumulator).
        #   - SM90+ (Hopper, Blackwell): H100, H200, B100 — FP8 con max speedup.
        # En PyTorch 2.6+ torch._scaled_mm acepta float8_e4m3fn como input y
        # bfloat16 como output, soportado en SM89+.
        self.has_fp8 = False
        self.compute_capability = (0, 0)
        if self.device.type == "cuda":
            self.compute_capability = torch.cuda.get_device_capability(self.device)
            sm = self.compute_capability[0] * 10 + self.compute_capability[1]
            self.has_fp8 = sm >= 89  # Ada Lovelace+

        # Detectar si BF16 esta disponible (Ampere+, SM80+)
        self.has_bf16 = (
            self.device.type == "cuda"
            and self.compute_capability[0] >= 8
        )

        # Fallback si hardware no soporta FP8
        if precision_level == "fp8" and not self.has_fp8:
            logger.warning(
                f"[MZTrain Precision] FP8 requiere SM89+ (Ada Lovelace+). "
                f"Hardware detectado: SM{self.compute_capability[0]}{self.compute_capability[1]}. "
                f"Fallback a 'aggressive'."
            )
            self.level = "aggressive"
        else:
            self.level = precision_level

        # Contadores para stats
        self._forward_count = 0
        self._quantize_activation_count = 0
        self._quantize_gradient_count = 0

        logger.info(
            f"[MZTrain Precision] Nivel: {self.level} "
            f"(solicitado: {self._requested_level}), "
            f"Device: {self.device}, "
            f"SM: {self.compute_capability[0]}{self.compute_capability[1]}, "
            f"FP8: {'si' if self.has_fp8 else 'no'}, "
            f"BF16: {'si' if self.has_bf16 else 'no'}"
        )

    @property
    def forward_dtype(self) -> torch.dtype:
        """Dtype para forward pass."""
        if self.level == "fp8" and self.has_fp8:
            return torch.float8_e4m3fn
        elif self.has_bf16:
            return torch.bfloat16
        else:
            return torch.float16

    @property
    def backward_dtype(self) -> torch.dtype:
        """Dtype para backward pass / gradientes."""
        if self.level == "fp8" and self.has_fp8:
            return torch.float8_e5m2
        else:
            return torch.float32

    @contextmanager
    def forward_context(self):
        """Context manager para forward pass con autocast apropiado.

        En todos los niveles, usa AMP autocast con el dtype apropiado.
        La diferencia entre niveles esta en activaciones y optimizer,
        no en el compute del forward.

        Yields:
            Context con autocast habilitado.
        """
        self._forward_count += 1

        if self.device.type != "cuda":
            yield
            return

        if self.level == "fp8" and self.has_fp8:
            # FP8 autocast — usa float16 como fallback seguro
            # (torch.amp.autocast no soporta fp8 directamente como dtype)
            # En practica, FP8 se aplica via per-tensor scaling, no via autocast
            with torch.amp.autocast("cuda", dtype=torch.float16):
                yield
        else:
            # Standard y Aggressive: AMP con BF16 o FP16
            compute_dtype = torch.bfloat16 if self.has_bf16 else torch.float16
            with torch.amp.autocast("cuda", dtype=compute_dtype):
                yield

    def quantize_activation(self, tensor: torch.Tensor) -> tuple:
        """Cuantizar activacion para checkpointing comprimido.

        Solo activo en niveles "aggressive" y "fp8".
        Usa INT8 block-wise (funciona en cualquier GPU).

        REGLA: Si el tensor contiene singular values (1D, < 1024 elementos),
        NO cuantizar — retornar tal cual.

        Args:
            tensor: Activacion a cuantizar.

        Returns:
            Tupla (quantized_data, metadata) para descomprimir despues.
            En nivel "standard", retorna (tensor, None) sin cuantizar.
        """
        if self.level == "standard":
            return tensor, None

        # Singular values protection: tensores 1D pequenos = probables S values
        if tensor.dim() == 1 and tensor.numel() < 1024:
            return tensor, None

        self._quantize_activation_count += 1
        return self._int8_quantize(tensor)

    def dequantize_activation(
        self, data: torch.Tensor, metadata: Optional[dict]
    ) -> torch.Tensor:
        """Descomprimir activacion cuantizada.

        Args:
            data: Datos cuantizados (o tensor original si metadata=None).
            metadata: Metadatos de cuantizacion (None si no fue cuantizado).

        Returns:
            Tensor descomprimido en dtype original.
        """
        if metadata is None:
            return data
        return self._int8_dequantize(data, metadata)

    def quantize_gradient_for_comm(self, gradient: torch.Tensor) -> tuple:
        """Cuantizar gradiente para comunicacion (FP8 E5M2).

        Solo activo en nivel "fp8" con hardware compatible.
        E5M2 tiene rango mas amplio (5 bits exp) — mejor para gradientes
        que pueden tener spikes.

        Args:
            gradient: Gradiente a cuantizar.

        Returns:
            Tupla (quantized_data, metadata) para descomprimir.
            En niveles sin FP8, retorna (gradient, None).
        """
        if self.level != "fp8" or not self.has_fp8:
            return gradient, None

        self._quantize_gradient_count += 1
        return self._fp8_quantize(gradient, max_val=FP8_E5M2_MAX)

    def dequantize_gradient(
        self, data: torch.Tensor, metadata: Optional[dict]
    ) -> torch.Tensor:
        """Descomprimir gradiente cuantizado.

        Args:
            data: Datos cuantizados (o gradiente original si metadata=None).
            metadata: Metadatos de cuantizacion.

        Returns:
            Gradiente descomprimido en FP32.
        """
        if metadata is None:
            return data
        return self._fp8_dequantize(data, metadata)

    def should_compress_optimizer_states(self) -> bool:
        """Indica si el optimizer deberia comprimir estados.

        En niveles "aggressive" y "fp8", siempre comprimir.
        En "standard", respetar la config del usuario.
        """
        return self.level in ("aggressive", "fp8")

    # === INT8 Block-Wise Quantization ===

    _INT8_BLOCK_SIZE = 2048

    def _int8_quantize(self, tensor: torch.Tensor) -> tuple:
        """INT8 block-wise quantization para activaciones.

        Divide el tensor en bloques, calcula abs_max por bloque,
        escala a [-127, 127] y cuantiza a INT8.

        Args:
            tensor: Tensor a cuantizar.

        Returns:
            Tupla (quantized_int8, metadata_dict).
        """
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        flat = tensor.float().reshape(-1)
        n = flat.numel()
        bs = self._INT8_BLOCK_SIZE
        n_blocks = (n + bs - 1) // bs

        # Pad
        if n % bs != 0:
            padded = torch.zeros(n_blocks * bs, device=flat.device, dtype=torch.float32)
            padded[:n] = flat
        else:
            padded = flat

        blocks = padded.reshape(n_blocks, bs)
        abs_max = blocks.abs().amax(dim=1).clamp(min=1e-12)  # (n_blocks,)
        scales = abs_max / 127.0  # Escalar al rango INT8 completo [-127, 127]
        quantized = (blocks / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)

        metadata = {
            "scales": scales,
            "shape": original_shape,
            "dtype": original_dtype,
            "numel": n,
        }
        return quantized, metadata

    def _int8_dequantize(self, quantized: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Descomprimir desde INT8 block-wise."""
        scales = metadata["scales"]
        shape = metadata["shape"]
        dtype = metadata["dtype"]
        n = metadata["numel"]

        blocks = quantized.float() * scales.unsqueeze(1)
        flat = blocks.reshape(-1)[:n]
        return flat.reshape(shape).to(dtype)

    # === FP8 Per-Tensor Quantization ===

    def _fp8_quantize(
        self, tensor: torch.Tensor, max_val: float = FP8_E4M3_MAX
    ) -> tuple:
        """FP8 per-tensor quantization con dynamic scaling.

        Per-tensor scaling obligatorio: E4M3 tiene rango [-448, 448].
        Sin scaling, overflow es comun.

        Usa stochastic rounding para evitar sesgo acumulado.

        Args:
            tensor: Tensor a cuantizar.
            max_val: Valor maximo del formato FP8 (448 para E4M3, 57344 para E5M2).

        Returns:
            Tupla (scaled_tensor, metadata_dict).
            Nota: retorna en float16/float32 ya que FP8 ops no son nativos en Ada.
        """
        original_shape = tensor.shape
        original_dtype = tensor.dtype

        # Dynamic per-tensor scale factor
        abs_max = tensor.abs().max().clamp(min=1e-12)
        scale = abs_max / max_val

        # Scale + stochastic rounding
        scaled = tensor / scale
        noise = torch.rand_like(scaled) - 0.5
        quantized = (scaled + noise).round().clamp(-max_val, max_val)

        # En hardware sin FP8 nativo, almacenar como float16 (emulacion)
        if not self.has_fp8:
            quantized = quantized.to(torch.float16)

        metadata = {
            "scale": scale,
            "shape": original_shape,
            "dtype": original_dtype,
            "max_val": max_val,
        }
        return quantized, metadata

    def _fp8_dequantize(self, quantized: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Descomprimir desde FP8."""
        scale = metadata["scale"]
        dtype = metadata["dtype"]
        return (quantized.float() * scale).to(dtype)

    def get_stats(self) -> dict:
        """Estadisticas del precision manager.

        Returns:
            Dict con nivel activo, hardware info, y contadores.
        """
        return {
            "level": self.level,
            "requested_level": self._requested_level,
            "compute_capability": f"SM{self.compute_capability[0]}{self.compute_capability[1]}",
            "has_fp8": self.has_fp8,
            "has_bf16": self.has_bf16,
            "forward_dtype": str(self.forward_dtype),
            "backward_dtype": str(self.backward_dtype),
            "forward_passes": self._forward_count,
            "activations_quantized": self._quantize_activation_count,
            "gradients_quantized": self._quantize_gradient_count,
        }

    def __repr__(self) -> str:
        return (
            f"ZMultiPrecisionManager(level='{self.level}', "
            f"device={self.device}, "
            f"SM={self.compute_capability[0]}{self.compute_capability[1]})"
        )


# ============================================================================
# FP8 GEMM helper (modulo-level, usable desde layers.py sin un manager)
# ============================================================================

class _FP8LinearAutograd(torch.autograd.Function):
    """Autograd wrapper para FP8 linear forward + BF16 backward.

    PyTorch 2.6 no implementa la derivada de torch._scaled_mm. Esta funcion
    custom hace el forward en FP8 (rapido) y el backward en BF16 (con gradientes
    exactos y sin compounding de error de cuantizacion).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype):
        # Guardar tensores en BF16 para backward (no FP8: backward necesita
        # mas precision para no acumular error a traves de capas).
        ctx.save_for_backward(x.to(torch.bfloat16), weight.to(torch.bfloat16))
        ctx.x_orig_dtype = x.dtype

        return _fp8_gemm(x, weight, out_dtype=out_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_bf, w_bf = ctx.saved_tensors
        # Gradients en BF16: equivalente a F.linear backward.
        # F.linear: y = x @ w.T  =>  dy/dx = grad_y @ w  ;  dy/dw = grad_y.T @ x
        grad_out_bf = grad_output.to(torch.bfloat16)

        # Reshape a 2D si grad_out tiene mas dims (matchea forward).
        orig_shape = grad_out_bf.shape
        if grad_out_bf.dim() > 2:
            grad_out_2d = grad_out_bf.reshape(-1, orig_shape[-1])
            x_2d = x_bf.reshape(-1, x_bf.shape[-1])
        else:
            grad_out_2d = grad_out_bf
            x_2d = x_bf

        grad_x = grad_out_2d @ w_bf  # (M, N) @ (N, K) = (M, K)
        grad_w = grad_out_2d.t() @ x_2d  # (N, M) @ (M, K) = (N, K)

        if len(orig_shape) > 2:
            grad_x = grad_x.reshape(*orig_shape[:-1], -1)

        return grad_x.to(ctx.x_orig_dtype), grad_w, None


def _fp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Implementacion del forward FP8 GEMM (sin autograd)."""
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, orig_shape[-1])

    x_bf = x.to(torch.bfloat16)
    w_bf = weight.to(torch.bfloat16)

    x_amax = x_bf.abs().max().clamp(min=1e-12).float()
    w_amax = w_bf.abs().max().clamp(min=1e-12).float()
    x_scale = x_amax / FP8_E4M3_MAX
    w_scale = w_amax / FP8_E4M3_MAX

    x_fp8 = (x_bf / x_scale.to(torch.bfloat16)).to(torch.float8_e4m3fn)
    w_fp8 = (w_bf / w_scale.to(torch.bfloat16)).to(torch.float8_e4m3fn)

    out = torch._scaled_mm(
        x_fp8,
        w_fp8.t(),
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=out_dtype,
    )

    if len(orig_shape) > 2:
        out = out.reshape(*orig_shape[:-1], -1)
    return out


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Matmul x @ weight.T con FP8 e4m3 forward + BF16 backward.

    Equivalente numerico aproximado de F.linear(x, weight) pero ejecutado
    en FP8 tensor cores (SM89+). Pesos maestros en BF16/FP32 se conservan;
    solo el GEMM forward cae a FP8.

    Wrappeado por torch.autograd.Function para que backward funcione (PyTorch
    2.6 no implementa el adjoint de _scaled_mm directamente).

    Args:
        x: Tensor de entrada (..., in_features).
        weight: Tensor de pesos (out_features, in_features).
        out_dtype: Dtype de salida del forward (bfloat16 por default).

    Returns:
        Tensor (..., out_features) en out_dtype.

    Notas:
      - Per-tensor amax scaling (delayed scaling es mas eficiente pero
        requiere tracker de estado fuera del kernel).
      - Backward en BF16 (no FP8): mantener gradientes en mayor precision
        evita compounding de error a traves de capas profundas.
      - En SM89 (Ada Lovelace) es funcionalmente correcto pero el speedup
        es menor que en Hopper SM90+ (que tiene FP8 accumulator de 32-bit).
    """
    return _FP8LinearAutograd.apply(x, weight, out_dtype)


def fp8_supported(device: Optional[torch.device] = None) -> bool:
    """Indica si el device soporta FP8 GEMM via torch._scaled_mm (SM89+)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return False
    cap = torch.cuda.get_device_capability(device)
    return cap[0] * 10 + cap[1] >= 89
