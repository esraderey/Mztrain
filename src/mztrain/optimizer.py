"""
MZTrain - Optimizer con estados comprimidos.

Implementa ZCompressedAdam, un optimizer Adam que mantiene sus estados
(primer y segundo momento) comprimidos con cuantizacion INT8, reduciendo
la memoria del optimizer en ~75%.
"""

import math
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("mztrain")

try:
    import torch
    from torch.optim import Optimizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.optimizer. "
        "Instalar con: pip install torch>=2.0.0"
    )


class ZCompressedAdam(Optimizer):
    """Adam optimizer que mantiene estados (m, v) comprimidos con INT8.

    En Adam estandar, los estados m (1er momento) y v (2do momento)
    ocupan 2x el tamano del modelo. Para un modelo de 1B params:

    - Pesos:  4GB (FP32)
    - m:      4GB (FP32)
    - v:      4GB (FP32)
    - Total:  12GB solo en optimizer

    ZCompressedAdam comprime m y v con cuantizacion INT8, reduciendo
    a ~3GB total (75% de ahorro en estados del optimizer).

    Los estados se descomprimen durante el paso de update, se actualizan,
    y se re-comprimen periodicamente.

    Args:
        params: Parametros del modelo a optimizar.
        lr: Tasa de aprendizaje.
        betas: Coeficientes de momentos (beta1, beta2).
        eps: Epsilon para estabilidad numerica.
        weight_decay: Decaimiento de pesos (estilo AdamW).
        compress_states: Si comprimir estados del optimizer.
        compression_interval: Re-comprimir cada N steps.

    Example:
        >>> optimizer = ZCompressedAdam(model.parameters(), lr=3e-4)
        >>> optimizer.step()
        >>> print(optimizer.get_memory_stats())
    """

    # Block-wise INT8 quantization para los estados del optimizer.
    #
    # Para m (primer momento): cuantizacion lineal block-wise.
    # Para v (varianza, segundo momento): cuantizacion LOG block-wise. Razon:
    # v es no-negativa y muy sesgada (un outlier grande por capa, mayoria
    # tiny). Con INT8 lineal, los valores tiny dentro de un bloque que
    # contiene el outlier se redondean a 0, y como v va al denominador como
    # 1/(sqrt(v)+eps), cualquier cero hace que el step size salte a 1/eps =
    # 1e8 y causa divergencia. La cuantizacion logaritmica preserva precision
    # RELATIVA (no absoluta), que es exactamente lo que Adam necesita: el
    # paso es invariante a la escala de v. Con block_size=64 y log quant, el
    # error relativo por elemento es ~e^(20/127) ~= 1.17 (17% relativo),
    # suficiente para que sqrt(v) no colapse.
    #
    # Resultado: ~73% ahorro en optim states con training estable.
    _BLOCK_SIZE = 64
    _V_LOG_FLOOR = 1e-30  # piso para log(0); v=0 se mapea a log(1e-30)=-69

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        compress_states: bool = True,
        compression_interval: int = 5,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.compress_states = compress_states
        self.compression_interval = compression_interval
        self._step_count = 0

    def _compress_state(self, tensor: torch.Tensor) -> tuple:
        """Comprimir estado del optimizer a INT8 block-wise.

        Args:
            tensor: Tensor FP32 a comprimir (cualquier shape).

        Returns:
            Tupla (quantized_int8, scales_per_block, original_shape, original_numel).
        """
        flat = tensor.reshape(-1)
        n = flat.numel()
        bs = self._BLOCK_SIZE
        n_blocks = (n + bs - 1) // bs

        # Pad a multiplo de block_size para reshape exacto
        if n % bs != 0:
            padded = torch.zeros(n_blocks * bs, device=flat.device, dtype=flat.dtype)
            padded[:n] = flat
        else:
            padded = flat

        blocks = padded.reshape(n_blocks, bs)
        scales = blocks.abs().amax(dim=1)  # (n_blocks,)
        safe_scales = scales.clamp(min=1e-12)
        # Mapeo [-abs_max, abs_max] -> [-127, 127] usando todo el rango INT8.
        quantized = (
            (blocks / safe_scales.unsqueeze(1) * 127.0)
            .round().clamp(-127, 127).to(torch.int8)
        )

        return quantized, scales, tensor.shape, n

    def _decompress_state(self, compressed: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Descomprimir estado desde INT8 block-wise.

        Args:
            compressed: Tupla devuelta por _compress_state.
            dtype: Tipo de dato destino del tensor reconstruido.

        Returns:
            Tensor reconstruido con shape original.
        """
        quantized, scales, shape, orig_numel = compressed
        # Inversa de compress: dividir por 127 para deshacer el escalado INT8.
        blocks = quantized.to(dtype) * (scales.to(dtype) / 127.0).unsqueeze(1)
        flat = blocks.reshape(-1)[:orig_numel]
        return flat.reshape(shape)

    def _compress_v(self, v: torch.Tensor) -> tuple:
        """Cuantizacion logaritmica block-wise para Adam v.

        v >= 0 con distribucion muy sesgada. Cuantizamos log(v) linealmente,
        lo que da precision relativa uniforme — exactamente lo que Adam
        necesita ya que el step depende de v solo via sqrt(v) en el
        denominador.
        """
        # log(v) — piso para evitar -inf. Valores < piso se mapean al piso.
        safe_v = v.clamp(min=self._V_LOG_FLOOR)
        log_v = safe_v.log()
        # log_v esta en rango aproximado [-69, log(max_v)]. Lineal-INT8 OK.
        return self._compress_state(log_v)

    def _decompress_v(self, compressed: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Inversa de _compress_v: INT8 -> log_v -> exp -> v."""
        log_v = self._decompress_state(compressed, dtype)
        return log_v.exp()

    @torch.no_grad()
    def step(self, closure=None):
        """Paso de optimizacion con estados comprimidos.

        1. Descomprimir m, v (si estan comprimidos)
        2. Aplicar AdamW update
        3. Re-comprimir m, v (cada N steps)

        Args:
            closure: Closure que re-evalua el modelo y retorna el loss.

        Returns:
            Loss opcional del closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ZCompressedAdam no soporta gradientes sparse")

                state = self.state[p]

                # Inicializar estados
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["compressed"] = False

                state["step"] += 1

                # Descomprimir si necesario. m usa INT8 lineal, v usa INT8 log.
                if state["compressed"] and self.compress_states:
                    m = self._decompress_state(state["exp_avg"], p.dtype)
                    v = self._decompress_v(state["exp_avg_sq"], p.dtype)
                else:
                    m = state["exp_avg"]
                    v = state["exp_avg_sq"]

                # Weight decay (AdamW: decay antes del update)
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # Actualizar momentos
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                step = state["step"]
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Update parametros
                p.data.addcdiv_(m, denom, value=-step_size)

                # Re-comprimir estados periodicamente
                should_compress = (
                    self.compress_states
                    and self._step_count % self.compression_interval == 0
                )

                if should_compress:
                    state["exp_avg"] = self._compress_state(m)   # INT8 lineal
                    state["exp_avg_sq"] = self._compress_v(v)    # INT8 log
                    state["compressed"] = True
                else:
                    state["exp_avg"] = m
                    state["exp_avg_sq"] = v
                    state["compressed"] = False

        return loss

    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtener estadisticas de memoria del optimizer.

        Returns:
            Diccionario con metricas de uso de memoria.
        """
        total_state_bytes = 0
        total_full_bytes = 0
        compressed_count = 0
        total_count = 0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if len(state) == 0:
                    continue

                total_count += 1
                param_bytes = p.numel() * p.element_size()
                total_full_bytes += param_bytes * 2  # m + v en FP32

                if state.get("compressed", False):
                    # m y v ambos: INT8 (1 byte/elem) + scales FP32 (1 cada
                    # _BLOCK_SIZE elems).
                    n = p.numel()
                    bytes_per_state = n + (n // self._BLOCK_SIZE + 1) * 4
                    total_state_bytes += bytes_per_state * 2
                    compressed_count += 1
                else:
                    total_state_bytes += param_bytes * 2

        return {
            "total_params": total_count,
            "compressed_params": compressed_count,
            "state_memory_mb": total_state_bytes / (1024 * 1024),
            "full_memory_mb": total_full_bytes / (1024 * 1024),
            "memory_saved_pct": (
                (1 - total_state_bytes / max(total_full_bytes, 1)) * 100
            ),
            "step_count": self._step_count,
        }
