"""
MZTrain - Compresion de gradientes con error feedback.

Implementa multiples estrategias de compresion de gradientes para reducir
uso de memoria y comunicacion en entrenamiento distribuido.

Mejoras sobre la version original:
- Filtro de tamano minimo: no comprimir tensores pequenos (biases, norms).
- Error feedback con buffer comprimido (ConEF): reduce overhead de ~100% a ~25%.
- Block-wise INT8: cuantizacion por bloques en vez de global.
- Random-K: muestreo aleatorio de gradientes (compresor insesgado, sin EF).
- Estadisticas detalladas por metodo y por parametro.
"""

import logging
from typing import Dict, Any, Tuple, Optional, Union
from collections import defaultdict

logger = logging.getLogger("mztrain")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.gradient. "
        "Instalar con: pip install torch>=2.0.0"
    )

from .config import GradientCompression


class ZGradientCompressor:
    """Comprime gradientes con error feedback para reducir memoria y comunicacion.

    Metodos disponibles:

    - **TOP_K**: Solo mantiene los K gradientes mas grandes (biased, requiere EF).
    - **RANDOM_K**: Muestra aleatoria reescalada por n/k (insesgado; NO
      contractivo, se usa sin error-feedback).
    - **QUANTIZE_1BIT**: Solo signo del gradiente escalado por bloque.
    - **QUANTIZE_INT8**: Cuantiza gradientes a INT8 con scaling por bloque.
    - **SVD**: Descompone gradiente matricial con SVD de bajo rango.

    Error feedback (solo compresores contractivos: TOP_K/1BIT/INT8/SVD): los
    gradientes descartados por compresion se acumulan y se agregan al siguiente
    paso, garantizando convergencia. RANDOM_K queda excluido (su realimentacion
    diverge). Los buffers de error se almacenan comprimidos a INT8 (ConEF),
    reduciendo el overhead de memoria de ~100% a ~25%.

    Args:
        method: Metodo de compresion a usar.
        top_k_ratio: Ratio de gradientes a mantener para TOP_K / RANDOM_K.
        svd_rank: Rango para compresion SVD de gradientes.
        min_size_to_compress: Minimo de elementos para comprimir un tensor.
        compress_error_buffer: Comprimir los buffers de error a INT8 (ConEF).

    Example:
        >>> compressor = ZGradientCompressor(GradientCompression.TOP_K, top_k_ratio=0.1)
        >>> compressed_grad = compressor.compress("layer1.weight", grad_tensor)
    """

    # Tamano de bloque para cuantizacion block-wise
    _BLOCK_SIZE_INT8 = 2048
    _BLOCK_SIZE_1BIT = 1024

    def __init__(
        self,
        method: GradientCompression = GradientCompression.TOP_K,
        top_k_ratio: float = 0.1,
        svd_rank: int = 16,
        min_size_to_compress: int = 1024,
        compress_error_buffer: bool = True,
    ):
        self.method = method
        self.top_k_ratio = top_k_ratio
        self.svd_rank = svd_rank
        self.min_size = min_size_to_compress
        self.compress_error_buffer = compress_error_buffer

        # Error feedback buffers: {name: tensor} o {name: (int8_tensor, scale)}
        self._error_feedback: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = {}

        # Estadisticas
        self._stats: Dict[str, Any] = defaultdict(float)
        self._stats["skipped_small"] = 0

    def compress(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """Comprimir gradiente y aplicar error feedback.

        Args:
            name: Nombre del parametro (para tracking de error feedback).
            grad: Tensor de gradiente a comprimir.

        Returns:
            Gradiente comprimido (mismo shape que entrada).
        """
        if self.method == GradientCompression.NONE:
            return grad

        # Filtro de tamano: no comprimir tensores pequenos
        if grad.numel() < self.min_size:
            self._stats["skipped_small"] += 1
            return grad

        # Compensar con error acumulado del paso anterior. Si la topologia
        # del parametro cambio (crecimiento de rango, sleep/compactacion de
        # ElasticRank, refactorizacion), el buffer de error feedback quedo
        # con el shape viejo: descartarlo (no se puede arrastrar residuo de
        # cuantizacion a traves de un cambio de dimensionalidad). Es un
        # evento raro (boundary de epoch) y resetear el error feedback a 0
        # para ese parametro es la unica semantica correcta.
        # RANDOM_K es un estimador INSESGADO reescalado por n/k: NO es
        # contractivo (E||g-C(g)||^2 = (1/p-1)||g||^2 > ||g||^2 para k/n<1/2), y
        # realimentarlo por error-feedback hace divergir el buffer geometricamente.
        # El error-feedback solo aplica a los compresores contractivos.
        uses_error_feedback = self.method != GradientCompression.RANDOM_K

        if uses_error_feedback and name in self._error_feedback:
            error_prev = self._decompress_error(self._error_feedback[name], grad.device)
            if error_prev.shape != grad.shape:
                del self._error_feedback[name]
                self._stats["error_buffer_resets"] += 1
            else:
                grad = grad + error_prev

        # Comprimir segun metodo
        if self.method == GradientCompression.TOP_K:
            compressed, error = self._top_k(grad)
        elif self.method == GradientCompression.RANDOM_K:
            compressed, error = self._random_k(grad)
        elif self.method == GradientCompression.QUANTIZE_1BIT:
            compressed, error = self._one_bit_blockwise(grad)
        elif self.method == GradientCompression.QUANTIZE_INT8:
            compressed, error = self._int8_blockwise(grad)
        elif self.method == GradientCompression.SVD:
            compressed, error = self._svd(grad)
        else:
            return grad

        # Almacenar error solo para los compresores contractivos (ver arriba:
        # el error-feedback de RANDOM_K diverge).
        if uses_error_feedback:
            self._error_feedback[name] = self._compress_error(error)

        # Stats
        self._stats["total_compressed"] += 1
        self._stats["total_error_norm"] += float(error.norm())
        self._stats["total_grad_norm"] += float(grad.norm())

        return compressed

    # ========================================================================
    # METODOS DE COMPRESION
    # ========================================================================

    def _top_k(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mantener solo top-K gradientes por magnitud.

        Compresor SESGADO — requiere error feedback para convergencia.
        Complejidad: O(n log k) por el topk parcial.
        """
        flat = grad.reshape(-1)
        k = max(1, int(flat.numel() * self.top_k_ratio))
        _, indices = flat.abs().topk(k, sorted=False)

        compressed = torch.zeros_like(flat)
        compressed[indices] = flat[indices]
        error = flat - compressed

        self._stats["top_k_sparsity"] = 1.0 - (k / flat.numel())
        return compressed.reshape(grad.shape), error.reshape(grad.shape)

    def _random_k(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Muestreo aleatorio de K gradientes, reescalado por n/k.

        Compresor INSESGADO: E[C(g)] = g. NO es contractivo (la varianza crece,
        E||g-C(g)||^2 = (1/p-1)||g||^2), asi que se usa CRUDO en SGD, SIN
        error-feedback (el EF de un operador expansivo diverge; ver compress()).
        Converge por insesgadez con una penalizacion de varianza ~1/p.
        Complejidad: O(n) por el randperm.
        """
        flat = grad.reshape(-1)
        n = flat.numel()
        k = max(1, int(n * self.top_k_ratio))

        # Seleccionar indices aleatorios
        indices = torch.randperm(n, device=flat.device)[:k]

        # Escalar por n/k para que sea estimador no sesgado
        scale = n / k
        compressed = torch.zeros_like(flat)
        compressed[indices] = flat[indices] * scale
        error = grad - compressed.reshape(grad.shape)

        self._stats["random_k_ratio"] = k / n
        return compressed.reshape(grad.shape), error

    def _one_bit_blockwise(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """1-bit SGD con scaling por bloque.

        Cada bloque tiene su propia magnitud promedio, mejorando la calidad
        sobre el scaling global (que pierde informacion de varianza local).
        """
        flat = grad.reshape(-1)
        n = flat.numel()
        block_size = self._BLOCK_SIZE_1BIT
        compressed = torch.zeros_like(flat)

        for i in range(0, n, block_size):
            block = flat[i:i + block_size]
            magnitude = block.abs().mean()
            compressed[i:i + block_size] = block.sign() * magnitude

        error = flat - compressed
        return compressed.reshape(grad.shape), error.reshape(grad.shape)

    def _int8_blockwise(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """INT8 block-wise quantization.

        Cada bloque de 2048 elementos se cuantiza independientemente con su
        propio factor de escala. Esto aisla outliers a bloques individuales
        en vez de degradar precision en todo el tensor.
        (Inspirado en bitsandbytes block-wise quantization)
        """
        flat = grad.reshape(-1)
        n = flat.numel()
        block_size = self._BLOCK_SIZE_INT8
        compressed = torch.zeros_like(flat)

        for i in range(0, n, block_size):
            block = flat[i:i + block_size]
            abs_max = block.abs().max()
            if abs_max > 0:
                scale = abs_max / 127.0
                quantized = (block / scale).round().clamp(-127, 127)
                compressed[i:i + block_size] = quantized * scale
            # Si abs_max == 0, el bloque queda en ceros (correcto)

        error = flat - compressed
        self._stats["int8_blocks"] = (n + block_size - 1) // block_size
        return compressed.reshape(grad.shape), error.reshape(grad.shape)

    def _svd(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Comprimir gradiente con SVD de bajo rango (solo matrices 2D).

        Para tensores 1D o >2D, hace fallback a INT8 block-wise.
        Usa SVD truncada randomizada para eficiencia cuando el rango es
        mucho menor que las dimensiones.
        """
        if grad.dim() != 2:
            return self._int8_blockwise(grad)

        m, n = grad.shape
        r = min(self.svd_rank, min(m, n))

        if r >= min(m, n) // 2:
            # SVD exacta truncada: mas eficiente cuando r es grande relativo a dims
            U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
            compressed = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
        else:
            # SVD aleatorizada (Halko et al. 2011): eficiente para r << min(m,n)
            compressed = self._randomized_svd_compress(grad, r)

        error = grad - compressed
        self._stats["svd_rank_used"] = r
        return compressed, error

    @staticmethod
    def _randomized_svd_compress(M: torch.Tensor, rank: int) -> torch.Tensor:
        """SVD aleatorizada para compresion low-rank.

        Halko, Martinsson, Tropp (2011). Complejidad O(m*n*rank)
        en vez de O(m*n*min(m,n)) de SVD exacta.
        """
        m, n = M.shape
        k = rank + 10  # oversamples para estabilidad

        # Random projection
        Omega = torch.randn(n, k, device=M.device, dtype=M.dtype)
        Y = M @ Omega  # m x k

        # Power iteration (1-2 rondas mejoran precision para decaimiento lento de sigma)
        for _ in range(2):
            Y = M @ (M.T @ Y)

        # QR para base ortonormal
        Q, _ = torch.linalg.qr(Y)  # m x k

        # Proyectar y SVD pequena
        B = Q.T @ M  # k x n
        U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_hat  # m x k

        # Reconstruir con rank truncado
        return (U[:, :rank] * S[:rank].unsqueeze(0)) @ Vt[:rank, :]

    # ========================================================================
    # ERROR BUFFER COMPRESSION (ConEF)
    # ========================================================================

    def _compress_error(
        self, error: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Comprimir buffer de error para reducir overhead de memoria.

        ConEF (Contractive Error Feedback): almacena el error en INT8
        en vez de FP32, reduciendo el overhead de 4 bytes/elem a 1 byte/elem.
        Reduce el overhead de memoria del error feedback de ~100% a ~25%.
        """
        if not self.compress_error_buffer:
            return error.detach()

        abs_max = error.abs().max()
        if abs_max == 0:
            return (
                torch.zeros(error.shape, dtype=torch.int8, device=error.device),
                torch.tensor(0.0, device=error.device),
            )

        scale = abs_max / 127.0
        quantized = (error.detach() / scale).round().clamp(-127, 127).to(torch.int8)
        return (quantized, scale)

    def _decompress_error(
        self,
        stored: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """Descomprimir buffer de error."""
        if isinstance(stored, torch.Tensor):
            return stored.to(device)
        quantized, scale = stored
        return quantized.to(device=device, dtype=torch.float32) * scale.to(device)

    # ========================================================================
    # STATS / CONTROL
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadisticas de compresion de gradientes."""
        stats = dict(self._stats)

        # Calcular memoria usada por error buffers
        ef_bytes = 0
        for stored in self._error_feedback.values():
            if isinstance(stored, tuple):
                ef_bytes += stored[0].numel() * 1 + 4  # INT8 + scale float
            elif isinstance(stored, torch.Tensor):
                ef_bytes += stored.numel() * stored.element_size()
        stats["error_buffer_memory_mb"] = ef_bytes / (1024 * 1024)
        stats["num_tracked_params"] = len(self._error_feedback)

        # Ratio de error relativo
        if stats.get("total_grad_norm", 0) > 0:
            stats["avg_relative_error"] = (
                stats["total_error_norm"] / stats["total_grad_norm"]
            )

        return stats

    def invalidate(self, name: str) -> None:
        """Descartar el error-feedback de un parametro cuya topologia u orden
        de direcciones cambio.

        El check por shape en compress() solo detecta cambios de dimension; no
        detecta un reordenamiento a rango constante (ElasticRank durmiendo k
        direcciones y rellenando k), tras el cual el residuo de cuantizacion
        viejo apuntaria a la direccion singular equivocada. El caller (engine)
        invoca esto para cada parametro afectado por una cirugia estructural.
        """
        if name in self._error_feedback:
            del self._error_feedback[name]
            self._stats["error_buffer_resets"] += 1

    def reset(self) -> None:
        """Resetear error feedback y estadisticas."""
        self._error_feedback.clear()
        self._stats.clear()
        self._stats["skipped_small"] = 0
