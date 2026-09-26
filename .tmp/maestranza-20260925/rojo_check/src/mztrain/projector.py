"""
MZTrain - Proyeccion de gradientes a subespacios de bajo rango (GaLore).

En vez de que el optimizer (Adam) opere sobre los gradientes completos
de dimension (m x n), proyecta los gradientes a un subespacio de rango r
y opera Adam ahi. Los estados del optimizer se reducen de O(m*n) a O(r*n).

El subespacio se actualiza periodicamente via SVD aleatorizada (Halko et al.)
para permitir que la optimizacion explore todo el espacio de parametros
cumulativamente, a pesar de operar en bajo rango en cada paso.

Referencias:
- GaLore: Zhao et al., ICML 2024 Oral (arXiv:2403.03507)
- GaLore 2: Su & Gu et al., Abril 2025 (arXiv:2504.20437)
- Halko, Martinsson, Tropp 2011 - Randomized SVD
"""

import math
import logging
from typing import Optional

logger = logging.getLogger("mztrain")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.projector. "
        "Instalar con: pip install torch>=2.0.0"
    )


def randomized_svd(
    M: torch.Tensor,
    rank: int,
    n_oversamples: int = 10,
    n_power_iterations: int = 2,
) -> tuple:
    """SVD aleatorizada (Halko et al. 2011).

    Complejidad O(m * n * (rank + oversamples)) vs O(m * n * min(m,n)).
    GaLore 2 reporta 15x speedup sin degradacion medible.

    Args:
        M: Matriz de entrada (m x n).
        rank: Rango objetivo.
        n_oversamples: Oversamples para estabilidad.
        n_power_iterations: Iteraciones de potencia para mejorar precision
            cuando los valores singulares decaen lentamente.

    Returns:
        (U, S, Vt) truncados a rank.
    """
    m, n = M.shape
    k = min(rank + n_oversamples, min(m, n))

    # Paso 1: Random projection para capturar espacio columna
    Omega = torch.randn(n, k, device=M.device, dtype=M.dtype)
    Y = M @ Omega  # m x k

    # Paso 2: Power iterations (crucial para decaimiento lento de sigma)
    for _ in range(n_power_iterations):
        Z = M.T @ Y    # n x k
        Y = M @ Z       # m x k

    # Paso 3: QR para base ortonormal
    Q, _ = torch.linalg.qr(Y)  # m x k

    # Paso 4: Proyectar y SVD pequena
    B = Q.T @ M  # k x n
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Paso 5: Reconstruir U en espacio original
    U = Q @ U_hat  # m x k

    r = min(rank, S.shape[0])
    return U[:, :r], S[:r], Vt[:r, :]


class ZGaLoreProjector:
    """Proyector de gradientes a subespacio de bajo rango.

    Mantiene matrices de proyeccion P (m x r) que se actualizan cada T steps
    via SVD aleatorizada del gradiente. El gradiente se proyecta como:

        G_low = P^T @ G    (r x n, en vez de m x n)

    Y el update se reconstruye como:

        delta_W = P @ update_low   (m x n)

    El optimizer opera SOLO en el espacio (r x n), reduciendo dramaticamente
    la memoria de sus estados.

    Args:
        m: Dimension de filas del gradiente.
        n: Dimension de columnas del gradiente.
        rank: Rango del subespacio de proyeccion.
        update_freq: Cada cuantos steps actualizar el subespacio via SVD.

    Example:
        >>> proj = ZGaLoreProjector(4096, 4096, rank=128, update_freq=200)
        >>> g_low = proj.project(gradient)          # (128, 4096)
        >>> update = proj.project_back(adam_update)  # (4096, 4096)
    """

    def __init__(
        self,
        m: int,
        n: int,
        rank: int,
        update_freq: int = 200,
    ):
        self.m = m
        self.n = n
        self.rank = min(rank, min(m, n) // 2)  # No exceder la mitad
        self.update_freq = update_freq
        self.step_count = 0

        # Proyector: se inicializa en el primer project() call
        self._projector: Optional[torch.Tensor] = None  # (m, rank)
        self._project_dim: str = 'left'  # 'left' si m >= n, 'right' si m < n

        # Elegir dimension de proyeccion: proyectar la dimension mas grande
        if m >= n:
            self._project_dim = 'left'
            # P: (m, r), G_low = P^T @ G -> (r, n)
        else:
            self._project_dim = 'right'
            # Q: (n, r), G_low = G @ Q -> (m, r)

    @property
    def low_rank_shape(self) -> tuple:
        """Shape del gradiente en espacio low-rank."""
        if self._project_dim == 'left':
            return (self.rank, self.n)
        else:
            return (self.m, self.rank)

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """Proyectar gradiente al subespacio de bajo rango.

        Cada update_freq steps, recalcula el subespacio via SVD aleatorizada.

        Args:
            grad: Gradiente (m x n).

        Returns:
            Gradiente proyectado (r x n) o (m x r) segun dimension.
        """
        self.step_count += 1

        # Actualizar subespacio periodicamente
        if self._projector is None or self.step_count % self.update_freq == 0:
            self._update_subspace(grad)

        # Proyectar
        if self._project_dim == 'left':
            return self._projector.T @ grad  # (r, n)
        else:
            return grad @ self._projector    # (m, r)

    def project_back(self, low_rank_update: torch.Tensor) -> torch.Tensor:
        """Reconstruir update en espacio completo.

        Args:
            low_rank_update: Update en espacio low-rank.

        Returns:
            Update en espacio completo (m x n).
        """
        if self._projector is None:
            raise RuntimeError("Projector not initialized. Call project() first.")

        if self._project_dim == 'left':
            return self._projector @ low_rank_update  # (m, n)
        else:
            return low_rank_update @ self._projector.T  # (m, n)

    def _update_subspace(self, grad: torch.Tensor) -> None:
        """Actualizar subespacio de proyeccion via SVD aleatorizada."""
        if self._project_dim == 'left':
            # Necesitamos las top-r columnas singulares izquierdas de G
            U, _S, _Vt = randomized_svd(grad, self.rank)
            self._projector = U  # (m, r) — ortonormal
        else:
            # Necesitamos las top-r columnas singulares derechas de G
            _U, _S, Vt = randomized_svd(grad, self.rank)
            self._projector = Vt.T  # (n, r) — ortonormal


class ZGaLoreOptimizer(torch.optim.Optimizer):
    """Adam optimizer con proyeccion GaLore de gradientes.

    Para parametros 2D grandes (>= min_dim_to_project), proyecta los gradientes
    a un subespacio de bajo rango y opera Adam ahi. Los estados (m, v) se
    mantienen en el espacio reducido (r x n) en vez de (m x n).

    Para parametros pequenos o 1D (biases, norms), usa Adam estandar.

    Args:
        params: Parametros del modelo.
        lr: Learning rate.
        betas: Coeficientes de momentos.
        eps: Epsilon para estabilidad.
        weight_decay: Weight decay (decoupled, estilo AdamW).
        rank: Rango del subespacio de proyeccion.
        projection_update_freq: Cada cuantos steps actualizar subespacio.
        min_dim_to_project: Dimension minima para aplicar proyeccion.
        compress_states: Comprimir estados (m, v) con INT8 block-wise.
        compression_interval: Re-comprimir cada N steps.

    Example:
        >>> optimizer = ZGaLoreOptimizer(model.parameters(), lr=3e-4, rank=128)
        >>> optimizer.step()
        >>> print(optimizer.get_memory_stats())
    """

    _BLOCK_SIZE = 2048  # Block size para INT8 compression

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        rank: int = 128,
        projection_update_freq: int = 200,
        min_dim_to_project: int = 128,
        compress_states: bool = True,
        compression_interval: int = 10,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rank = rank
        self.projection_update_freq = projection_update_freq
        self.min_dim_to_project = min_dim_to_project
        self.compress_states = compress_states
        self.compression_interval = compression_interval
        self._step_count = 0

    def _should_project(self, param: torch.Tensor) -> bool:
        """Determinar si un parametro debe usar proyeccion GaLore."""
        if param.dim() < 2:
            return False
        if min(param.shape[0], param.shape[1]) < self.min_dim_to_project:
            return False
        return True

    def _compress_state(self, tensor: torch.Tensor) -> tuple:
        """Comprimir estado a INT8 block-wise."""
        flat = tensor.reshape(-1)
        n = flat.numel()
        bs = self._BLOCK_SIZE
        n_blocks = (n + bs - 1) // bs

        # Pad a multiplo de block_size
        if n % bs != 0:
            padded = torch.zeros(n_blocks * bs, device=flat.device, dtype=flat.dtype)
            padded[:n] = flat
        else:
            padded = flat

        blocks = padded.reshape(n_blocks, bs)
        scales = blocks.abs().amax(dim=1)  # (n_blocks,)
        safe_scales = scales.clamp(min=1e-12)
        # Use full INT8 range: map [-abs_max, abs_max] -> [-127, 127].
        # Without the *127 factor, values collapse to {-1, 0, 1} and Adam state
        # is destroyed on every compression cycle.
        quantized = (
            (blocks / safe_scales.unsqueeze(1) * 127.0)
            .round().clamp(-127, 127).to(torch.int8)
        )

        return quantized, scales, tensor.shape, n

    def _decompress_state(self, compressed: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Descomprimir estado desde INT8 block-wise."""
        quantized, scales, shape, orig_numel = compressed
        # Inverse of compress: divide by 127 to undo the INT8 range scaling.
        blocks = quantized.to(dtype) * (scales.to(dtype) / 127.0).unsqueeze(1)
        flat = blocks.reshape(-1)[:orig_numel]
        return flat.reshape(shape)

    _V_LOG_FLOOR = 1e-30

    def _compress_v(self, v: torch.Tensor) -> tuple:
        """Cuantizacion LOGARITMICA del 2o momento de Adam (v >= 0).

        La cuantizacion INT8 LINEAL (_compress_state) colapsa a 0 los valores
        pequenos de un bloque que contiene un outlier; entonces el denominador de
        Adam cae a eps y el paso salta a ~1/eps (1e8), divergiendo. Cuantizar
        log(v) da precision RELATIVA uniforme y preserva los v pequenos (mismo
        esquema que ZCompressedAdam._compress_v).
        """
        safe_v = v.clamp(min=self._V_LOG_FLOOR)
        return self._compress_state(safe_v.log())

    def _decompress_v(self, compressed: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Inversa de _compress_v: INT8 -> log_v -> exp -> v."""
        log_v = self._decompress_state(compressed, dtype)
        return log_v.exp()

    @torch.no_grad()
    def step(self, closure=None):
        """Paso de optimizacion con proyeccion GaLore."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # === INIT ===
                if len(state) == 0:
                    state['step'] = 0
                    if self._should_project(p):
                        m, n = p.shape[0], p.shape[1]
                        proj = ZGaLoreProjector(
                            m, n, self.rank, self.projection_update_freq,
                        )
                        state['projector'] = proj
                        lr_shape = proj.low_rank_shape
                        state['exp_avg'] = torch.zeros(lr_shape, device=p.device, dtype=p.dtype)
                        state['exp_avg_sq'] = torch.zeros(lr_shape, device=p.device, dtype=p.dtype)
                        state['is_projected'] = True
                    else:
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        state['is_projected'] = False
                    state['compressed'] = False

                state['step'] += 1
                t = state['step']

                # Descomprimir si necesario
                if state['compressed'] and self.compress_states:
                    m_val = self._decompress_state(state['exp_avg'], p.dtype)
                    v_val = self._decompress_v(state['exp_avg_sq'], p.dtype)
                else:
                    m_val = state['exp_avg']
                    v_val = state['exp_avg_sq']

                # Weight decay (decoupled)
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                if state['is_projected']:
                    # === PROJECTED PATH ===
                    proj = state['projector']
                    grad_2d = grad.reshape(p.shape[0], -1)

                    # Proyectar gradiente
                    g_low = proj.project(grad_2d)

                    # Adam en espacio low-rank
                    m_val.mul_(beta1).add_(g_low, alpha=1 - beta1)
                    v_val.mul_(beta2).addcmul_(g_low, g_low, value=1 - beta2)

                    # Bias correction
                    bc1 = 1 - beta1 ** t
                    bc2 = 1 - beta2 ** t
                    step_size = lr / bc1
                    denom = (v_val.sqrt() / math.sqrt(bc2)).add_(eps)

                    # Update en espacio low-rank
                    update_low = m_val / denom

                    # Reconstruir y aplicar
                    update_full = proj.project_back(update_low * (-step_size))
                    p.data.add_(update_full.reshape(p.shape))
                else:
                    # === STANDARD ADAM PATH ===
                    m_val.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v_val.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bc1 = 1 - beta1 ** t
                    bc2 = 1 - beta2 ** t
                    step_size = lr / bc1
                    denom = (v_val.sqrt() / math.sqrt(bc2)).add_(eps)

                    p.data.addcdiv_(m_val, denom, value=-step_size)

                # Re-comprimir periodicamente
                should_compress = (
                    self.compress_states
                    and self._step_count % self.compression_interval == 0
                )
                if should_compress:
                    state['exp_avg'] = self._compress_state(m_val)
                    state['exp_avg_sq'] = self._compress_v(v_val)   # log-quant, no lineal
                    state['compressed'] = True
                else:
                    state['exp_avg'] = m_val
                    state['exp_avg_sq'] = v_val
                    state['compressed'] = False

        return loss

    def get_memory_stats(self) -> dict:
        """Estadisticas de memoria del optimizer."""
        total_state_bytes = 0
        total_full_bytes = 0
        projected_count = 0
        standard_count = 0

        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if len(state) == 0:
                    continue

                param_bytes = p.numel() * p.element_size()
                total_full_bytes += param_bytes * 2  # m + v full en FP32/FP16

                if state.get('is_projected', False):
                    projected_count += 1
                    proj = state['projector']
                    lr_numel = proj.low_rank_shape[0] * proj.low_rank_shape[1]
                    if state.get('compressed', False):
                        total_state_bytes += lr_numel * 1 * 2  # INT8 m + v
                    else:
                        total_state_bytes += lr_numel * p.element_size() * 2
                    # Projector storage
                    total_state_bytes += proj.m * proj.rank * p.element_size()
                else:
                    standard_count += 1
                    if state.get('compressed', False):
                        total_state_bytes += p.numel() * 1 * 2
                    else:
                        total_state_bytes += param_bytes * 2

        return {
            "total_params": projected_count + standard_count,
            "projected_params": projected_count,
            "standard_params": standard_count,
            "state_memory_mb": total_state_bytes / (1024 * 1024),
            "full_memory_mb": total_full_bytes / (1024 * 1024),
            "memory_saved_pct": (
                (1 - total_state_bytes / max(total_full_bytes, 1)) * 100
            ),
            "rank": self.rank,
            "step_count": self._step_count,
        }
