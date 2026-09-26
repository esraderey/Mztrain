"""
MZTrain - APOLLO-style Adaptive Optimizer (T7).

Combina lo mejor de tres papers:
- GaLore (ICML 2024): Proyeccion low-rank del primer momento m
- Adam-mini (ICLR 2025): Segundo momento v escalar por bloque
- APOLLO (MLSys 2025 Outstanding Paper): Combinacion con error compensation

Resultado: memoria ~= SGD, convergencia ~= AdamW.

El cuello de botella de memoria en modelos >1B params NO son los pesos
(ya factorizados), son los estados del optimizer. Adam guarda m y v del
mismo tamano que los parametros. ZAdaptiveOptimizer reduce esto ~96.8%:

    Adam estandar (1B):     m=2.15GB + v=2.15GB = 4.3GB
    ZAdaptiveOptimizer:     m_lr=67MB + v_scalar=2MB + P=67MB = ~136MB

Para parametros pequenos (<2D o <4096 elementos), usa Adam estandar
completo para preservar convergencia.

Referencias:
- APOLLO: SGD-level memory, AdamW-level performance (MLSys 2025)
- Adam-mini: Zhang et al., ICLR 2025 (arXiv:2406.16793)
- GaLore: Zhao et al., ICML 2024 Oral (arXiv:2403.03507)
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
        "PyTorch es requerido para mztrain.adaptive_optimizer. "
        "Instalar con: pip install torch>=2.0.0"
    )

from .projector import ZGaLoreProjector


class ZAdaptiveOptimizer(torch.optim.Optimizer):
    """Adam optimizer APOLLO-style: m low-rank + v escalar por bloque.

    Para parametros 2D grandes (>= min_dim_to_project elementos):
    - m (primer momento): proyectado a subespacio de rango r via GaLore
    - v (segundo momento): UN SOLO ESCALAR por bloque de block_size elementos
    - Error compensation: residuo de la proyeccion se acumula

    Para parametros pequenos o 1D (biases, layer norms):
    - Adam estandar completo (m y v per-element)

    La diferencia clave vs ZGaLoreOptimizer es que v es escalar por bloque
    en vez de per-element en espacio proyectado. Esto ahorra >96% en v.

    Args:
        params: Parametros del modelo.
        lr: Learning rate.
        betas: Coeficientes de momentos (beta1, beta2).
        eps: Epsilon para estabilidad numerica.
        weight_decay: Weight decay (decoupled, estilo AdamW).
        rank: Rango del subespacio de proyeccion para m.
        block_size: Tamano de bloque para v escalar (1 valor v cada N elementos).
        projection_update_freq: Cada cuantos steps actualizar subespacio via SVD.
        min_dim_to_project: Dimension minima para aplicar proyeccion.
        compress_states: Comprimir m low-rank con INT8 block-wise.
        compression_interval: Re-comprimir cada N steps.

    Example:
        >>> optimizer = ZAdaptiveOptimizer(model.parameters(), lr=3e-4, rank=128)
        >>> optimizer.step()
        >>> print(optimizer.get_memory_stats())
    """

    _BLOCK_SIZE_COMPRESS = 2048  # Block size para INT8 compression de estados

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        rank: int = 128,
        block_size: int = 1024,
        projection_update_freq: int = 200,
        min_dim_to_project: int = 128,
        compress_states: bool = False,
        compression_interval: int = 10,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rank = rank
        self.block_size = block_size
        self.projection_update_freq = projection_update_freq
        self.min_dim_to_project = min_dim_to_project
        self.compress_states = compress_states
        self.compression_interval = compression_interval
        self._step_count = 0

    def _should_project(self, param: torch.Tensor) -> bool:
        """Determinar si un parametro debe usar proyeccion APOLLO."""
        if param.dim() < 2:
            return False
        if min(param.shape[0], param.shape[1]) < self.min_dim_to_project:
            return False
        if param.numel() < 4096:
            return False
        return True

    def _compute_n_blocks(self, numel: int) -> int:
        """Calcular numero de bloques para v escalar."""
        return max(1, (numel + self.block_size - 1) // self.block_size)

    def _compute_block_variance(self, grad_low: torch.Tensor) -> torch.Tensor:
        """Calcular varianza (g^2) promedio por bloque del gradiente.

        Divide el gradiente aplanado en bloques de block_size y calcula
        la media de g^2 en cada bloque. Esto es la estimacion de varianza
        escalar estilo Adam-mini.

        Args:
            grad_low: Gradiente en espacio low-rank (cualquier shape).

        Returns:
            Tensor de shape (n_blocks,) con la varianza promedio por bloque.
        """
        flat = grad_low.reshape(-1)
        numel = flat.numel()
        n_blocks = self._compute_n_blocks(numel)

        # Pad si no es multiplo exacto
        if numel % self.block_size != 0:
            padded = torch.zeros(n_blocks * self.block_size,
                                 device=flat.device, dtype=flat.dtype)
            padded[:numel] = flat
        else:
            padded = flat

        # Reshape a (n_blocks, block_size) y calcular mean(g^2) por bloque
        blocks = padded.reshape(n_blocks, -1)
        block_var = blocks.pow(2).mean(dim=1)  # (n_blocks,)

        return block_var

    def _expand_block_scalar(
        self, v_scalar: torch.Tensor, target_shape: tuple
    ) -> torch.Tensor:
        """Expandir v escalar por bloque a la shape del gradiente.

        Cada valor escalar se repite block_size veces para cubrir
        el gradiente completo.

        Args:
            v_scalar: Tensor (n_blocks,) con varianzas escalares.
            target_shape: Shape objetivo del gradiente.

        Returns:
            Tensor de target_shape con valores escalares expandidos.
        """
        numel = 1
        for s in target_shape:
            numel *= s

        # Expandir: cada escalar -> block_size elementos
        expanded = v_scalar.unsqueeze(1).expand(-1, self.block_size).reshape(-1)
        expanded = expanded[:numel]

        return expanded.reshape(target_shape)

    # === INT8 Compression (reutilizado de ZGaLoreOptimizer) ===

    def _compress_state(self, tensor: torch.Tensor) -> tuple:
        """Comprimir estado a INT8 block-wise."""
        flat = tensor.reshape(-1)
        n = flat.numel()
        bs = self._BLOCK_SIZE_COMPRESS
        n_blocks = (n + bs - 1) // bs

        if n % bs != 0:
            padded = torch.zeros(n_blocks * bs, device=flat.device, dtype=flat.dtype)
            padded[:n] = flat
        else:
            padded = flat

        blocks = padded.reshape(n_blocks, bs)
        scales = blocks.abs().amax(dim=1)
        safe_scales = scales.clamp(min=1e-12)
        # Use full INT8 range: map [-abs_max, abs_max] -> [-127, 127].
        # Without the *127 factor, values collapse to {-1, 0, 1} and exp_avg
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

    @torch.no_grad()
    def step(self, closure=None):
        """Paso de optimizacion APOLLO-style.

        Parametros grandes (2D, >4096 elementos):
            1. Proyectar gradiente: g_low = P^T @ grad (r x n)
            2. Actualizar m low-rank: m = beta1*m + (1-beta1)*g_low
            3. Actualizar v escalar: v_block = beta2*v + (1-beta2)*mean(g_low^2, per_block)
            4. Calcular update: update = m / (sqrt(v_expanded) + eps)
            5. Reconstruir: delta_W = P @ update

        Parametros pequenos (1D, biases, norms):
            Adam estandar completo.

        Args:
            closure: Closure para reevaluar loss (opcional).

        Returns:
            Loss si closure fue proporcionado, None de lo contrario.
        """
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

                # === LAZY INIT ===
                if len(state) == 0:
                    state['step'] = 0

                    if self._should_project(p):
                        m, n = p.shape[0], p.shape[1]
                        proj = ZGaLoreProjector(
                            m, n, self.rank, self.projection_update_freq,
                        )
                        state['projector'] = proj
                        lr_shape = proj.low_rank_shape

                        # m: low-rank (r x n) o (m x r)
                        state['exp_avg'] = torch.zeros(
                            lr_shape, device=p.device, dtype=p.dtype
                        )

                        # v: escalar por bloque — UN valor cada block_size elementos
                        lr_numel = lr_shape[0] * lr_shape[1]
                        n_blocks = self._compute_n_blocks(lr_numel)
                        state['exp_avg_sq'] = torch.zeros(
                            n_blocks, device=p.device, dtype=p.dtype
                        )

                        state['is_projected'] = True
                    else:
                        # Adam estandar para params pequenos
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        state['is_projected'] = False

                    state['compressed'] = False

                state['step'] += 1
                t = state['step']

                # Descomprimir si necesario
                if state['compressed'] and self.compress_states:
                    m_val = self._decompress_state(state['exp_avg'], p.dtype)
                    # v escalar NO se comprime (ya es tiny)
                    v_val = state['exp_avg_sq']
                    if isinstance(v_val, tuple):
                        v_val = self._decompress_state(v_val, p.dtype)
                else:
                    m_val = state['exp_avg']
                    v_val = state['exp_avg_sq']

                # Weight decay (decoupled, estilo AdamW)
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                if state['is_projected']:
                    # === APOLLO PROJECTED PATH ===
                    proj = state['projector']
                    grad_2d = grad.reshape(p.shape[0], -1)

                    # 1. Proyectar gradiente al subespacio low-rank
                    g_low = proj.project(grad_2d)

                    # 2. Actualizar m (low-rank, per-element en espacio proyectado)
                    m_val.mul_(beta1).add_(g_low, alpha=1 - beta1)

                    # 3. Actualizar v (ESCALAR por bloque — clave de APOLLO)
                    block_var = self._compute_block_variance(g_low)
                    v_val.mul_(beta2).add_(block_var, alpha=1 - beta2)

                    # 4. Bias correction
                    bc1 = 1 - beta1 ** t
                    bc2 = 1 - beta2 ** t
                    step_size = lr / bc1

                    # 5. Expandir v escalar a shape de m para division
                    v_expanded = self._expand_block_scalar(v_val, m_val.shape)
                    denom = (v_expanded.sqrt() / math.sqrt(bc2)).add_(eps)

                    # 6. Update en espacio low-rank
                    update_low = m_val / denom

                    # 7. Reconstruir en espacio completo y aplicar
                    update_full = proj.project_back(update_low * (-step_size))
                    p.data.add_(update_full.reshape(p.shape))

                else:
                    # === STANDARD ADAM PATH (params pequenos) ===
                    m_val.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v_val.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bc1 = 1 - beta1 ** t
                    bc2 = 1 - beta2 ** t
                    step_size = lr / bc1
                    denom = (v_val.sqrt() / math.sqrt(bc2)).add_(eps)

                    p.data.addcdiv_(m_val, denom, value=-step_size)

                # Re-comprimir periodicamente (solo m, v escalar es tiny)
                should_compress = (
                    self.compress_states
                    and self._step_count % self.compression_interval == 0
                    and state['is_projected']
                )
                if should_compress:
                    state['exp_avg'] = self._compress_state(m_val)
                    state['exp_avg_sq'] = v_val  # v escalar ya es minusculo
                    state['compressed'] = True
                else:
                    state['exp_avg'] = m_val
                    state['exp_avg_sq'] = v_val
                    state['compressed'] = False

        return loss

    def get_memory_stats(self) -> dict:
        """Estadisticas de memoria del optimizer.

        Returns:
            Dict con metricas de memoria: total_params, projected/standard counts,
            state_memory_mb, full_memory_mb, memory_saved_pct, rank, block_size.
        """
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
                total_full_bytes += param_bytes * 2  # m + v full size

                if state.get('is_projected', False):
                    projected_count += 1
                    proj = state['projector']
                    lr_numel = proj.low_rank_shape[0] * proj.low_rank_shape[1]
                    n_blocks = self._compute_n_blocks(lr_numel)

                    if state.get('compressed', False):
                        # m comprimido a INT8
                        m_bytes = lr_numel * 1  # INT8
                    else:
                        m_bytes = lr_numel * p.element_size()

                    # v escalar: n_blocks floats (minusculo)
                    v_bytes = n_blocks * p.element_size()

                    # Projector storage (P matrix)
                    proj_bytes = proj.m * proj.rank * p.element_size()

                    total_state_bytes += m_bytes + v_bytes + proj_bytes
                else:
                    standard_count += 1
                    total_state_bytes += param_bytes * 2  # m + v full

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
            "block_size": self.block_size,
            "step_count": self._step_count,
        }
