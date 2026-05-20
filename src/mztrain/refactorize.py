"""
MZTrain - Refactorizacion periodica anti-rank-collapse.

El rango efectivo de las matrices de peso decae durante el entrenamiento.
Weight decay lo acelera (empuja hacia rank-2 en redes ReLU). Esto significa
que los componentes de rango anadidos por crecimiento progresivo pueden
colapsar si no se re-factorizan periodicamente.

SLTrain (NeurIPS 2024) con re-starts cada ~200 iteraciones logra 14.37
perplexity en modelos 1B, vs 18.22 sin refactorizacion (-3.85 puntos).

Este modulo implementa la refactorizacion periodica (estilo ReLoRA):
1. Reconstruir peso completo W = U @ diag(S) @ V  (+ sparse si aplica)
2. SVD fresca sobre W
3. Actualizar factores U, S, V  (+ re-calcular sparse si aplica)
4. Resetear estados del optimizer para capas afectadas (lazy init + warmup)

La rotacion de momentum (LDAdam-style) fue probada y descartada:
el mapping U_new^T @ U_old NO es ortogonal cuando los subespacios cambian,
corrompiendo exp_avg y exp_avg_sq de Adam (test_08: NaN, test_09: 89% gap).

Referencias:
- SLTrain benchmark: Mayo 2025 (arXiv:2505.22922v1)
- ReLoRA: Lialin et al. 2023 — merge + re-init (arXiv:2307.05695)
- Implicit Low-Rank Bias: 2024 (OpenReview:3zw9NhLhBM)
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger("mztrain")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.refactorize. "
        "Instalar con: pip install torch>=2.0.0"
    )


def refactorize_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
    factorized_linear_cls,
) -> int:
    """Refactorizar todas las capas factorizadas con SVD fresca.

    Reconstruye W (incluyendo componente sparse si existe), aplica SVD fresca,
    y resetea los estados del optimizer para las capas afectadas.

    El optimizer se adapta naturalmente en los siguientes steps. Se recomienda
    combinar con warmup post-refactorizacion (igual que rank growth).

    Args:
        model: Modelo con capas factorizadas.
        optimizer: Optimizer cuyo estado sera reseteado para capas afectadas.
        rank: Rango actual de factorizacion.
        factorized_linear_cls: Clase(s) de la capa factorizada.
            Puede ser una clase o tupla de clases.

    Returns:
        Numero de capas refactorizadas.
    """
    if not isinstance(factorized_linear_cls, tuple):
        factorized_linear_cls = (factorized_linear_cls,)

    refactorized = 0

    for _name, module in model.named_modules():
        if not isinstance(module, factorized_linear_cls):
            continue

        r = min(rank, module.rank)

        # 1. Reconstruir peso completo actual (incluye sparse si existe)
        W = module.reconstruct_weight().detach()

        # Guard: skip si hay NaN/Inf (modelo divergio)
        if not torch.isfinite(W).all():
            logger.warning(f"[MZTrain] Skipping refactorize for '{_name}': non-finite weights")
            continue

        # 2. Guardar refs a parametros viejos para limpiar optimizer
        old_params = [module.U, module.S, module.V]
        has_sparse = hasattr(module, 'sparse_values') and module.sparse_values is not None

        # 3. SVD fresca
        U_new, S_new, Vt_new = torch.linalg.svd(W.float(), full_matrices=False)
        U_new = U_new[:, :r].to(W.dtype)
        S_new = S_new[:r].to(W.dtype)
        V_new = Vt_new[:r, :].to(W.dtype)

        # 4. Actualizar parametros del modulo
        with torch.no_grad():
            if module.U.shape == U_new.shape:
                module.U.data.copy_(U_new)
            else:
                module.U = nn.Parameter(U_new)

            if module.S.shape == S_new.shape:
                module.S.data.copy_(S_new)
            else:
                module.S = nn.Parameter(S_new)

            if module.V.shape == V_new.shape:
                module.V.data.copy_(V_new)
            else:
                module.V = nn.Parameter(V_new)

        # 5. Re-calcular sparse component si existe
        if has_sparse:
            _update_sparse_component(module, W, U_new, S_new, V_new)

        # 6. Resetear estados del optimizer para los parametros afectados
        #    (estilo ReLoRA: merge -> re-init -> warmup)
        _reset_optimizer_states(optimizer, old_params)
        # Tambien resetear para los nuevos parametros si cambiaron de identidad
        new_params = [module.U, module.S, module.V]
        if has_sparse:
            new_params.append(module.sparse_values)
        for p in new_params:
            if p in optimizer.state:
                del optimizer.state[p]

        refactorized += 1

    if refactorized > 0:
        logger.info(
            f"[MZTrain] Refactorizacion completada: {refactorized} capas, rank={rank} "
            f"(optimizer states reseteados — aplicar warmup)"
        )

    return refactorized


def _update_sparse_component(
    module: nn.Module,
    W_full: torch.Tensor,
    U_new: torch.Tensor,
    S_new: torch.Tensor,
    V_new: torch.Tensor,
) -> None:
    """Re-calcular valores sparse desde el residual post-SVD.

    Args:
        module: Capa con componente sparse (ZSparseFactorizedLinear).
        W_full: Peso completo reconstruido.
        U_new: Nuevos vectores singulares izquierdos.
        S_new: Nuevos valores singulares.
        V_new: Nuevos vectores singulares derechos.
    """
    W_lr = (U_new * S_new.unsqueeze(0)) @ V_new
    residual = W_full - W_lr

    with torch.no_grad():
        module.sparse_values.data.copy_(
            residual[module.sparse_row_idx, module.sparse_col_idx].to(module.sparse_values.dtype)
        )


def _reset_optimizer_states(
    optimizer: torch.optim.Optimizer,
    params: list,
) -> None:
    """Eliminar estados del optimizer para los parametros dados.

    El optimizer hara lazy init en el siguiente step, empezando con
    exp_avg=0 y exp_avg_sq=0 (equivalente a un Adam fresco).

    Combinar con warmup post-refactorizacion para suavizar la transicion.

    Args:
        optimizer: El optimizer.
        params: Lista de parametros cuyos estados se eliminan.
    """
    for param in params:
        if param in optimizer.state:
            del optimizer.state[param]
