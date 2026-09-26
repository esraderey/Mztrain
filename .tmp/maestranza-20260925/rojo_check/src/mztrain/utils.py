"""
MZTrain - Utilidades de conversion y estimacion.

Funciones para factorizar modelos existentes y estimar ahorros
de memoria sin modificar el modelo original.
"""

import copy
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("mztrain")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.utils. "
        "Instalar con: pip install torch>=2.0.0"
    )

from .layers import ZFactorizedLinear


def factorize_existing_model(
    model: nn.Module,
    rank: int = 64,
    min_params: int = 4096,
    exclude_patterns: Optional[List[str]] = None,
    preserve_map: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Factorizar un modelo existente para entrenamiento MZTrain.

    Convierte capas nn.Linear a ZFactorizedLinear via SVD truncada. Por
    defecto aplica EPSI (escala los valores singulares para preservar la
    energia de Frobenius): esto conserva la VARIANZA de las activaciones —
    buena inicializacion para re-entrenar — pero NO reproduce exactamente el
    mapa lineal original cuando el espectro no esta concentrado en el top-r.
    Con ``preserve_map=True`` se usa la SVD truncada pura, que es la mejor
    aproximacion rango-r del mapa (mejor para inferencia inmediata sin
    re-entrenar), a costa de posible colapso de varianza con rangos agresivos.

    Args:
        model: Modelo PyTorch existente.
        rank: Rango de factorizacion.
        min_params: Minimo de parametros para factorizar una capa.
        exclude_patterns: Patrones de nombre de capa a excluir.
        preserve_map: Si True, desactiva EPSI y preserva el mapa lineal
            (SVD truncada fiel) en vez de la varianza.

    Returns:
        Tupla (modelo_factorizado, estadisticas).

    Example:
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> z_model, stats = factorize_existing_model(model, rank=32)
        >>> print(f"Ahorro: {stats['total_savings_pct']:.1f}%")
    """
    model = copy.deepcopy(model)
    stats: Dict[str, Any] = {
        "original_params": sum(p.numel() for p in model.parameters()),
        "replaced_layers": [],
        "skipped_layers": [],
    }

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if exclude_patterns:
            if any(pat in name for pat in exclude_patterns):
                stats["skipped_layers"].append({
                    "name": name or "<root>",
                    "reason": "excluded_pattern",
                    "params": module.weight.numel(),
                })
                continue

        num_params = module.weight.numel()
        if num_params < min_params:
            stats["skipped_layers"].append({
                "name": name or "<root>",
                "reason": f"too_small ({num_params} < {min_params})",
                "params": num_params,
            })
            continue

        z_linear = ZFactorizedLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            bias=module.bias is not None,
            existing_weight=module.weight.data,
            epsi_scaling=not preserve_map,
        )
        if module.bias is not None:
            z_linear.bias.data.copy_(module.bias.data)

        # Reemplazar en el modelo. named_modules() usa "" para el modulo raiz.
        if name == "":
            model = z_linear
        else:
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            if parts[-1].isdigit():
                parent[int(parts[-1])] = z_linear
            else:
                setattr(parent, parts[-1], z_linear)

        stats["replaced_layers"].append({
            "name": name or "<root>",
            "original_params": num_params,
            "factorized_params": z_linear.num_parameters,
            "compression": f"{z_linear.compression_ratio:.2%}",
            "rank": rank,
            "recon_error": f"{z_linear._reconstruction_error:.6f}",
        })

    stats["factorized_params"] = sum(p.numel() for p in model.parameters())
    stats["total_savings_pct"] = (
        (1 - stats["factorized_params"] / max(stats["original_params"], 1)) * 100
    )

    return model, stats


def estimate_memory_savings(
    model: nn.Module,
    rank: int = 64,
    min_params: int = 4096,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Estimar ahorro de memoria de MZTrain vs entrenamiento tradicional.

    Retorna comparacion detallada sin modificar el modelo.

    Args:
        model: Modelo PyTorch a analizar.
        rank: Rango de factorizacion propuesto.
        min_params: Minimo de parametros para factorizar.
        batch_size: Tamano de batch (para estimacion de activaciones).

    Returns:
        Diccionario con comparacion detallada de memoria.

    Example:
        >>> model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 512))
        >>> savings = estimate_memory_savings(model, rank=32)
        >>> print(f"Factor de ahorro: {savings['savings']['factor']:.1f}x")
    """
    total_params = sum(p.numel() for p in model.parameters())
    factorizable_params = 0
    factorized_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n = module.weight.numel()
            if n >= min_params:
                factorizable_params += n
                m, k = module.weight.shape
                r = min(rank, min(m, k))
                factorized_params += m * r + r + r * k
                if module.bias is not None:
                    factorized_params += m
                    # el bias tambien es "factorizable" a efectos de conteo:
                    # sin esto queda en non_factorizable y se suma dos veces.
                    factorizable_params += m

    # Memoria tradicional (FP32)
    param_bytes = total_params * 4
    grad_bytes = total_params * 4
    adam_bytes = total_params * 4 * 2  # m + v
    traditional_total = param_bytes + grad_bytes + adam_bytes

    # Memoria MZTrain
    non_factorizable = total_params - factorizable_params
    z_param_bytes = (factorized_params + non_factorizable) * 4
    z_grad_bytes = (factorized_params + non_factorizable) * 4
    z_adam_bytes = (factorized_params + non_factorizable) * 4 * 2 * 0.25  # INT8
    z_total = z_param_bytes + z_grad_bytes + z_adam_bytes

    return {
        "traditional": {
            "params_mb": param_bytes / (1024**2),
            "gradients_mb": grad_bytes / (1024**2),
            "optimizer_mb": adam_bytes / (1024**2),
            "total_mb": traditional_total / (1024**2),
        },
        "mztrain": {
            "params_mb": z_param_bytes / (1024**2),
            "gradients_mb": z_grad_bytes / (1024**2),
            "optimizer_mb": z_adam_bytes / (1024**2),
            "total_mb": z_total / (1024**2),
        },
        "savings": {
            "total_pct": (1 - z_total / max(traditional_total, 1)) * 100,
            "params_pct": (
                (1 - factorized_params / max(factorizable_params, 1)) * 100
                if factorizable_params > 0 else 0
            ),
            "factor": traditional_total / max(z_total, 1),
        },
        "details": {
            "total_params": total_params,
            "factorizable_params": factorizable_params,
            "factorized_params": factorized_params,
            "rank": rank,
        },
    }
