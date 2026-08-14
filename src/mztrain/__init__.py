"""
MZTrain - Motor de Entrenamiento en Espacio Comprimido v1.0

Tecnologia que permite entrenar modelos de IA sin miles de GPUs.
En vez de entrenar tensores completos (W: m x n), entrena sus factores
descompuestos (U: m x r, S: r, V: r x n) donde r << min(m,n).

Principio: Si MNEME puede comprimir un modelo entrenado 93%+, entonces
podemos ENTRENAR directamente en ese espacio comprimido.

Componentes:
    - ZFactorizedLinear: Entrena factores SVD directamente
    - ZFactorizedAttention: Multi-head attention factorizado
    - ZFactorizedTransformerBlock: Bloque transformer completo factorizado
    - ZCompressedAdam: Optimizer con estados comprimidos INT8
    - ZGradientCompressor: Compresion de gradientes con error feedback
    - ZRankScheduler: Crecimiento progresivo de rango
    - ZActivationCheckpoint: Checkpointing comprimido de activaciones
    - ZTrainEngine: Orquestador principal del entrenamiento

Ahorro de memoria vs entrenamiento tradicional:
    - Pesos:       ~20-30% del original
    - Gradientes:  ~20-30% del original
    - Optimizer:   ~20-30% del original
    - Activaciones: ~40-60% del original
    - TOTAL:       Un modelo de 1B params que necesita 32GB VRAM
                   ahora necesita ~6-10GB VRAM

Autores: MSC Star Team (Esraderey y Raul Cruz Acosta)
Basado en: MNEME Motor de Memoria Neural Morfica v2.0
"""

from .config import (
    ZTrainConfig,
    RankSchedule,
    GradientCompression,
)

from .layers import (
    ZFactorizedLinear,
    ZFactorizedAttention,
    ZFactorizedTransformerBlock,
    ZSparseFactorizedLinear,
)

from .engine import ZTrainEngine

from .optimizer import ZCompressedAdam

from .gradient import ZGradientCompressor

from .projector import ZGaLoreProjector, ZGaLoreOptimizer

from .adaptive_optimizer import ZAdaptiveOptimizer

from .precision import ZMultiPrecisionManager

from .scheduler import ZRankScheduler, ZSpectralRankScheduler

from .checkpoint import (
    ZActivationCheckpoint,
    z_checkpoint,
)

from .refactorize import refactorize_model

from .elastic_rank import (
    ElasticRankController,
    SleepingDirection,
)

from .vram_governor import ZVRAMGovernor

from .utils import (
    factorize_existing_model,
    estimate_memory_savings,
)

# Sub-packages
from . import data
from . import models
from .models.zcodebert import (
    ZCodeBERTConfig,
    ZCodeBERT,
    ZCodeBERTForMLM,
    ZCodeBERTForCausalLM,
    ZCodeBERTForSequenceClassification,
)
from .data.tokenizer import CodeTokenizer
from .data.dataset import SyntheticCodeGenerator, CodeDataset, MLMDataset, CausalCodeDataset
from .shape_ops import (
    dense_to_factorized,
    pad_adam_entry,
    pad_state_tensor,
    scale_output_rows,
    widen_embedding,
    widen_factorized_linear,
    widen_layernorm,
    zero_block_outputs,
)
from .elastic_shape import (
    GrowthEvent,
    LrWarmup,
    ShapeSchedule,
    apply_event,
)

__version__ = "1.3.1"
__author__ = "MSC Star Team (Esraderey y Raul Cruz Acosta)"
__email__ = "msc.framework@gmail.com"

__all__ = [
    # ElasticShape v1 (crecimiento de forma; claim T8 en docs/evidencia/)
    "dense_to_factorized",
    "pad_adam_entry",
    "pad_state_tensor",
    "scale_output_rows",
    "widen_embedding",
    "widen_factorized_linear",
    "widen_layernorm",
    "zero_block_outputs",
    "GrowthEvent",
    "LrWarmup",
    "ShapeSchedule",
    "apply_event",
    # Config
    "ZTrainConfig",
    "RankSchedule",
    "GradientCompression",
    # Layers
    "ZFactorizedLinear",
    "ZFactorizedAttention",
    "ZFactorizedTransformerBlock",
    "ZSparseFactorizedLinear",
    # Engine
    "ZTrainEngine",
    # Optimizer
    "ZCompressedAdam",
    # Gradient
    "ZGradientCompressor",
    # Projector (GaLore)
    "ZGaLoreProjector",
    "ZGaLoreOptimizer",
    # Adaptive Optimizer (APOLLO, T7)
    "ZAdaptiveOptimizer",
    # Precision Manager (T8)
    "ZMultiPrecisionManager",
    # Scheduler
    "ZRankScheduler",
    "ZSpectralRankScheduler",
    # Checkpointing
    "ZActivationCheckpoint",
    "z_checkpoint",
    # Refactorize
    "refactorize_model",
    # ElasticRank (rango bidireccional)
    "ElasticRankController",
    "SleepingDirection",
    # VRAM Governor (MVP: gating de growth + OOM-retry)
    "ZVRAMGovernor",
    # Utilities
    "factorize_existing_model",
    "estimate_memory_savings",
    # Models
    "ZCodeBERTConfig",
    "ZCodeBERT",
    "ZCodeBERTForMLM",
    "ZCodeBERTForCausalLM",
    "ZCodeBERTForSequenceClassification",
    # Data
    "CodeTokenizer",
    "SyntheticCodeGenerator",
    "CodeDataset",
    "MLMDataset",
    "CausalCodeDataset",
]
