"""
MZTrain - Data package para procesamiento de codigo.

Contiene tokenizador code-aware, datasets para pre-entrenamiento MLM,
y generador de codigo sintetico multi-lenguaje.
"""

from .tokenizer import CodeTokenizer
from .dataset import SyntheticCodeGenerator, CodeDataset, MLMDataset, CausalCodeDataset

__all__ = [
    "CodeTokenizer",
    "SyntheticCodeGenerator",
    "CodeDataset",
    "MLMDataset",
    "CausalCodeDataset",
]
