"""
Fixtures compartidos para tests de MZTrain.
"""

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def simple_model():
    """Modelo MLP simple para tests."""
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


@pytest.fixture
def small_model():
    """Modelo pequeno (debajo del umbral de factorizacion)."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


@pytest.fixture
def sample_input():
    """Tensor de entrada de ejemplo."""
    return torch.randn(8, 128)


@pytest.fixture
def sample_batch():
    """Batch de ejemplo (input, target)."""
    x = torch.randn(8, 128)
    y = torch.randint(0, 10, (8,))
    return x, y


@pytest.fixture
def device():
    """Dispositivo de computo."""
    return torch.device("cpu")
