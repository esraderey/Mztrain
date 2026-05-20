"""
Task setup for the MZTrain benchmark.

Provides:
  - MNIST data loaders (subsampled for speed)
  - Factorizable MLP model
  - Reproducibility helpers
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Deterministic subsetting indices — fixed regardless of seed so all runs see same data.
_TRAIN_INDICES = list(range(10_000))
_VAL_INDICES = list(range(2_000))


def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility (best-effort, not bit-exact on GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model() -> nn.Module:
    """Standard MLP for MNIST.

    Layer sizes chosen so all hidden Linear layers exceed min_params_to_factorize=4096:
      - Linear(784, 512) -> 401k params  [factorizable]
      - Linear(512, 256) -> 131k params  [factorizable]
      - Linear(256, 128) ->  32k params  [factorizable]
      - Linear(128,  10) ->  1.2k params [skipped: too small]
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def get_mnist_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """MNIST loaders subsampled to 10k train / 2k val for benchmark speed.

    The subset is fixed (not seed-dependent); only DataLoader shuffling uses seed.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    full_val = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    train_subset = Subset(full_train, _TRAIN_INDICES)
    val_subset = Subset(full_val, _VAL_INDICES)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,  # keep deterministic, no worker fork
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy on a loader (for end-of-training eval)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)
