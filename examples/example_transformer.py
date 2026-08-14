#!/usr/bin/env python3
"""
MZTrain - Ejemplo de Transformer factorizado.

Demuestra como construir y entrenar un Transformer completo
usando bloques factorizados ZFactorizedTransformerBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mztrain import (
    ZFactorizedTransformerBlock,
    ZFactorizedLinear,
    ZTrainConfig,
    RankSchedule,
)


class ZFactorizedTransformer(nn.Module):
    """Transformer completo con todas las capas factorizadas."""

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        num_classes: int = 10,
        rank: int = 16,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList([
            ZFactorizedTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rank=rank,
                mlp_ratio=4.0,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = ZFactorizedLinear(embed_dim, num_classes, rank=rank)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

    def grow_rank(self, new_rank):
        for block in self.blocks:
            block.grow_rank(new_rank)
        self.classifier.grow_rank(new_rank)


def main():
    print("=" * 60)
    print("MZTrain - Ejemplo Transformer Factorizado")
    print("=" * 60)

    # Crear modelo
    model = ZFactorizedTransformer(
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=10,
        rank=16,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTransformer factorizado: {total_params:,} parametros")

    # Datos sinteticos
    n_samples = 200
    seq_len = 32
    X = torch.randint(0, 1000, (n_samples, seq_len))
    y = torch.randint(0, 10, (n_samples,))

    train_loader = DataLoader(
        TensorDataset(X[:160], y[:160]), batch_size=16, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X[160:], y[160:]), batch_size=16
    )

    # Configurar training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n--- Entrenamiento manual (3 epochs) ---")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}")

    # Crecer rango
    print("\n--- Crecimiento de rango: 16 -> 32 ---")
    model.grow_rank(32)
    new_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros despues del crecimiento: {new_params:,}")

    # Continuar entrenamiento
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 3}: loss={avg_loss:.4f}")

    print("\nEjemplo completado.")


if __name__ == "__main__":
    main()
