#!/usr/bin/env python3
"""
MZTrain - Ejemplo basico de uso.

Demuestra como factorizar un modelo MLP simple y entrenarlo
con el motor MZTrain usando todas las optimizaciones de memoria.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mztrain import (
    ZTrainEngine,
    ZTrainConfig,
    RankSchedule,
    factorize_existing_model,
    estimate_memory_savings,
)


def create_synthetic_data(n_samples=1000, n_features=256, n_classes=10):
    """Crear dataset sintetico para demostracion."""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(X, y)

    n_train = int(0.8 * n_samples)
    train_dataset = TensorDataset(X[:n_train], y[:n_train])
    val_dataset = TensorDataset(X[n_train:], y[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader


def main():
    print("=" * 60)
    print("MZTrain - Ejemplo Basico")
    print("=" * 60)

    # 1. Crear modelo original
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo original: {total_params:,} parametros")

    # 2. Estimar ahorro de memoria
    print("\n--- Estimacion de Ahorro ---")
    savings = estimate_memory_savings(model, rank=32)
    print(f"Tradicional:  {savings['traditional']['total_mb']:.1f} MB")
    print(f"MZTrain:      {savings['mztrain']['total_mb']:.1f} MB")
    print(f"Ahorro:       {savings['savings']['total_pct']:.1f}%")
    print(f"Factor:       {savings['savings']['factor']:.1f}x")

    # 3. Configurar entrenamiento
    config = ZTrainConfig(
        initial_rank=16,
        max_rank=64,
        rank_schedule=RankSchedule.EXPONENTIAL,
        rank_growth_interval=5,
        compress_optimizer_states=True,
        use_amp=False,  # False para CPU
        learning_rate=1e-3,
        log_interval=50,
    )

    # 4. Crear engine y datos
    train_loader, val_loader = create_synthetic_data()
    engine = ZTrainEngine(model, config, device=torch.device("cpu"))

    # 5. Funcion de perdida
    def loss_fn(model, batch):
        x, y = batch
        return F.cross_entropy(model(x), y)

    # 6. Entrenar
    print("\n--- Entrenamiento ---")
    summary = engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        epochs=10,
        early_stopping_patience=5,
    )

    # 7. Resultados
    print("\n--- Resultados ---")
    print(f"Epochs completados: {len(summary['train_losses'])}")
    print(f"Loss final (train): {summary['train_losses'][-1]:.4f}")
    if summary['val_losses']:
        print(f"Loss final (val):   {summary['val_losses'][-1]:.4f}")
    print(f"Rango final:        {summary['ranks'][-1]}")
    print(f"Tiempo total:       {summary['total_time_s']:.1f}s")

    # 8. Exportar modelo
    print("\n--- Exportacion ---")
    full_model = engine.export_full_model()
    print(f"Modelo exportado: {sum(p.numel() for p in full_model.parameters()):,} params")

    print("\nEjemplo completado.")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
