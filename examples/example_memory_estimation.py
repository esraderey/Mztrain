#!/usr/bin/env python3
"""
MZTrain - Ejemplo de estimacion de ahorro de memoria.

Demuestra como usar estimate_memory_savings para analizar modelos
antes de decidir si usar MZTrain.
"""

import torch
import torch.nn as nn

from mztrain import estimate_memory_savings, factorize_existing_model


def print_table(data: dict, title: str):
    """Imprimir tabla formateada."""
    print(f"\n{title}")
    print("-" * 50)
    for key, value in data.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:>10.2f} MB")
        else:
            print(f"  {key:.<30} {value:>10}")


def analyze_model(model: nn.Module, name: str, ranks: list):
    """Analizar un modelo con diferentes rangos."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 60}")
    print(f"Modelo: {name}")
    print(f"Parametros totales: {total_params:,}")
    print(f"{'=' * 60}")

    for rank in ranks:
        result = estimate_memory_savings(model, rank=rank)

        print(f"\n--- Rango r={rank} ---")
        print_table(result["traditional"], "Entrenamiento Tradicional")
        print_table(result["mztrain"], f"MZTrain (r={rank})")

        savings = result["savings"]
        print(f"\n  Ahorro total:     {savings['total_pct']:.1f}%")
        print(f"  Ahorro en params: {savings['params_pct']:.1f}%")
        print(f"  Factor:           {savings['factor']:.1f}x")

        # Factorizar para ver datos reales
        z_model, stats = factorize_existing_model(model, rank=rank)
        print(f"  Capas reemplazadas: {len(stats['replaced_layers'])}")
        print(f"  Capas omitidas:     {len(stats['skipped_layers'])}")


def main():
    print("MZTrain - Estimacion de Ahorro de Memoria")

    # Modelo 1: MLP simple
    mlp = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    analyze_model(mlp, "MLP (MNIST-like)", [8, 16, 32, 64])

    # Modelo 2: Red profunda
    layers = []
    for i in range(10):
        layers.append(nn.Linear(512, 512))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(512, 10))
    deep_net = nn.Sequential(*layers)
    analyze_model(deep_net, "Red Profunda (10 capas x 512)", [16, 32, 64])

    # Modelo 3: Red ancha
    wide_net = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
    )
    analyze_model(wide_net, "Red Ancha (4096 ocultas)", [32, 64, 128])

    print(f"\n{'=' * 60}")
    print("Analisis completado.")


if __name__ == "__main__":
    main()
