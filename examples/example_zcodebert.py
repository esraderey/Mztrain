#!/usr/bin/env python3
"""
MZTrain - Ejemplo de entrenamiento ZCodeBERT.

Demuestra como entrenar un modelo ZCodeBERT con:
- Factorizacion SVD de MZTrain (3-5x menos parametros)
- Compresion de activaciones via ZActivationCheckpoint
- Optimizer comprimido ZCompressedAdam
- Crecimiento progresivo de rango
- Compresion MNEME post-entrenamiento
- Almacenamiento seguro con MNEME SecureStorageBackend

Usa datos sinteticos multi-lenguaje generados por SyntheticCodeGenerator.
"""

import time
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mztrain import (
    ZTrainConfig,
    RankSchedule,
    ZCompressedAdam,
    ZRankScheduler,
    ZGradientCompressor,
    ZActivationCheckpoint,
)
from mztrain.data import CodeTokenizer, SyntheticCodeGenerator, CodeDataset, MLMDataset
from mztrain.models import ZCodeBERTConfig, ZCodeBERTForMLM

# Intentar importar MNEME
try:
    from mneme import compress_model, get_compression_stats
    HAS_MNEME = True
except ImportError:
    HAS_MNEME = False


def get_config():
    """Obtener configuracion adaptada al hardware disponible."""
    if torch.cuda.is_available():
        # GPU: config grande
        return ZCodeBERTConfig(
            vocab_size=50000,
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=256,
            rank=32,
            hidden_dropout_prob=0.1,
            use_activation_checkpointing=True,
        )
    else:
        # CPU: config reducida para demo
        return ZCodeBERTConfig.small()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("mztrain")

    print("=" * 70)
    print("ZCodeBERT - Modelo de Comprension de Codigo con MZTrain + MNEME")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"MNEME disponible: {HAS_MNEME}")

    # =========================================================================
    # 1. Configuracion del modelo
    # =========================================================================
    config = get_config()
    print(f"\n--- Configuracion ---")
    print(f"  Hidden size:     {config.hidden_size}")
    print(f"  Num layers:      {config.num_hidden_layers}")
    print(f"  Num heads:       {config.num_attention_heads}")
    print(f"  Intermediate:    {config.intermediate_size}")
    print(f"  Max seq len:     {config.max_position_embeddings}")
    print(f"  Rank:            {config.rank}")
    print(f"  Vocab size:      {config.vocab_size}")

    # =========================================================================
    # 2. Crear modelo
    # =========================================================================
    print(f"\n--- Creando modelo ---")
    model = ZCodeBERTForMLM(config).to(device)

    stats = model.get_model_stats()
    print(f"  Parametros reales:       {stats['total_params']:,}")
    print(f"  Parametros equivalentes: {stats['equivalent_params']:,}")
    print(f"  Capas factorizadas:      {stats['factorized_layers']}")
    print(f"  Ratio compresion:        {stats['compression_ratio']:.3f}")
    print(f"  Ahorro:                  {(1 - stats['compression_ratio']) * 100:.1f}%")

    # =========================================================================
    # 3. Generar datos sinteticos
    # =========================================================================
    print(f"\n--- Generando datos sinteticos ---")
    generator = SyntheticCodeGenerator(seed=42)
    tokenizer = CodeTokenizer(vocab_size=config.vocab_size, max_length=config.max_position_embeddings)

    n_samples = 200 if device.type == "cpu" else 1000
    snippets = generator.generate(n_samples)
    print(f"  Snippets generados: {len(snippets)}")
    print(f"  Lenguajes: Python, JavaScript, Java, Go, Ruby, PHP")
    print(f"  Ejemplo: {snippets[0][:80]}...")

    # Crear datasets
    code_dataset = CodeDataset(snippets, tokenizer, max_length=config.max_position_embeddings)
    mlm_dataset = MLMDataset(code_dataset, mask_prob=0.15, seed=42)

    n_train = int(0.8 * len(mlm_dataset))
    n_val = len(mlm_dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        mlm_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = 8 if device.type == "cpu" else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"  Train samples: {n_train}, Val samples: {n_val}")
    print(f"  Batch size: {batch_size}")

    # =========================================================================
    # 4. Configurar optimizer y scheduler
    # =========================================================================
    optimizer = ZCompressedAdam(
        model.parameters(),
        lr=1e-4 if device.type == "cuda" else 5e-4,
        weight_decay=0.01,
        compress_states=True,
    )

    n_epochs = 5 if device.type == "cpu" else 15
    rank_scheduler = ZRankScheduler(
        initial_rank=config.rank,
        max_rank=config.rank * 4,
        total_epochs=n_epochs,
        schedule=RankSchedule.EXPONENTIAL,
        growth_interval=max(3, n_epochs // 3),
        growth_factor=2.0,
    )

    grad_compressor = ZGradientCompressor()

    print(f"\n--- Entrenamiento ---")
    print(f"  Epochs: {n_epochs}")
    print(f"  Rango inicial: {config.rank}")
    print(f"  Rango maximo: {config.rank * 4}")
    print(f"  Schedule: EXPONENTIAL")

    # =========================================================================
    # 5. Training loop
    # =========================================================================
    ZActivationCheckpoint.reset()
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Crecimiento de rango
        new_rank = rank_scheduler.get_rank(epoch)
        if new_rank > rank_scheduler.current_rank:
            print(f"\n  >>> Crecimiento de rango: {rank_scheduler.current_rank} -> {new_rank}")
            model.grow_rank(new_rank)
            rank_scheduler.current_rank = new_rank
            # Re-crear optimizer con nuevos parametros
            optimizer = ZCompressedAdam(
                model.parameters(),
                lr=optimizer.defaults["lr"],
                weight_decay=optimizer.defaults["weight_decay"],
                compress_states=True,
            )

        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            masked_ids, attn_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            loss, logits = model(masked_ids, attention_mask=attn_mask, labels=labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                masked_ids, attn_mask, labels = [b.to(device) for b in batch]
                loss, _ = model(masked_ids, attention_mask=attn_mask, labels=labels)
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)
        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " *"
        else:
            marker = ""

        print(
            f"  Epoch {epoch:2d}/{n_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"rank={rank_scheduler.current_rank}, "
            f"time={epoch_time:.1f}s{marker}"
        )

    total_time = time.time() - start_time

    # =========================================================================
    # 6. Resultados
    # =========================================================================
    print(f"\n--- Resultados ---")
    print(f"  Tiempo total:    {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Mejor val_loss:  {best_val_loss:.4f}")
    print(f"  Rango final:     {rank_scheduler.current_rank}")

    # Stats del optimizer
    opt_stats = optimizer.get_memory_stats()
    print(f"  Optimizer stats: {opt_stats}")

    # Stats de activation checkpoint
    ckpt_stats = ZActivationCheckpoint.get_stats()
    print(f"  Activaciones comprimidas: {ckpt_stats['total_compressed']}")

    # =========================================================================
    # 7. Compresion MNEME post-entrenamiento
    # =========================================================================
    if HAS_MNEME:
        print(f"\n--- Compresion MNEME post-entrenamiento ---")
        mneme_stats = model.compress_with_mneme(compression_level="balanced")
        if mneme_stats:
            print(f"  Capas comprimidas: {mneme_stats.get('compressed_layers', 'N/A')}")
            print(f"  Params originales: {mneme_stats.get('original_params', 'N/A'):,}")
            print(f"  Params comprimidos: {mneme_stats.get('compressed_params', 'N/A'):,}")
    else:
        print(f"\n  (MNEME no disponible - saltando compresion post-entrenamiento)")

    # =========================================================================
    # 8. Guardar modelo
    # =========================================================================
    print(f"\n--- Guardando modelo ---")
    save_path = "zcodebert_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "stats": model.get_model_stats(),
    }, save_path)
    print(f"  Guardado en: {save_path}")

    # =========================================================================
    # 9. Resumen final
    # =========================================================================
    final_stats = model.get_model_stats()
    print(f"\n{'=' * 70}")
    print(f"ZCodeBERT - Resumen Final")
    print(f"{'=' * 70}")
    print(f"  Parametros reales:       {final_stats['total_params']:,}")
    print(f"  Parametros equivalentes: {final_stats['equivalent_params']:,}")
    print(f"  Compresion factorizacion: {(1 - final_stats['compression_ratio']) * 100:.1f}%")
    print(f"  Capas factorizadas:      {final_stats['factorized_layers']}")
    print(f"  Rango final:             {final_stats['rank']}")
    print(f"  MNEME compresion:        {'Si' if HAS_MNEME else 'No disponible'}")
    print(f"  Tiempo entrenamiento:    {total_time:.1f}s")
    print(f"{'=' * 70}")
    print(f"Ejemplo completado.")


if __name__ == "__main__":
    main()
