#!/usr/bin/env python3
"""
ZCodeBERT - Analisis Completo, Entrenamiento Largo y Pruebas de Precision.

MSC Star Team (Esraderey y Raul Cruz Acosta)

Este script ejecuta:
1. Analisis arquitectonico detallado (parametros, compresion, memoria)
2. Entrenamiento largo (~1 hora) con metricas completas
3. Pruebas de precision (MLM accuracy, top-k accuracy, perplexity)
4. Prueba de coherencia: prediccion de codigo real

Hardware target: NVIDIA RTX 4060 8GB VRAM
"""

import os
import sys
import time
import json
import math
import random
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# MZTrain
from mztrain import (
    ZTrainConfig, RankSchedule, GradientCompression,
    ZCompressedAdam, ZRankScheduler, ZGradientCompressor,
    ZActivationCheckpoint, ZFactorizedLinear,
)
from mztrain.data import CodeTokenizer, SyntheticCodeGenerator, CodeDataset, MLMDataset
from mztrain.models import ZCodeBERTConfig, ZCodeBERTForMLM, ZCodeBERTForSequenceClassification

# MNEME
try:
    from mneme import compress_model, get_compression_stats, CompressionConfig
    HAS_MNEME = True
except ImportError:
    HAS_MNEME = False

# ============================================================================
# Configuracion
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_HOURS = 1.0  # Tiempo maximo de entrenamiento
RESULTS_FILE = "analysis_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analysis")


def fmt_num(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


# ============================================================================
# FASE 1: ANALISIS ARQUITECTONICO
# ============================================================================

def phase1_architecture_analysis():
    """Analisis detallado de la arquitectura ZCodeBERT."""
    print("\n" + "=" * 80)
    print(" FASE 1: ANALISIS ARQUITECTONICO")
    print("=" * 80)

    results = {}

    # --- 1.1 Configs disponibles ---
    configs = {
        "Small (test)": ZCodeBERTConfig.small(),
        "Base (BERT-Base)": ZCodeBERTConfig.base(),
        "Large (BERT-Large)": ZCodeBERTConfig.large(),
    }

    print("\n--- 1.1 Configuraciones Disponibles ---")
    print(f"{'Config':<22} {'Layers':>6} {'Hidden':>7} {'Heads':>6} {'Rank':>5} {'MLP':>5}")
    print("-" * 60)
    for name, cfg in configs.items():
        print(f"{name:<22} {cfg.num_hidden_layers:>6} {cfg.hidden_size:>7} "
              f"{cfg.num_attention_heads:>6} {cfg.rank:>5} {cfg.intermediate_size:>5}")

    # --- 1.2 Analisis de parametros por config ---
    print("\n--- 1.2 Analisis de Parametros ---")
    print(f"{'Config':<22} {'Reales':>12} {'Equivalentes':>14} {'Compresion':>11} {'Capas Fact':>11}")
    print("-" * 75)

    for name, cfg in configs.items():
        model = ZCodeBERTForMLM(cfg)
        stats = model.get_model_stats()
        results[name] = {
            "real_params": stats["total_params"],
            "equiv_params": stats["equivalent_params"],
            "compression": f"{(1 - stats['compression_ratio']) * 100:.1f}%",
            "factorized_layers": stats["factorized_layers"],
        }
        print(f"{name:<22} {fmt_num(stats['total_params']):>12} "
              f"{fmt_num(stats['equivalent_params']):>14} "
              f"{(1 - stats['compression_ratio']) * 100:>10.1f}% "
              f"{stats['factorized_layers']:>11}")
        del model

    # --- 1.3 Modelo de entrenamiento: analisis profundo ---
    # Config para RTX 4060 8GB: balancear tamano y batch size
    train_config = ZCodeBERTConfig(
        vocab_size=50000,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=256,
        rank=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        use_activation_checkpointing=False,
    )

    model = ZCodeBERTForMLM(train_config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n--- 1.3 Modelo de Entrenamiento (GPU config) ---")
    print(f"  Config:                  512h / 12L / 8H / r=32")
    print(f"  Parametros totales:      {total_params:,}")
    print(f"  Parametros entrenables:  {trainable_params:,}")
    print(f"  Device:                  {DEVICE}")

    # Analisis por tipo de capa
    layer_analysis = defaultdict(lambda: {"count": 0, "params": 0, "equiv": 0})

    for name, module in model.named_modules():
        if isinstance(module, ZFactorizedLinear):
            layer_analysis["ZFactorizedLinear"]["count"] += 1
            layer_analysis["ZFactorizedLinear"]["params"] += module.num_parameters
            layer_analysis["ZFactorizedLinear"]["equiv"] += module.full_parameters
        elif isinstance(module, nn.Linear):
            p = sum(pp.numel() for pp in module.parameters())
            layer_analysis["nn.Linear"]["count"] += 1
            layer_analysis["nn.Linear"]["params"] += p
            layer_analysis["nn.Linear"]["equiv"] += p
        elif isinstance(module, nn.Embedding):
            p = sum(pp.numel() for pp in module.parameters())
            layer_analysis["nn.Embedding"]["count"] += 1
            layer_analysis["nn.Embedding"]["params"] += p
            layer_analysis["nn.Embedding"]["equiv"] += p
        elif isinstance(module, nn.LayerNorm):
            p = sum(pp.numel() for pp in module.parameters())
            layer_analysis["nn.LayerNorm"]["count"] += 1
            layer_analysis["nn.LayerNorm"]["params"] += p
            layer_analysis["nn.LayerNorm"]["equiv"] += p

    print(f"\n  Desglose por tipo de capa:")
    print(f"  {'Tipo':<25} {'Count':>6} {'Params':>12} {'Equiv':>12} {'Compress':>10}")
    print(f"  {'-' * 70}")
    for ltype, info in sorted(layer_analysis.items()):
        comp = (1 - info["params"] / max(info["equiv"], 1)) * 100
        print(f"  {ltype:<25} {info['count']:>6} {fmt_num(info['params']):>12} "
              f"{fmt_num(info['equiv']):>12} {comp:>9.1f}%")

    # Memoria estimada
    param_mem = total_params * 4 / (1024 * 1024)  # float32
    grad_mem = trainable_params * 4 / (1024 * 1024)
    opt_mem = trainable_params * 8 / (1024 * 1024)  # Adam: m + v
    opt_mem_compressed = opt_mem * 0.25  # INT8 compression 75%

    print(f"\n  Estimacion de Memoria (float32):")
    print(f"  {'Componente':<30} {'Tradicional':>12} {'MZTrain':>12} {'Ahorro':>10}")
    print(f"  {'-' * 68}")

    stats_model = model.get_model_stats()
    equiv_param_mem = stats_model["equivalent_params"] * 4 / (1024 * 1024)
    equiv_grad_mem = equiv_param_mem
    equiv_opt_mem = equiv_param_mem * 2

    print(f"  {'Parametros':<30} {equiv_param_mem:>10.1f}MB {param_mem:>10.1f}MB "
          f"{(1 - param_mem / equiv_param_mem) * 100:>9.1f}%")
    print(f"  {'Gradientes':<30} {equiv_grad_mem:>10.1f}MB {grad_mem:>10.1f}MB "
          f"{(1 - grad_mem / equiv_grad_mem) * 100:>9.1f}%")
    print(f"  {'Optimizer (Adam m+v)':<30} {equiv_opt_mem:>10.1f}MB {opt_mem_compressed:>10.1f}MB "
          f"{(1 - opt_mem_compressed / equiv_opt_mem) * 100:>9.1f}%")
    total_trad = equiv_param_mem + equiv_grad_mem + equiv_opt_mem
    total_mz = param_mem + grad_mem + opt_mem_compressed
    print(f"  {'TOTAL':<30} {total_trad:>10.1f}MB {total_mz:>10.1f}MB "
          f"{(1 - total_mz / total_trad) * 100:>9.1f}%")

    results["memory_analysis"] = {
        "traditional_total_mb": round(total_trad, 1),
        "mztrain_total_mb": round(total_mz, 1),
        "savings_pct": round((1 - total_mz / total_trad) * 100, 1),
    }

    # Forward pass benchmark
    print(f"\n  Benchmark Forward Pass:")
    model.eval()
    batch = torch.randint(0, 50000, (16, 256), device=DEVICE)
    mask = torch.ones(16, 256, device=DEVICE)
    labels = torch.randint(0, 50000, (16, 256), device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(batch, attention_mask=mask, labels=labels)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(20):
            start = time.perf_counter()
            model(batch, attention_mask=mask, labels=labels)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    peak_mem = gpu_mem_mb()
    tokens_per_sec = (16 * 256) / avg_time

    print(f"  Tiempo promedio (bs=16, seq=256): {avg_time * 1000:.1f}ms")
    print(f"  Tokens/segundo:                   {tokens_per_sec:,.0f}")
    print(f"  Memoria GPU pico:                 {peak_mem:.0f}MB")

    results["benchmark"] = {
        "avg_forward_ms": round(avg_time * 1000, 1),
        "tokens_per_sec": int(tokens_per_sec),
        "peak_gpu_mb": round(peak_mem, 0),
    }

    del model, batch, mask, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, train_config


# ============================================================================
# FASE 2: ENTRENAMIENTO LARGO (1 HORA)
# ============================================================================

def phase2_long_training(train_config):
    """Entrenamiento largo con metricas completas."""
    print("\n" + "=" * 80)
    print(" FASE 2: ENTRENAMIENTO LARGO (~1 HORA)")
    print("=" * 80)

    results = {
        "train_losses": [],
        "val_losses": [],
        "mlm_accuracies": [],
        "top5_accuracies": [],
        "perplexities": [],
        "ranks": [],
        "learning_rates": [],
        "epoch_times": [],
        "gpu_memory_mb": [],
    }

    # --- 2.1 Generar datos ---
    print(f"\n--- 2.1 Generando datos de entrenamiento ---")
    gen = SyntheticCodeGenerator(seed=42)
    tok = CodeTokenizer(vocab_size=train_config.vocab_size, max_length=train_config.max_position_embeddings)

    n_samples = 10000
    snippets = gen.generate(n_samples)
    log.info(f"Snippets generados: {n_samples} (6 lenguajes)")

    # Estadisticas de datos
    avg_len = sum(len(s) for s in snippets) / len(snippets)
    print(f"  Snippets totales:        {n_samples}")
    print(f"  Longitud promedio:       {avg_len:.0f} caracteres")

    code_ds = CodeDataset(snippets, tok, max_length=train_config.max_position_embeddings)
    mlm_ds = MLMDataset(code_ds, mask_prob=0.15, seed=42)

    n_train = int(0.85 * len(mlm_ds))
    n_val = len(mlm_ds) - n_train
    train_ds, val_ds = random_split(
        mlm_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"  Train:                   {n_train} samples ({len(train_loader)} batches)")
    print(f"  Validation:              {n_val} samples ({len(val_loader)} batches)")
    print(f"  Batch size:              {batch_size}")
    print(f"  Mask prob:               15%")

    # --- 2.2 Crear modelo ---
    print(f"\n--- 2.2 Creando modelo ---")
    model = ZCodeBERTForMLM(train_config).to(DEVICE)
    stats = model.get_model_stats()
    print(f"  Params reales:           {stats['total_params']:,}")
    print(f"  Params equivalentes:     {stats['equivalent_params']:,}")
    print(f"  Compresion:              {(1 - stats['compression_ratio']) * 100:.1f}%")

    # --- 2.3 Optimizer con compresion ---
    lr_init = 5e-5
    optimizer = ZCompressedAdam(
        model.parameters(),
        lr=lr_init,
        weight_decay=0.01,
        compress_states=True,
        betas=(0.9, 0.98),
    )

    # Scheduler: warmup + cosine decay
    warmup_epochs = 3
    total_steps_estimate = len(train_loader) * 200  # Max epochs estimate

    # Rank scheduler
    rank_scheduler = ZRankScheduler(
        initial_rank=train_config.rank,
        max_rank=128,
        total_epochs=200,
        schedule=RankSchedule.EXPONENTIAL,
        growth_interval=20,
        growth_factor=2.0,
    )

    print(f"  Optimizer:               ZCompressedAdam (lr={lr_init}, wd=0.01)")
    print(f"  Rank schedule:           EXPONENTIAL (32 -> 128, cada 20 epochs)")
    print(f"  Gradient compression:    None (pure Adam)")
    print(f"  Tiempo objetivo:         {TRAIN_HOURS} hora(s)")

    # --- 2.4 Training loop ---
    print(f"\n--- 2.4 Entrenamiento ---")
    print(f"  {'Epoch':>5} {'Train Loss':>11} {'Val Loss':>10} {'MLM Acc':>8} "
          f"{'Top5 Acc':>9} {'PPL':>8} {'Rank':>5} {'LR':>10} {'Time':>6} {'GPU MB':>7}")
    print(f"  {'-' * 95}")

    ZActivationCheckpoint.reset()
    max_time = TRAIN_HOURS * 3600
    training_start = time.time()
    best_val_loss = float("inf")
    best_state = None
    patience = 20
    patience_counter = 0
    epoch = 0
    global_step = 0

    while True:
        elapsed = time.time() - training_start
        if elapsed >= max_time:
            print(f"\n  >>> Tiempo limite alcanzado: {elapsed/60:.1f} min")
            break

        epoch_start = time.time()

        # Rank growth
        new_rank = rank_scheduler.get_rank(epoch)
        if new_rank > rank_scheduler.current_rank:
            log.info(f"Rango: {rank_scheduler.current_rank} -> {new_rank}")
            model.grow_rank(new_rank)
            rank_scheduler.current_rank = new_rank
            optimizer = ZCompressedAdam(
                model.parameters(),
                lr=lr,  # Mantener LR actual
                weight_decay=0.01,
                compress_states=True,
            )

        # Learning rate warmup
        if epoch < warmup_epochs:
            lr = lr_init * (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / max(200 - warmup_epochs, 1)
            lr = lr_init * 0.5 * (1 + math.cos(math.pi * progress))
            lr = max(lr, 1e-6)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train epoch
        model.train()
        total_loss = 0
        total_correct = 0
        total_top5_correct = 0
        total_masked = 0
        n_batches = 0

        for batch in train_loader:
            masked_ids, attn_mask, labels = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            loss, logits = model(masked_ids, attention_mask=attn_mask, labels=labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accuracy en posiciones maskeadas
            mask_positions = labels != -100
            if mask_positions.any():
                preds = logits[mask_positions].argmax(dim=-1)
                targets = labels[mask_positions]
                total_correct += (preds == targets).sum().item()

                # Top-5
                top5 = logits[mask_positions].topk(5, dim=-1).indices
                total_top5_correct += (top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item()

                total_masked += mask_positions.sum().item()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

        train_loss = total_loss / max(n_batches, 1)
        mlm_acc = total_correct / max(total_masked, 1) * 100
        top5_acc = total_top5_correct / max(total_masked, 1) * 100

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_top5_correct = 0
        val_masked = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                masked_ids, attn_mask, labels = [b.to(DEVICE) for b in batch]
                loss, logits = model(masked_ids, attention_mask=attn_mask, labels=labels)
                val_loss += loss.item()
                val_batches += 1

                mask_positions = labels != -100
                if mask_positions.any():
                    preds = logits[mask_positions].argmax(dim=-1)
                    targets = labels[mask_positions]
                    val_correct += (preds == targets).sum().item()
                    top5 = logits[mask_positions].topk(5, dim=-1).indices
                    val_top5_correct += (top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
                    val_masked += mask_positions.sum().item()

        val_loss = val_loss / max(val_batches, 1)
        val_mlm_acc = val_correct / max(val_masked, 1) * 100
        val_top5_acc = val_top5_correct / max(val_masked, 1) * 100
        try:
            perplexity = min(math.exp(min(val_loss, 20.0)), 99999.0)
        except OverflowError:
            perplexity = 99999.0

        epoch_time = time.time() - epoch_start
        mem = gpu_mem_mb()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        # Log
        print(f"  {epoch:>5} {train_loss:>11.4f} {val_loss:>10.4f} {val_mlm_acc:>7.2f}% "
              f"{val_top5_acc:>8.2f}% {perplexity:>8.1f} {rank_scheduler.current_rank:>5} "
              f"{lr:>10.2e} {epoch_time:>5.1f}s {mem:>6.0f}M{marker}")

        # Record metrics
        results["train_losses"].append(round(train_loss, 4))
        results["val_losses"].append(round(val_loss, 4))
        results["mlm_accuracies"].append(round(val_mlm_acc, 2))
        results["top5_accuracies"].append(round(val_top5_acc, 2))
        results["perplexities"].append(round(perplexity, 1))
        results["ranks"].append(rank_scheduler.current_rank)
        results["learning_rates"].append(lr)
        results["epoch_times"].append(round(epoch_time, 1))
        results["gpu_memory_mb"].append(round(mem, 0))

        if patience_counter >= patience:
            print(f"\n  >>> Early stopping en epoch {epoch} (paciencia: {patience})")
            break

        epoch += 1

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    total_time = time.time() - training_start
    results["total_epochs"] = epoch + 1
    results["total_time_s"] = round(total_time, 1)
    results["best_val_loss"] = round(best_val_loss, 4)
    results["final_rank"] = rank_scheduler.current_rank
    results["global_steps"] = global_step

    print(f"\n  Resumen entrenamiento:")
    print(f"  Epochs completados:      {epoch + 1}")
    print(f"  Tiempo total:            {total_time/60:.1f} min")
    print(f"  Steps totales:           {global_step:,}")
    print(f"  Mejor val_loss:          {best_val_loss:.4f}")
    print(f"  Mejor MLM accuracy:      {max(results['mlm_accuracies']):.2f}%")
    print(f"  Mejor Top-5 accuracy:    {max(results['top5_accuracies']):.2f}%")
    print(f"  Mejor perplexity:        {min(results['perplexities']):.1f}")
    print(f"  Rango final:             {rank_scheduler.current_rank}")

    # Optimizer stats
    opt_stats = optimizer.get_memory_stats()
    print(f"\n  Optimizer ZCompressedAdam:")
    print(f"  Params comprimidos:      {opt_stats['compressed_params']}")
    print(f"  Memoria estados:         {opt_stats['state_memory_mb']:.1f} MB")
    print(f"  Memoria sin comprimir:   {opt_stats['full_memory_mb']:.1f} MB")
    print(f"  Ahorro:                  {opt_stats['memory_saved_pct']:.0f}%")

    results["optimizer_stats"] = opt_stats

    return results, model, tok, train_config


# ============================================================================
# FASE 3: PRUEBAS DE PRECISION
# ============================================================================

def phase3_precision_tests(model, tokenizer, config):
    """Pruebas de precision detalladas."""
    print("\n" + "=" * 80)
    print(" FASE 3: PRUEBAS DE PRECISION")
    print("=" * 80)

    results = {}
    model.eval()

    # --- 3.1 MLM Precision por tipo de token ---
    print(f"\n--- 3.1 Precision por Tipo de Token ---")

    test_cases = {
        "Keywords Python": [
            "def {name}({param}):\n    return {param} + 1",
            "for item in {name}:\n    if item > 0:\n        print(item)",
            "class {name}:\n    def __init__(self):\n        self.data = []",
            "try:\n    result = int(value)\nexcept ValueError:\n    result = None",
            "import os\nimport sys\nfrom pathlib import Path",
        ],
        "Keywords JavaScript": [
            "function {name}(data) {{\n    return data.map(x => x * 2);\n}}",
            "const {name} = async () => {{\n    const res = await fetch(url);\n    return res.json();\n}};",
            "class {name} extends Base {{\n    constructor() {{\n        super();\n    }}\n}}",
        ],
        "Operadores": [
            "result = a + b * c - d / e",
            "if x >= 0 and x <= 100:",
            "data = [x for x in range(10) if x % 2 == 0]",
            "flag = (a == b) or (c != d) and not e",
        ],
        "Estructuras": [
            "for i in range(len(items)):\n    total += items[i]",
            "while count > 0:\n    count -= 1\n    process(count)",
            "if condition:\n    do_this()\nelse:\n    do_that()",
        ],
        "Strings y literales": [
            'name = "hello world"',
            "path = '/usr/local/bin'",
            "count = 42\npi = 3.14159",
        ],
    }

    names = ["data", "result", "items", "config", "handler", "manager"]

    category_results = {}

    for category, templates in test_cases.items():
        correct = 0
        top5_correct = 0
        total = 0

        for template in templates:
            code = template.replace("{name}", random.choice(names)).replace("{param}", "value")
            ids = tokenizer.encode(code, max_length=config.max_position_embeddings)
            input_ids = torch.tensor([ids], device=DEVICE)
            attn_mask = (input_ids != 0).long()

            # Maskear tokens no especiales
            labels = torch.full_like(input_ids, -100)
            maskable = []
            for i in range(len(ids)):
                if ids[i] not in (0, 1, 2, 3, 4):  # Skip special tokens
                    maskable.append(i)

            if not maskable:
                continue

            # Maskear ~30% para testing
            n_mask = max(1, len(maskable) // 3)
            mask_indices = random.sample(maskable, min(n_mask, len(maskable)))

            masked_input = input_ids.clone()
            for idx in mask_indices:
                labels[0, idx] = input_ids[0, idx]
                masked_input[0, idx] = 4  # MASK_ID

            with torch.no_grad():
                _, logits = model(masked_input, attention_mask=attn_mask, labels=labels)

            for idx in mask_indices:
                pred = logits[0, idx].argmax().item()
                target = input_ids[0, idx].item()

                if pred == target:
                    correct += 1

                top5_preds = logits[0, idx].topk(5).indices.tolist()
                if target in top5_preds:
                    top5_correct += 1

                total += 1

        acc = correct / max(total, 1) * 100
        top5 = top5_correct / max(total, 1) * 100
        category_results[category] = {"accuracy": round(acc, 2), "top5": round(top5, 2), "total": total}
        print(f"  {category:<25} Acc: {acc:>6.2f}% | Top-5: {top5:>6.2f}% | N={total}")

    results["per_category"] = category_results

    # --- 3.2 Perplexity en codigo fresh ---
    print(f"\n--- 3.2 Perplexity en Codigo Nuevo ---")

    gen = SyntheticCodeGenerator(seed=999)  # Seed diferente = datos no vistos
    fresh_snippets = gen.generate(200)
    fresh_ds = CodeDataset(fresh_snippets, tokenizer, max_length=config.max_position_embeddings)
    fresh_mlm = MLMDataset(fresh_ds, mask_prob=0.15, seed=999)
    fresh_loader = DataLoader(fresh_mlm, batch_size=16)

    total_loss = 0
    total_correct = 0
    total_top5 = 0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in fresh_loader:
            masked_ids, attn_mask, labels = [b.to(DEVICE) for b in batch]
            loss, logits = model(masked_ids, attention_mask=attn_mask, labels=labels)
            total_loss += loss.item()
            n_batches += 1

            mask_pos = labels != -100
            if mask_pos.any():
                preds = logits[mask_pos].argmax(dim=-1)
                targets = labels[mask_pos]
                total_correct += (preds == targets).sum().item()
                top5 = logits[mask_pos].topk(5, dim=-1).indices
                total_top5 += (top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
                total_tokens += mask_pos.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    try:
        ppl = min(math.exp(min(avg_loss, 20.0)), 99999.0)
    except OverflowError:
        ppl = 99999.0
    fresh_acc = total_correct / max(total_tokens, 1) * 100
    fresh_top5 = total_top5 / max(total_tokens, 1) * 100

    print(f"  Loss promedio:           {avg_loss:.4f}")
    print(f"  Perplexity:              {ppl:.1f}")
    print(f"  MLM Accuracy:            {fresh_acc:.2f}%")
    print(f"  Top-5 Accuracy:          {fresh_top5:.2f}%")
    print(f"  Tokens evaluados:        {total_tokens:,}")

    results["fresh_data"] = {
        "loss": round(avg_loss, 4),
        "perplexity": round(ppl, 1),
        "accuracy": round(fresh_acc, 2),
        "top5_accuracy": round(fresh_top5, 2),
        "tokens": total_tokens,
    }

    # --- 3.3 Analisis de confianza de predicciones ---
    print(f"\n--- 3.3 Analisis de Confianza ---")

    confidences = []
    correct_confidences = []
    wrong_confidences = []

    with torch.no_grad():
        for batch in fresh_loader:
            masked_ids, attn_mask, labels = [b.to(DEVICE) for b in batch]
            _, logits = model(masked_ids, attention_mask=attn_mask, labels=labels)

            mask_pos = labels != -100
            if mask_pos.any():
                probs = F.softmax(logits[mask_pos], dim=-1)
                max_probs = probs.max(dim=-1).values
                preds = probs.argmax(dim=-1)
                targets = labels[mask_pos]

                for prob, pred, target in zip(max_probs.cpu(), preds.cpu(), targets.cpu()):
                    conf = prob.item()
                    confidences.append(conf)
                    if pred == target:
                        correct_confidences.append(conf)
                    else:
                        wrong_confidences.append(conf)

    avg_conf = sum(confidences) / max(len(confidences), 1)
    avg_correct = sum(correct_confidences) / max(len(correct_confidences), 1) if correct_confidences else 0
    avg_wrong = sum(wrong_confidences) / max(len(wrong_confidences), 1) if wrong_confidences else 0

    print(f"  Confianza promedio:      {avg_conf:.4f}")
    print(f"  Confianza (correctas):   {avg_correct:.4f}")
    print(f"  Confianza (incorrectas): {avg_wrong:.4f}")
    print(f"  Separacion:              {avg_correct - avg_wrong:.4f}")

    results["confidence"] = {
        "avg": round(avg_conf, 4),
        "correct": round(avg_correct, 4),
        "wrong": round(avg_wrong, 4),
        "separation": round(avg_correct - avg_wrong, 4),
    }

    # --- 3.4 Clasificacion de lenguaje (fine-tune rapido) ---
    print(f"\n--- 3.4 Clasificacion de Lenguaje (Fine-tune) ---")

    clf_config = ZCodeBERTConfig(
        vocab_size=50000,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rank=config.rank,
        hidden_dropout_prob=0.1,
        use_activation_checkpointing=False,
    )

    clf_model = ZCodeBERTForSequenceClassification(clf_config, num_labels=6).to(DEVICE)

    # Datos por lenguaje
    languages = ["python", "javascript", "java", "go", "ruby", "php"]
    clf_inputs = []
    clf_labels = []

    for lang_id, lang in enumerate(languages):
        lang_gen = SyntheticCodeGenerator(languages=[lang], seed=lang_id + 100)
        lang_snippets = lang_gen.generate(50)
        for snippet in lang_snippets:
            ids = tokenizer.encode(snippet, max_length=config.max_position_embeddings)
            clf_inputs.append(ids)
            clf_labels.append(lang_id)

    indices = list(range(len(clf_inputs)))
    random.shuffle(indices)
    clf_inputs = [clf_inputs[i] for i in indices]
    clf_labels = [clf_labels[i] for i in indices]

    n_clf_train = int(0.8 * len(clf_inputs))
    train_x = torch.tensor(clf_inputs[:n_clf_train], device=DEVICE)
    train_y = torch.tensor(clf_labels[:n_clf_train], device=DEVICE)
    test_x = torch.tensor(clf_inputs[n_clf_train:], device=DEVICE)
    test_y = torch.tensor(clf_labels[n_clf_train:], device=DEVICE)

    clf_opt = torch.optim.Adam(clf_model.parameters(), lr=5e-4)

    # Train 20 epochs
    for ep in range(20):
        clf_model.train()
        for i in range(0, len(train_x), 16):
            bx = train_x[i:i+16]
            by = train_y[i:i+16]
            mask = (bx != 0).long()
            clf_opt.zero_grad()
            loss, _ = clf_model(bx, attention_mask=mask, labels=by)
            loss.backward()
            clf_opt.step()

    # Eval
    clf_model.eval()
    with torch.no_grad():
        mask = (test_x != 0).long()
        _, logits = clf_model(test_x, attention_mask=mask)
        preds = logits.argmax(dim=-1)
        clf_acc = (preds == test_y).float().mean().item() * 100

    # Per-class accuracy
    print(f"  Clasificacion global:    {clf_acc:.1f}%")
    per_lang = {}
    for lang_id, lang in enumerate(languages):
        mask_lang = test_y == lang_id
        if mask_lang.any():
            lang_acc = (preds[mask_lang] == test_y[mask_lang]).float().mean().item() * 100
            per_lang[lang] = round(lang_acc, 1)
            print(f"    {lang:<15}         {lang_acc:.1f}%")

    results["classification"] = {
        "overall_accuracy": round(clf_acc, 1),
        "per_language": per_lang,
    }

    del clf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ============================================================================
# FASE 4: PRUEBA DE COHERENCIA
# ============================================================================

def phase4_coherence_test(model, tokenizer, config):
    """Probar que el modelo genera predicciones coherentes en codigo real."""
    print("\n" + "=" * 80)
    print(" FASE 4: PRUEBA DE COHERENCIA - PREDICCION DE CODIGO REAL")
    print("=" * 80)

    results = {"predictions": []}
    model.eval()

    # Codigo real para probar prediccion
    test_codes = [
        {
            "name": "Funcion Python - Sort",
            "code": "def sort_list(items):\n    result = sorted(items, reverse=True)\n    return result",
            "description": "Funcion simple de ordenamiento",
        },
        {
            "name": "Clase Python - Stack",
            "code": "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()",
            "description": "Implementacion de stack",
        },
        {
            "name": "Loop con condicion",
            "code": "def filter_positive(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item)\n    return result",
            "description": "Filtrar elementos positivos",
        },
        {
            "name": "Try/Except",
            "code": "def safe_divide(a, b):\n    try:\n        result = a / b\n        return result\n    except ZeroDivisionError:\n        return None",
            "description": "Division segura con manejo de error",
        },
        {
            "name": "List comprehension",
            "code": "def transform(items):\n    return [x * 2 for x in items if x > 0]",
            "description": "Transformacion con comprehension",
        },
        {
            "name": "JavaScript async",
            "code": "async function fetchData(url) {\n    const response = await fetch(url);\n    const data = await response.json();\n    return data;\n}",
            "description": "Funcion async JavaScript",
        },
        {
            "name": "Java clase",
            "code": "public class Calculator {\n    private int result;\n\n    public Calculator() {\n        this.result = 0;\n    }\n\n    public int add(int value) {\n        this.result += value;\n        return this.result;\n    }\n}",
            "description": "Clase Java Calculator",
        },
        {
            "name": "Go funcion",
            "code": "func processItems(items []string) []string {\n    result := make([]string, 0)\n    for _, item := range items {\n        if len(item) > 0 {\n            result = append(result, item)\n        }\n    }\n    return result\n}",
            "description": "Funcion Go con slices",
        },
    ]

    total_correct = 0
    total_top5 = 0
    total_tokens = 0

    for test in test_codes:
        code = test["code"]
        ids = tokenizer.encode(code, max_length=config.max_position_embeddings)
        input_ids = torch.tensor([ids], device=DEVICE)
        attn_mask = (input_ids != 0).long()

        # Encontrar tokens maskeables (no especiales, no padding)
        maskable_indices = []
        for i, tid in enumerate(ids):
            if tid not in (0, 1, 2, 3, 4):
                maskable_indices.append(i)

        if not maskable_indices:
            continue

        # Maskear 5 tokens aleatorios
        n_mask = min(5, len(maskable_indices))
        mask_indices = random.sample(maskable_indices, n_mask)

        masked_input = input_ids.clone()
        original_tokens = []
        for idx in mask_indices:
            original_tokens.append((idx, ids[idx]))
            masked_input[0, idx] = 4  # MASK_ID

        labels = torch.full_like(input_ids, -100)
        for idx, orig_id in original_tokens:
            labels[0, idx] = orig_id

        with torch.no_grad():
            _, logits = model(masked_input, attention_mask=attn_mask, labels=labels)

        print(f"\n  >> {test['name']}: {test['description']}")
        print(f"     Codigo: {code[:80]}...")

        predictions = []
        for idx, orig_id in original_tokens:
            pred_id = logits[0, idx].argmax().item()
            top5_ids = logits[0, idx].topk(5).indices.tolist()
            top5_probs = F.softmax(logits[0, idx], dim=-1).topk(5)

            orig_token = tokenizer.id_to_token.get(orig_id, f"[{orig_id}]")
            pred_token = tokenizer.id_to_token.get(pred_id, f"[{pred_id}]")
            is_correct = pred_id == orig_id
            in_top5 = orig_id in top5_ids

            if is_correct:
                total_correct += 1
            if in_top5:
                total_top5 += 1
            total_tokens += 1

            status = "OK" if is_correct else ("~5" if in_top5 else "X ")
            conf = top5_probs.values[0].item()

            top5_tokens = [tokenizer.id_to_token.get(t, f"[{t}]") for t in top5_ids]

            print(f"     [{status}] pos={idx:>3}: "
                  f"orig='{orig_token}' pred='{pred_token}' "
                  f"conf={conf:.3f} top5={top5_tokens[:3]}")

            predictions.append({
                "position": idx,
                "original": orig_token,
                "predicted": pred_token,
                "correct": is_correct,
                "in_top5": in_top5,
                "confidence": round(conf, 4),
            })

        results["predictions"].append({
            "name": test["name"],
            "predictions": predictions,
        })

    overall_acc = total_correct / max(total_tokens, 1) * 100
    overall_top5 = total_top5 / max(total_tokens, 1) * 100

    print(f"\n  Resumen Coherencia:")
    print(f"  Tokens evaluados:        {total_tokens}")
    print(f"  Accuracy (exacta):       {overall_acc:.1f}%")
    print(f"  Top-5 Accuracy:          {overall_top5:.1f}%")

    results["coherence_accuracy"] = round(overall_acc, 1)
    results["coherence_top5"] = round(overall_top5, 1)
    results["total_tokens"] = total_tokens

    return results


# ============================================================================
# FASE 5: COMPRESION MNEME
# ============================================================================

def phase5_mneme_compression(model, config):
    """Comprimir modelo con MNEME si disponible."""
    print("\n" + "=" * 80)
    print(" FASE 5: COMPRESION MNEME POST-ENTRENAMIENTO")
    print("=" * 80)

    results = {"available": HAS_MNEME}

    if not HAS_MNEME:
        print(f"\n  MNEME no instalado - mostrando estimaciones teoricas")

        stats = model.get_model_stats()
        param_bytes = stats["total_params"] * 4
        # Estimacion MNEME: compresion tipica 93%+
        compressed_bytes = param_bytes * 0.07  # 93% compression

        print(f"\n  Modelo actual (MZTrain factorizado):")
        print(f"    Params:                {stats['total_params']:,}")
        print(f"    Tamano:                {param_bytes / 1024 / 1024:.1f} MB")

        print(f"\n  Estimacion con MNEME (93% adicional):")
        print(f"    Tamano comprimido:     {compressed_bytes / 1024 / 1024:.1f} MB")
        print(f"    Factor total:          {stats['equivalent_params'] * 4 / compressed_bytes:.1f}x")

        equiv_bytes = stats["equivalent_params"] * 4
        print(f"\n  Comparacion vs modelo original sin comprimir:")
        print(f"    Original:              {equiv_bytes / 1024 / 1024:.1f} MB")
        print(f"    MZTrain:               {param_bytes / 1024 / 1024:.1f} MB ({(1 - param_bytes/equiv_bytes)*100:.1f}% ahorro)")
        print(f"    MZTrain + MNEME:       {compressed_bytes / 1024 / 1024:.1f} MB ({(1 - compressed_bytes/equiv_bytes)*100:.1f}% ahorro)")

        results["estimated_compression"] = {
            "original_mb": round(equiv_bytes / 1024 / 1024, 1),
            "mztrain_mb": round(param_bytes / 1024 / 1024, 1),
            "mztrain_mneme_mb": round(compressed_bytes / 1024 / 1024, 1),
            "total_savings_pct": round((1 - compressed_bytes / equiv_bytes) * 100, 1),
        }
    else:
        print(f"\n  Comprimiendo modelo con MNEME...")
        mneme_stats = model.compress_with_mneme("balanced")
        if mneme_stats:
            results["mneme_stats"] = mneme_stats
            print(f"  Compresion completada.")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print(" ZCodeBERT - ANALISIS COMPLETO, ENTRENAMIENTO Y PRUEBAS")
    print(" MSC Star Team (Esraderey y Raul Cruz Acosta)")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f" MNEME: {'Disponible' if HAS_MNEME else 'No instalado'}")
    print("=" * 80)

    all_results = {}

    # Fase 1: Analisis
    arch_results, train_config = phase1_architecture_analysis()
    all_results["architecture"] = arch_results

    # Fase 2: Entrenamiento largo
    train_results, model, tokenizer, config = phase2_long_training(train_config)
    all_results["training"] = train_results

    # Fase 3: Precision
    precision_results = phase3_precision_tests(model, tokenizer, config)
    all_results["precision"] = precision_results

    # Fase 4: Coherencia
    coherence_results = phase4_coherence_test(model, tokenizer, config)
    all_results["coherence"] = coherence_results

    # Fase 5: MNEME
    mneme_results = phase5_mneme_compression(model, config)
    all_results["mneme"] = mneme_results

    # Guardar resultados
    # Convertir valores no serializables
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
            return obj
        return obj

    with open(RESULTS_FILE, "w") as f:
        json.dump(sanitize(all_results), f, indent=2, default=str)

    # Guardar modelo entrenado
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "training_results": sanitize(train_results),
    }, "zcodebert_trained.pt")

    # ========================================================================
    # REPORTE FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print(" REPORTE FINAL - ZCodeBERT")
    print("=" * 80)

    t = train_results
    p = precision_results
    c = coherence_results

    print(f"""
  ARQUITECTURA
  ----------------------------------------------------------------
  Modelo:                    ZCodeBERT (512h / 12L / 8H)
  Params reales:             {all_results['architecture']['memory_analysis']['mztrain_total_mb']:.1f} MB
  Params equivalentes:       {all_results['architecture']['memory_analysis']['traditional_total_mb']:.1f} MB
  Ahorro memoria total:      {all_results['architecture']['memory_analysis']['savings_pct']:.1f}%
  Throughput:                {all_results['architecture']['benchmark']['tokens_per_sec']:,} tokens/s

  ENTRENAMIENTO ({t['total_time_s']/60:.0f} min, {t['total_epochs']} epochs)
  ----------------------------------------------------------------
  Mejor val_loss:            {t['best_val_loss']:.4f}
  Mejor MLM Accuracy:        {max(t['mlm_accuracies']):.2f}%
  Mejor Top-5 Accuracy:      {max(t['top5_accuracies']):.2f}%
  Mejor Perplexity:          {min(t['perplexities']):.1f}
  Rango final:               {t['final_rank']}
  Steps totales:             {t['global_steps']:,}

  PRECISION (datos no vistos)
  ----------------------------------------------------------------
  MLM Accuracy:              {p['fresh_data']['accuracy']:.2f}%
  Top-5 Accuracy:            {p['fresh_data']['top5_accuracy']:.2f}%
  Perplexity:                {p['fresh_data']['perplexity']:.1f}
  Confianza (correctas):     {p['confidence']['correct']:.4f}
  Confianza (incorrectas):   {p['confidence']['wrong']:.4f}
  Separacion:                {p['confidence']['separation']:.4f}

  CLASIFICACION DE LENGUAJE
  ----------------------------------------------------------------
  Accuracy global:           {p['classification']['overall_accuracy']:.1f}%

  COHERENCIA (codigo real)
  ----------------------------------------------------------------
  Accuracy exacta:           {c['coherence_accuracy']:.1f}%
  Top-5 Accuracy:            {c['coherence_top5']:.1f}%

  COMPRESION
  ----------------------------------------------------------------
  MZTrain factorizacion:     {all_results['architecture']['memory_analysis']['savings_pct']:.1f}% ahorro
  Optimizer INT8:            75% ahorro estados
  MNEME post-train:          {'Aplicada' if HAS_MNEME else 'Estimada ~93% adicional'}

  Resultados guardados en:   {RESULTS_FILE}
  Modelo guardado en:        zcodebert_trained.pt
""")

    print("=" * 80)
    print(" ANALISIS COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
