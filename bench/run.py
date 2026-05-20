"""
Run the MZTrain benchmark and dump JSON results.

Two phases:
  1. Train bench: 3 seeds x 20 epochs MNIST → captures impact of bug #1 (rank schedule).
  2. INT8 sanity: numerical reconstruction-error test on ZGaLoreOptimizer._compress_state
     and ZAdaptiveOptimizer._compress_state → captures bugs #2 and #3 without training.

Usage:
    python -m bench.run --label baseline   # before fixes
    # ... apply fixes ...
    python -m bench.run --label fixed      # after fixes

Output: bench/results/<label>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

# Allow `python -m bench.run` from project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mztrain import (  # noqa: E402
    ZTrainEngine,
    ZTrainConfig,
    RankSchedule,
)
from mztrain.projector import ZGaLoreOptimizer  # noqa: E402
from mztrain.adaptive_optimizer import ZAdaptiveOptimizer  # noqa: E402

from bench.task import (  # noqa: E402
    set_seed,
    build_model,
    get_mnist_loaders,
    evaluate_accuracy,
)


SEEDS = [42, 123, 7]
EPOCHS = 20
BATCH_SIZE = 128

# Config that exercises bug #1: initial_rank << max_rank with EXPONENTIAL schedule.
TRAIN_CONFIG = dict(
    initial_rank=16,
    max_rank=64,
    rank_schedule=RankSchedule.EXPONENTIAL,
    rank_growth_interval=4,         # growth events at epochs 4, 8, 12, 16
    rank_growth_factor=2.0,
    compress_optimizer_states=True,
    use_amp=False,                  # keep deterministic on small model
    learning_rate=1e-3,
    log_interval=200,
    precision_level="standard",     # override via --precision
)


def loss_fn(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)


def train_one_seed(seed: int, device: torch.device) -> Dict[str, Any]:
    """One full training run. Returns metrics dict."""
    set_seed(seed)

    # Build plain nn.Linear MLP — the engine factorizes internally.
    base_model = build_model().to(device)

    config = ZTrainConfig(**TRAIN_CONFIG)
    engine = ZTrainEngine(base_model, config, device=device)

    train_loader, val_loader = get_mnist_loaders(batch_size=BATCH_SIZE, seed=seed)

    t0 = time.time()
    summary = engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        early_stopping_patience=999,  # disable, we want full curve
    )
    wall_time = time.time() - t0

    # Final accuracy (use exported full model so eval matches inference path).
    full_model = engine.export_full_model().to(device)
    val_acc = evaluate_accuracy(full_model, val_loader, device)

    ranks = summary.get("ranks", [])
    return {
        "seed": seed,
        "wall_time_s": wall_time,
        "train_losses": summary.get("train_losses", []),
        "val_losses": summary.get("val_losses", []),
        "ranks": ranks,
        "rank_min": min(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "rank_grew": (max(ranks) > min(ranks)) if ranks else False,
        "final_val_loss": summary.get("val_losses", [None])[-1] if summary.get("val_losses") else None,
        "final_train_loss": summary.get("train_losses", [None])[-1] if summary.get("train_losses") else None,
        "final_val_accuracy": val_acc,
    }


def int8_sanity_check(device: torch.device) -> Dict[str, Any]:
    """Numerical test of the INT8 compress/decompress round-trip.

    A correct INT8 quantizer should give relative error < 1% for smooth tensors.
    The buggy version (scale = abs_max instead of abs_max/127) collapses every
    block to 3 distinct values, producing massive reconstruction error.
    """
    torch.manual_seed(0)
    # Realistic Adam-state-shaped tensor: small values, varied magnitudes.
    tensor = torch.randn(4096, device=device) * 0.01

    # Build minimal optimizer instances purely to access their compress methods.
    dummy = torch.nn.Parameter(torch.zeros(1, device=device))
    galore = ZGaLoreOptimizer([dummy], lr=1e-3)
    adaptive = ZAdaptiveOptimizer([dummy], lr=1e-3)

    results: Dict[str, Any] = {"input_unique_values": int(tensor.unique().numel())}

    for name, opt in [("galore", galore), ("adaptive", adaptive)]:
        compressed = opt._compress_state(tensor)
        recovered = opt._decompress_state(compressed, tensor.dtype)
        diff = (tensor - recovered)
        rel_err = (diff.norm() / tensor.norm().clamp(min=1e-12)).item()
        results[name] = {
            "rel_reconstruction_error": rel_err,
            "recovered_unique_values": int(recovered.unique().numel()),
            "max_abs_error": float(diff.abs().max().item()),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Output filename label (baseline | fixed | ...)")
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 (default: auto)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Override seed list")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--precision",
        choices=["standard", "aggressive", "fp8"],
        default="standard",
        help="Precision level (standard=BF16, fp8=FP8 GEMM via _scaled_mm)",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = args.seeds if args.seeds else SEEDS
    if args.epochs:
        TRAIN_CONFIG["log_interval"] = 200  # avoid log spam if epochs override
        global EPOCHS
        EPOCHS = args.epochs
    TRAIN_CONFIG["precision_level"] = args.precision

    print(f"[bench] label={args.label} device={device} seeds={seeds} epochs={EPOCHS}")

    out: Dict[str, Any] = {
        "label": args.label,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "torch_version": torch.__version__,
        "config": {**TRAIN_CONFIG, "rank_schedule": TRAIN_CONFIG["rank_schedule"].value},
        "epochs": EPOCHS,
        "seeds": seeds,
        "runs": [],
        "int8_sanity": None,
    }

    # ---- Phase 1: training runs ----
    for seed in seeds:
        print(f"\n[bench] === seed={seed} ===")
        run_result = train_one_seed(seed, device)
        out["runs"].append(run_result)
        print(
            f"[bench] seed={seed} done: "
            f"final_val_acc={run_result['final_val_accuracy']:.4f} "
            f"final_val_loss={run_result['final_val_loss']:.4f} "
            f"rank_min={run_result['rank_min']} rank_max={run_result['rank_max']} "
            f"rank_grew={run_result['rank_grew']} "
            f"wall_time={run_result['wall_time_s']:.1f}s"
        )

    # ---- Phase 2: INT8 sanity check ----
    print("\n[bench] === INT8 sanity ===")
    out["int8_sanity"] = int8_sanity_check(device)
    for opt_name in ("galore", "adaptive"):
        r = out["int8_sanity"][opt_name]
        print(
            f"[bench] {opt_name}: rel_err={r['rel_reconstruction_error']:.4f} "
            f"unique_values={r['recovered_unique_values']} "
            f"max_abs_err={r['max_abs_error']:.4f}"
        )

    # ---- Save ----
    out_path = ROOT / "bench" / "results" / f"{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[bench] saved -> {out_path}")


if __name__ == "__main__":
    main()
