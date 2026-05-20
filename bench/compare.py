"""
Compare two benchmark JSONs (baseline vs fixed) and print delta table.

Usage:
    python -m bench.compare baseline fixed
    # reads bench/results/baseline.json and bench/results/fixed.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (xs[0], 0.0)
    return (statistics.mean(xs), statistics.stdev(xs))


def _fmt(mean: float, std: float, pct: bool = False) -> str:
    if mean != mean:  # NaN
        return "—"
    suffix = "%" if pct else ""
    if pct:
        return f"{mean*100:.2f}{suffix} ± {std*100:.2f}"
    return f"{mean:.4f} ± {std:.4f}"


def _delta(base_mean: float, fixed_mean: float) -> str:
    if base_mean == 0 or base_mean != base_mean or fixed_mean != fixed_mean:
        return "—"
    abs_d = fixed_mean - base_mean
    rel_d = abs_d / abs(base_mean) * 100
    sign = "+" if abs_d >= 0 else ""
    return f"{sign}{abs_d:.4f} ({sign}{rel_d:.1f}%)"


def aggregate(data: Dict[str, Any]) -> Dict[str, Any]:
    runs = data["runs"]
    return {
        "final_val_acc":  _mean_std([r["final_val_accuracy"] for r in runs]),
        "final_val_loss": _mean_std([r["final_val_loss"] for r in runs]),
        "final_train_loss": _mean_std([r["final_train_loss"] for r in runs]),
        "rank_max":       _mean_std([float(r["rank_max"]) for r in runs]),
        "rank_grew_frac": _mean_std([1.0 if r["rank_grew"] else 0.0 for r in runs]),
        "wall_time_s":    _mean_std([r["wall_time_s"] for r in runs]),
        "int8":           data.get("int8_sanity"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="Baseline label (reads bench/results/<label>.json)")
    parser.add_argument("fixed", help="Fixed label")
    args = parser.parse_args()

    base = json.loads((RESULTS / f"{args.baseline}.json").read_text())
    fixed = json.loads((RESULTS / f"{args.fixed}.json").read_text())

    a = aggregate(base)
    b = aggregate(fixed)

    print(f"\nComparison: {args.baseline}  ->  {args.fixed}")
    print(f"  Seeds: {base['seeds']} | Epochs: {base['epochs']} | Device: {base['device']}\n")

    rows = [
        ("Final val accuracy",  a["final_val_acc"],   b["final_val_acc"],   True),
        ("Final val loss",      a["final_val_loss"],  b["final_val_loss"],  False),
        ("Final train loss",    a["final_train_loss"], b["final_train_loss"], False),
        ("Max rank reached",    a["rank_max"],        b["rank_max"],        False),
        ("Rank grew (fraction of seeds)", a["rank_grew_frac"], b["rank_grew_frac"], True),
        ("Wall time (s)",       a["wall_time_s"],     b["wall_time_s"],     False),
    ]

    col_w = (32, 22, 22, 24)
    header = f"{'Metric'.ljust(col_w[0])}{'Baseline'.ljust(col_w[1])}{'Fixed'.ljust(col_w[2])}{'Delta'.ljust(col_w[3])}"
    print(header)
    print("-" * sum(col_w))
    for name, a_val, b_val, is_pct in rows:
        am, asd = a_val
        bm, bsd = b_val
        print(
            f"{name.ljust(col_w[0])}"
            f"{_fmt(am, asd, is_pct).ljust(col_w[1])}"
            f"{_fmt(bm, bsd, is_pct).ljust(col_w[2])}"
            f"{_delta(am, bm).ljust(col_w[3])}"
        )

    # INT8 sanity
    print("\n--- INT8 reconstruction (single deterministic input) ---")
    if a["int8"] and b["int8"]:
        key = "rel_reconstruction_error"
        print(
            f"{'Optimizer'.ljust(15)}"
            f"{'Baseline rel_err'.ljust(22)}"
            f"{'Fixed rel_err'.ljust(22)}"
            f"{'Baseline #unique'.ljust(20)}"
            f"{'Fixed #unique'}"
        )
        print("-" * 90)
        for opt in ("galore", "adaptive"):
            ar = a["int8"][opt]
            br = b["int8"][opt]
            print(
                f"{opt.ljust(15)}"
                f"{f'{ar[key]:.4f}'.ljust(22)}"
                f"{f'{br[key]:.4f}'.ljust(22)}"
                f"{str(ar['recovered_unique_values']).ljust(20)}"
                f"{str(br['recovered_unique_values'])}"
            )
    else:
        print("(int8_sanity missing in one of the inputs)")
    print()


if __name__ == "__main__":
    main()
