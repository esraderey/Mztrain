"""Audita seleccion, procedencia y resultados de calidad; solo lectura."""

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=HERE / "results.json")
parser.add_argument("--seeds", default="17,29,43")
args = parser.parse_args()
seeds = [int(value) for value in args.seeds.split(",")]
data = json.loads(args.results.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert sha(HERE / "PROTOCOL.md") == data["protocol_sha256"]
assert sha(HERE / "benchmark.py") == data["harness_sha256"]
assert sha(HERE.parent / "elasticshape-evidence-20260907" / "benchmark.py") == data["base_harness_sha256"]
for name, value in data["source_sha256"].items():
    assert sha(ROOT / name) == value
assert data["checkpoint_selection_locked_before_test"]

rows = []
for seed in seeds:
    pair = {}
    for kind in ("R", "M"):
        run = next(r for r in data["runs"] if r["kind"] == kind and r["seed"] == seed)
        assert run["completed"]
        if "test" not in run:
            assert run["stop_reason"] == "growth_rejected" and not run["surgery_report"]["accepted"]
            pair[kind] = {
                "stop_reason": run["stop_reason"],
                "test": None,
                "final_step": run["final_step"],
                "parameters": run["params_end"],
                "surgery_report": run["surgery_report"],
            }
            continue
        assert sha(HERE / run["best_checkpoint"]) == run["best_checkpoint_sha256"]
        final_points = [p for p in run["curve"] if p["width"] == 64]
        best = min(final_points, key=lambda p: p["val_bpc"])
        assert best["val_bpc"] == run["best_val_bpc"] and best["step"] == run["best_step"]
        assert all(math.isfinite(p["val_bpc"]) for p in run["curve"])
        if run["stop_reason"] in ("validation_plateau", "validation_overfit"):
            assert run["final_step"] >= 12000
            assert run["final_lr"] <= 3e-5 * (1 + 1e-8)
            assert run["final_step"] - run["min_lr_since"] >= 3000
            assert run["final_step"] - run["last_significant_step"] >= 6000
            if run["stop_reason"] == "validation_plateau":
                assert abs(run["final_improvement_bpc_per_500"]) < 0.005
            else:
                assert run["final_val_bpc"] >= run["best_val_bpc"] + 0.02
        elif run["stop_reason"] == "budget_cap":
            assert run["final_step"] == 40000
        metric = run["test"]
        assert math.isclose(metric["character_perplexity"], 2 ** metric["bpc"], rel_tol=1e-12)
        assert 0 <= metric["next_character_accuracy"] <= 1
        pair[kind] = {
            "stop_reason": run["stop_reason"],
            "final_step": run["final_step"],
            "best_step": run["best_step"],
            "best_val_bpc": run["best_val_bpc"],
            "test": metric,
            "parameters": run["params_end"],
            "slope_bpc_per_500": run["final_improvement_bpc_per_500"],
            "lr_changes": run["lr_changes"],
            "wall_s": run["wall_s"],
        }
    r = next(r for r in data["runs"] if r["kind"] == "R" and r["seed"] == seed)
    m = next(r for r in data["runs"] if r["kind"] == "M" and r["seed"] == seed)
    same_layout = r["layout_end"] == m["layout_end"]
    if pair["M"]["test"] is not None:
        assert same_layout and m["surgery_report"]["accepted"]
    delta = pair["M"]["test"]["bpc"] - pair["R"]["test"]["bpc"] if pair["M"]["test"] and pair["R"]["test"] else None
    rows.append({"seed": seed, **pair, "same_layout": same_layout, "delta_test_bpc_M_minus_R": delta})

means = {}
for kind in ("R", "M"):
    if any(r[kind]["test"] is None for r in rows):
        means[kind] = None  # No promediar solo los supervivientes.
        continue
    means[kind] = {
        name: sum(r[kind]["test"][name] for r in rows) / len(rows)
        for name in ("bpc", "character_perplexity", "next_character_accuracy")
    }
print(
    json.dumps(
        {
            "hashes_and_selection_verified": True,
            "n_pairs": len(rows),
            "all_operationally_stopped": all(
                r[k]["stop_reason"] in ("validation_plateau", "validation_overfit") for r in rows for k in ("R", "M")
            ),
            "mean_metrics": means,
            "mean_delta_test_bpc": means["M"]["bpc"] - means["R"]["bpc"] if means["M"] and means["R"] else None,
            "rows": rows,
        },
        indent=2,
    )
)
