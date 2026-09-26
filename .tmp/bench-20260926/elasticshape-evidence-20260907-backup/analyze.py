"""Auditoria numerica del benchmark; no entrena ni modifica las observaciones."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
data = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
assert "error" not in data and "summary" in data, "Experimento incompleto"
assert all(run["completed"] for run in data["runs"])
assert len(data["runs"]) == 6
assert hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest() == data["protocol_sha256"]
assert hashlib.sha256((HERE / "benchmark.py").read_bytes()).hexdigest() == data["harness_sha256"]
for name, recorded_hash in data["source_sha256"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == recorded_hash, name

rows = []
for pair in data["summary"]["pairs"]:
    seed = pair["seed"]
    ref = next(run for run in data["runs"] if run["kind"] == "R" and run["seed"] == seed)
    morph = next(run for run in data["runs"] if run["kind"] == "M" and run["seed"] == seed)
    q = ref["final_bpc"]
    assert q == pair["target_bpc"]
    ref_points = [point for point in ref["curve"] if point["bpc"] <= q]
    morph_points = [point for point in morph["curve"] if point["width"] == 192 and point["bpc"] <= q]
    r_hit = ref_points[0]
    m_hit = morph_points[0] if morph_points else None
    assert r_hit == pair["R_hit"]["point"]
    assert (m_hit is None) == (pair["M_hit"] is None)
    row = {
        "seed": seed,
        "target_bpc": q,
        "R_hit_s": r_hit["wall_s"],
        "R_hit_step": r_hit["step"],
        "M_hit_s": m_hit["wall_s"] if m_hit else None,
        "M_hit_step": m_hit["step"] if m_hit else None,
        "M_hit_bpc": m_hit["bpc"] if m_hit else None,
        "params_R": ref["params_end"],
        "params_M_start": morph["params_start"],
        "params_M_end": morph["params_end"],
        "same_layout": ref["layout_end"] == morph["layout_end"],
        "surgery_s": morph["surgery_s"],
        "surgery_loss_delta_nats": morph["surgery_report"]["loss_delta"],
        "surgery_kl": morph["surgery_report"]["kl_div"],
    }
    if m_hit:
        ratio = m_hit["wall_s"] / r_hit["wall_s"]
        assert math.isclose(ratio, pair["ratio"], rel_tol=1e-12)
        previous_r = pair["R_hit"]["previous_observation_wall_s"]
        previous_m = pair["M_hit"]["previous_observation_wall_s"]
        row.update(
            ratio=ratio,
            savings_pct=100 * (1 - ratio),
            local_crossing_window_R_s=[previous_r, r_hit["wall_s"]],
            local_crossing_window_M_s=[previous_m, m_hit["wall_s"]],
            conservative_local_savings_pct=100 * (1 - m_hit["wall_s"] / previous_r) if previous_r > 0 else None,
            train_plus_surgery_ratio=(m_hit["train_s"] + morph["surgery_s"]) / r_hit["train_s"],
        )
    # Secundario descriptivo POSTERIOR; no interviene en la decision preregistrada.
    budget_points = [p for p in morph["curve"] if p["width"] == 192 and p["wall_s"] <= r_hit["wall_s"]]
    if budget_points:
        observation = budget_points[-1]
        row["secondary_at_no_more_than_R_time"] = {
            "M_observation_s": observation["wall_s"],
            "R_budget_s": r_hit["wall_s"],
            "M_bpc": observation["bpc"],
            "R_bpc": q,
            "delta_bpc_M_minus_R": observation["bpc"] - q,
            "label": "Descriptivo posterior; no calidad asintotica ni criterio de decision",
        }
    for name, run in (("R", ref), ("M", morph)):
        curve = run["curve"]
        first, last = curve[-3], curve[-1]
        row[f"{name}_final_improvement_bpc_per_500_steps"] = (
            500 * (first["bpc"] - last["bpc"]) / (last["step"] - first["step"])
        )
    rows.append(row)

ratios = [row["ratio"] for row in rows if "ratio" in row]
geo = math.exp(sum(math.log(r) for r in ratios) / 3) if len(ratios) == 3 else None
assert geo == data["summary"]["geometric_mean_ratio"]
report = {
    "provenance_hashes_match": True,
    "primary_local_savings_supported": data["summary"]["local_savings_supported"],
    "geometric_mean_ratio": geo,
    "geometric_savings_pct": 100 * (1 - geo) if geo is not None else None,
    "mean_R_hit_s": sum(row["R_hit_s"] for row in rows) / 3,
    "mean_M_hit_s": sum(row["M_hit_s"] for row in rows) / 3 if len(ratios) == 3 else None,
    "total_measured_run_seconds": sum(run["wall_s"] for run in data["runs"]),
    "rows": rows,
    "caveats": [
        "Tres semillas, una escala/configuracion y una GPU; no prueba universal ni significacion poblacional.",
        "Cruces observados cada250pasos: las ventanas locales no son intervalos de confianza y no excluyen oscilaciones no observadas.",
        "Objetivo definido por R a4000pasos; no representa convergencia ni calidad maxima alcanzable.",
        "No se afino LR ni inicializacion por brazo; no demuestra superar toda linea base optimizada.",
        "Los valores de calidad con presupuesto no mayor se calcularon despues: son descriptivos.",
    ],
}
print(json.dumps(report, indent=2))
