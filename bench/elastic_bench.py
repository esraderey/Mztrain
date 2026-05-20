"""
ElasticRank memory-control benchmark.

Tesis a validar: "mejor control de memoria durante entrenamientos largos".

Setup honesto: tarea sintetica cuyo mapeo verdadero es de rango bajo, y un
modelo factorizado SOBRE-aprovisionado (rango fijo alto, sin crecimiento). El
baseline (grow-only / ElasticRank OFF) arrastra todas las direcciones para
siempre; ElasticRank deberia DORMIR las redundantes y mover sus factores +
momentum al sleep bank (CPU, baja precision), reduciendo:
  - parametros factorizados ACTIVOS (memoria de pesos + gradientes),
  - memoria de estados del optimizer (Adam m, v ~ 2x params activos),
manteniendo la accuracy (porque la señal real cabe en pocas direcciones).

Mismo seed / mismos datos / misma init para ambas corridas. Sin descargas
(dataset sintetico), rapido en CPU.

Uso:
    python -m bench.elastic_bench
    python -m bench.elastic_bench --epochs 60 --seed 42 --device cpu
Salida: tabla por consola + bench/results/elastic_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mztrain import ZTrainEngine, ZTrainConfig, RankSchedule  # noqa: E402
from mztrain.layers import ZFactorizedLinear  # noqa: E402


D_IN = 192
D_OUT = 160
TRUE_RANK = 6            # rango intrinseco del target (<< rank del modelo, 48)
N_TRAIN = 4096
N_VAL = 1024
BATCH = 128


def _low_rank_weight(seed: int) -> torch.Tensor:
    """W (D_OUT, D_IN) de rango exacto TRUE_RANK (formato nn.Linear: y=x@W.T)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D_OUT, TRUE_RANK, generator=g)
    B = torch.randn(TRUE_RANK, D_IN, generator=g)
    return (A @ B) / (TRUE_RANK ** 0.5)


def make_data(seed: int):
    """Regresion al MISMO mapeo low-rank que inicializa la capa: Y = X @ W.T,
    rank(W) = 6. Escenario real del codebase (premisa MNEME): se factoriza un
    peso aproximadamente low-rank a un rango SOBRE-aprovisionado (48). ~42
    direcciones nacen con valor singular ~0 -> capacidad desperdiciada que
    ElasticRank debe dormir/podar durante el entrenamiento largo.
    """
    g = torch.Generator().manual_seed(seed + 1)
    W = _low_rank_weight(seed)
    X = torch.randn(N_TRAIN + N_VAL, D_IN, generator=g)
    Y = X @ W.T + 0.02 * torch.randn(N_TRAIN + N_VAL, D_OUT, generator=g)
    tr = TensorDataset(X[:N_TRAIN], Y[:N_TRAIN])
    va = TensorDataset(X[N_TRAIN:], Y[N_TRAIN:])
    gl = torch.Generator().manual_seed(seed)
    return (
        DataLoader(tr, batch_size=BATCH, shuffle=True, generator=gl),
        DataLoader(va, batch_size=BATCH, shuffle=False),
    )


def build_model(seed: int, scenario: str) -> nn.Module:
    """Capa ancha factorizada a rank 48. Dos escenarios:

    - lowrank_init: el peso YA es ~rank-6 -> ~42 direcciones nacen con S~0
      (capacidad sobrante explicita; la señal v1 espectral basta).
    - from_scratch: init aleatoria full-rank; el target es rango 6, asi que
      el W APRENDIDO converge a rango ~6 -> los 48 terminos rango-1 quedan
      linealmente dependientes pero con |S_i|·‖U_i‖·‖V_i‖ comparable: v1 NO
      lo ve (benchmark previo: 0 dormidas), v2 SI (leverage del Gram).
    """
    torch.manual_seed(seed)
    lin = nn.Linear(D_IN, D_OUT)
    if scenario == "lowrank_init":
        with torch.no_grad():
            lin.weight.copy_(_low_rank_weight(seed))
            lin.bias.zero_()
    return nn.Sequential(lin)


def loss_fn(model, batch):
    x, y = batch
    return F.mse_loss(model(x), y)


def active_factor_params(model: nn.Module) -> int:
    """Suma de U+S+V sobre capas factorizadas = proxy de memoria activa
    (pesos; gradientes y estados del optimizer escalan con esto)."""
    tot = 0
    for m in model.modules():
        if isinstance(m, ZFactorizedLinear):
            tot += m.U.numel() + m.S.numel() + m.V.numel()
    return tot


def adam_state_mb(active_params: int) -> float:
    """Estimacion DETERMINISTA de la memoria de estados de Adam: m + v en
    FP32 = 2 * params_activos * 4 bytes. (El get_memory_stats del optimizer
    fluctua porque ZCompressedAdam re-comprime cada 5 steps; este proxy
    permite comparar baseline vs elastic de forma estable.)"""
    return 2 * active_params * 4 / (1024 * 1024)


@torch.no_grad()
def target_variance(loader: DataLoader) -> float:
    """Var(Y) del conjunto de validacion, para normalizar el MSE a R^2."""
    ys = [y for _, y in loader]
    Y = torch.cat(ys, dim=0)
    return float(Y.var())




def run(label: str, elastic: bool, redundancy: bool, seed: int,
        epochs: int, device, scenario: str) -> Dict[str, Any]:
    train_loader, val_loader = make_data(seed)
    model = build_model(seed, scenario).to(device)

    cfg_kw = dict(
        initial_rank=48,
        max_rank=48,                       # rango FIJO: sin crecimiento
        rank_schedule=RankSchedule.CONSTANT,
        min_params_to_factorize=4096,
        compress_optimizer_states=True,
        use_amp=False,
        learning_rate=1e-2,
        weight_decay=1e-4,
        log_interval=10_000,
    )
    if elastic:
        steps_per_epoch = max(1, N_TRAIN // BATCH)
        cfg_kw.update(
            use_elastic_rank=True,
            elastic_rank_check_interval=max(1, steps_per_epoch // 4),
            elastic_rank_ema_beta=0.5,
            elastic_rank_score_quantile=1.0,   # normalizar por max
            elastic_rank_sleep_spectral_threshold=3e-2,
            elastic_rank_sleep_update_threshold=2e-1,
            elastic_rank_sleep_patience_checks=2,
            elastic_rank_min_age_checks=1,
            elastic_rank_post_growth_grace_checks=0,
            elastic_rank_post_refactor_grace_checks=0,
            elastic_rank_min_per_layer=8,
            elastic_rank_compact_min_dirs=2,
            elastic_rank_prune_after_epochs=8,
            # v2: señal de redundancia funcional (la clave del from_scratch)
            elastic_rank_use_redundancy_signal=redundancy,
            elastic_rank_redundancy_threshold=0.9,
            elastic_rank_redundancy_patience_checks=2,
        )
    config = ZTrainConfig(**cfg_kw)
    engine = ZTrainEngine(model, config, device=device)

    init_active = active_factor_params(engine.model)
    per_epoch: List[Dict[str, Any]] = []

    def cb(eng, epoch, metrics):
        per_epoch.append({
            "epoch": epoch,
            "active_params": active_factor_params(eng.model),
            "opt_state_mb": eng.optimizer.get_memory_stats()["state_memory_mb"],
            "train_loss": metrics["train_loss"],
            "val_loss": metrics["val_loss"],
        })

    t0 = time.time()
    summary = engine.train(
        train_loader, val_loader, loss_fn,
        epochs=epochs, early_stopping_patience=10_000,  # curva completa
        callbacks=[cb],
    )
    wall = time.time() - t0

    diag = None
    if elastic and engine.elastic_rank is not None:
        ec = engine.elastic_rank
        tau_s = config.elastic_rank_sleep_spectral_threshold
        tau_u = config.elastic_rank_sleep_update_threshold
        diag = []
        for nm, stt in ec._states.items():
            sp = stt.spectral_ema.float()
            up = stt.update_ema.float()
            diag.append({
                "layer": nm,
                "rank": int(sp.numel()),
                "spectral_min": round(float(sp.min()), 5),
                "spectral_p50": round(float(sp.median()), 5),
                "spectral_max": round(float(sp.max()), 5),
                "update_min": round(float(up.min()), 5),
                "update_p50": round(float(up.median()), 5),
                "update_max": round(float(up.max()), 5),
                "n_spectral_below_tau": int((sp < tau_s).sum()),
                "n_update_below_tau": int((up < tau_u).sum()),
                "n_both_below_tau": int(((sp < tau_s) & (up < tau_u)).sum()),
            })

    actives = [e["active_params"] for e in per_epoch] or [init_active]
    final_active = actives[-1]
    var_y = target_variance(val_loader)
    final_mse = summary["val_losses"][-1]
    val_r2 = 1.0 - final_mse / max(var_y, 1e-12)
    return {
        "label": label,
        "elastic": elastic,
        "seed": seed,
        "epochs": epochs,
        "wall_time_s": round(wall, 2),
        "init_active_params": init_active,
        "final_active_params": final_active,
        "min_active_params": min(actives),
        "active_param_reduction_pct": round(
            100.0 * (init_active - final_active) / max(init_active, 1), 2
        ),
        "adam_state_mb": round(adam_state_mb(final_active), 5),
        "final_val_mse": round(final_mse, 6),
        "val_r2": round(val_r2, 5),
        "final_train_loss": round(summary["train_losses"][-1], 6),
        "elastic_stats": summary.get("elastic_rank_stats"),
        "active_trajectory": actives,
        "diag": diag,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--scenario", choices=["from_scratch", "lowrank_init"],
                    default="from_scratch",
                    help="from_scratch: init aleatoria, target rank-6 (caso v2). "
                         "lowrank_init: peso ya ~rank-6 (caso v1).")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[elastic-bench] device={device} seed={args.seed} "
          f"epochs={args.epochs} scenario={args.scenario}")
    print(f"[elastic-bench] target rank-{TRUE_RANK}, modelo factorizado "
          f"rank-48 fijo\n")

    base = run("baseline", False, False, args.seed, args.epochs,
               device, args.scenario)
    print(f"  baseline (ElasticRank OFF)        done  ({base['wall_time_s']}s)")
    v1 = run("v1", True, False, args.seed, args.epochs, device, args.scenario)
    print(f"  v1 (espectral+actividad)          done  ({v1['wall_time_s']}s)")
    v2 = run("v2", True, True, args.seed, args.epochs, device, args.scenario)
    print(f"  v2 (+ redundancia funcional)      done  ({v2['wall_time_s']}s)\n")

    def cell(v):
        if isinstance(v, float):
            return f"{v:,.4f}"
        return f"{v:,}"

    def row(name, b, x1, x2):
        print(f"  {name:<30}{cell(b):<14}{cell(x1):<14}{cell(x2):<14}")

    print("=" * 74)
    print(f"  {'Metrica':<30}{'baseline':<14}{'v1':<14}{'v2':<14}")
    print("-" * 74)
    row("Params activos (final)", base["final_active_params"],
        v1["final_active_params"], v2["final_active_params"])
    row("Reduccion params %",
        base["active_param_reduction_pct"],
        v1["active_param_reduction_pct"],
        v2["active_param_reduction_pct"])
    row("Adam state MB (det.)", base["adam_state_mb"],
        v1["adam_state_mb"], v2["adam_state_mb"])
    row("Val R2 (1=perfecto)", base["val_r2"], v1["val_r2"], v2["val_r2"])
    row("Val MSE", base["final_val_mse"],
        v1["final_val_mse"], v2["final_val_mse"])
    row("Wall time (s)", base["wall_time_s"],
        v1["wall_time_s"], v2["wall_time_s"])
    print("=" * 74)
    for tag, r in (("v1", v1), ("v2", v2)):
        es = r["elastic_stats"] or {}
        print(f"  {tag}: slept={es.get('total_slept')} "
              f"pruned={es.get('total_pruned')} "
              f"sleeping_now={es.get('sleeping_now')} "
              f"max_coherence_ema={es.get('max_coherence_ema')} "
              f"-> active {r['active_trajectory'][0]:,} -> "
              f"{r['final_active_params']:,}")
    d = (v2.get("diag") or [{}])[0]
    if d:
        v1_slept = ((v1["elastic_stats"] or {}).get("total_slept") or 0)
        if v1_slept == 0:
            verdict = ("v1 ciego (ningun spectral<tau): la redundancia "
                       "funcional es lo unico que dispara v2")
        else:
            verdict = ("v1 ya dormia por la via espectral (S~0 de nacimiento); "
                       "v2 reproduce ese resultado, la redundancia no aporta extra")
        print(f"\n  score-diag capa {d.get('layer')}: "
              f"spectral[min={d.get('spectral_min')} "
              f"p50={d.get('spectral_p50')}] -> {verdict}")

    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "torch_version": torch.__version__,
        "scenario": args.scenario,
        "task": {"d_in": D_IN, "d_out": D_OUT, "true_rank": TRUE_RANK,
                 "model_rank": 48, "schedule": "CONSTANT"},
        "baseline": base,
        "v1": v1,
        "v2": v2,
    }
    p = ROOT / "bench" / "results" / f"elastic_bench_{args.scenario}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[elastic-bench] saved -> {p}")


if __name__ == "__main__":
    main()
