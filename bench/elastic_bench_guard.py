"""
ElasticRank LOSS-GUARD benchmark: ¿el rollback por perdida hace ElasticRank
"seguro por construccion"?

El guard solo es ejercitable donde la compactacion realmente ocurre (regimen
low-rank sintetico; en ZCodeBERT real nada se duerme y el guard ni se
engancha). Aqui se prueban las DOS mitades de la promesa, en una sola tarea
(regresion a un target de rango bajo, capa factorizada sobre-aprovisionada),
comparando guard OFF vs ON:

  A) SAFE: la compactacion es genuinamente inofensiva (direcciones nacidas
     con S~0). El guard NO debe sabotear el ahorro: OFF y ON deben lograr la
     misma reduccion de memoria con R^2 intacto, y ~0 rollbacks.

  B) RISKY: config sobre-agresiva que duerme tambien direcciones UTILES
     (umbral espectral alto + min_per_layer bajo). OFF debe destrozar la
     calidad (pico de loss) a cambio de memoria; ON debe revertir esas
     compactaciones dañinas -> R^2 preservado (memoria casi sin reducir).

Veredicto: el guard convierte una configuracion destructiva en un no-op
seguro, sin perder el ahorro cuando la compactacion es legitima.

Uso: python -m bench.elastic_bench_guard [--epochs 70 --seed 42]
Salida: tabla + bench/results/elastic_bench_guard.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mztrain import ZTrainConfig, RankSchedule, ZTrainEngine  # noqa: E402
from mztrain.layers import ZFactorizedLinear  # noqa: E402

D_IN, D_OUT = 192, 160
TRUE_RANK = 16          # la tarea NECESITA ~16 direcciones
MODEL_RANK = 48         # sobre-aprovisionado
N_TRAIN, N_VAL, BATCH = 4096, 1024, 128


def _W(seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D_OUT, TRUE_RANK, generator=g)
    B = torch.randn(TRUE_RANK, D_IN, generator=g)
    return (A @ B) / (TRUE_RANK ** 0.5)


def make_loaders(seed):
    g = torch.Generator().manual_seed(seed + 1)
    W = _W(seed)
    X = torch.randn(N_TRAIN + N_VAL, D_IN, generator=g)
    Y = X @ W.T + 0.02 * torch.randn(N_TRAIN + N_VAL, D_OUT, generator=g)
    tr = TensorDataset(X[:N_TRAIN], Y[:N_TRAIN])
    va = TensorDataset(X[N_TRAIN:], Y[N_TRAIN:])
    gl = torch.Generator().manual_seed(seed)
    return (DataLoader(tr, batch_size=BATCH, shuffle=True, generator=gl),
            DataLoader(va, batch_size=BATCH))


def build_model(seed):
    # Peso ya ~rank-16 -> SVD-init da 16 S grandes + 32 S~0 (capacidad
    # sobrante explicita); las 16 reales SI importan.
    torch.manual_seed(seed)
    lin = nn.Linear(D_IN, D_OUT)
    with torch.no_grad():
        lin.weight.copy_(_W(seed))
        lin.bias.zero_()
    return nn.Sequential(lin)


def loss_fn(model, batch):
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


def active_fp(model):
    return sum(m.U.numel() + m.S.numel() + m.V.numel()
               for m in model.modules() if isinstance(m, ZFactorizedLinear))


@torch.no_grad()
def var_y(loader):
    return float(torch.cat([y for _, y in loader]).var())


def run(scenario: str, guard: bool, seed: int, epochs: int,
        device) -> Dict[str, Any]:
    train_loader, val_loader = make_loaders(seed)
    model = build_model(seed)
    spe = max(1, N_TRAIN // BATCH)

    if scenario == "safe":
        min_per_layer, spec_thr = 16, 1e-2     # solo duerme las S~0
    else:  # risky: agresivo -> tambien direcciones utiles
        min_per_layer, spec_thr = 2, 0.6

    cfg = ZTrainConfig(
        initial_rank=MODEL_RANK, max_rank=MODEL_RANK,
        rank_schedule=RankSchedule.CONSTANT,
        min_params_to_factorize=4096, compress_optimizer_states=True,
        use_amp=False, learning_rate=1e-2, weight_decay=1e-4,
        log_interval=10_000,
        use_elastic_rank=True,
        elastic_rank_check_interval=max(1, spe // 4),
        elastic_rank_ema_beta=0.5,
        elastic_rank_score_quantile=1.0,
        elastic_rank_sleep_spectral_threshold=spec_thr,
        elastic_rank_sleep_update_threshold=10.0,   # update no bloquea
        elastic_rank_sleep_patience_checks=2,
        elastic_rank_min_age_checks=1,
        elastic_rank_post_growth_grace_checks=0,
        elastic_rank_post_refactor_grace_checks=0,
        elastic_rank_min_per_layer=min_per_layer,
        elastic_rank_compact_min_dirs=2,
        elastic_rank_prune_after_epochs=8,
        elastic_rank_use_redundancy_signal=False,   # aislar via espectral
        elastic_rank_loss_guard_enabled=guard,
        elastic_rank_loss_guard_threshold=1e-3,
        elastic_rank_loss_guard_cooldown_epochs=3,
    )
    engine = ZTrainEngine(model, cfg, device=device)
    init_fp = active_fp(engine.model)
    t0 = time.time()
    summary = engine.train(train_loader, val_loader, loss_fn,
                           epochs=epochs, early_stopping_patience=10_000)
    wall = time.time() - t0

    fin_fp = active_fp(engine.model)
    vy = var_y(val_loader)
    mse = summary["val_losses"][-1]
    es = summary.get("elastic_rank_stats") or {}
    return {
        "scenario": scenario, "guard": guard,
        "wall_s": round(wall, 1),
        "init_fp": init_fp, "final_fp": fin_fp,
        "fp_reduction_pct": round(100.0 * (init_fp - fin_fp) /
                                  max(init_fp, 1), 1),
        "val_r2": round(1.0 - mse / max(vy, 1e-12), 5),
        "val_mse": round(mse, 6),
        "slept": es.get("total_slept"),
        "rollbacks": es.get("total_rollbacks"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device if args.device else
                        ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[guard-bench] device={dev} seed={args.seed} "
          f"epochs={args.epochs}")
    print(f"[guard-bench] target rank-{TRUE_RANK}, modelo rank-{MODEL_RANK} "
          f"fijo; capa factorizada unica\n")

    res = {}
    for sc in ("safe", "risky"):
        for guard in (False, True):
            key = f"{sc}/{'ON' if guard else 'OFF'}"
            res[key] = run(sc, guard, args.seed, args.epochs, dev)
            r = res[key]
            print(f"  {key:<12} done ({r['wall_s']}s) "
                  f"fp-red={r['fp_reduction_pct']}% R2={r['val_r2']} "
                  f"slept={r['slept']} rollbacks={r['rollbacks']}")

    def block(title, off, on, note):
        print(f"\n  == {title} ==")
        print(f"  {'metric':<26}{'guard OFF':<16}{'guard ON':<16}")
        print("  " + "-" * 56)
        print(f"  {'Reduccion params %':<26}"
              f"{off['fp_reduction_pct']:<16}{on['fp_reduction_pct']:<16}")
        print(f"  {'Val R2 (1=perfecto)':<26}"
              f"{off['val_r2']:<16}{on['val_r2']:<16}")
        print(f"  {'Val MSE':<26}{off['val_mse']:<16}{on['val_mse']:<16}")
        print(f"  {'slept / rollbacks':<26}"
              f"{str(off['slept'])+' / '+str(off['rollbacks']):<16}"
              f"{str(on['slept'])+' / '+str(on['rollbacks']):<16}")
        print(f"  -> {note}")

    print("\n" + "=" * 60)
    block("A) SAFE (compactacion legitima)",
          res["safe/OFF"], res["safe/ON"],
          "el guard NO debe sabotear: misma reduccion, R2 intacto, "
          "rollbacks~0")
    block("B) RISKY (config sobre-agresiva)",
          res["risky/OFF"], res["risky/ON"],
          "guard OFF destroza R2 por memoria; guard ON revierte -> "
          "R2 preservado (memoria casi sin tocar)")
    print("=" * 60)

    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(dev),
        "task": {"true_rank": TRUE_RANK, "model_rank": MODEL_RANK},
        "results": res,
    }
    p = ROOT / "bench" / "results" / "elastic_bench_guard.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[guard-bench] saved -> {p}")


if __name__ == "__main__":
    main()
