"""
ElasticRank benchmark REALISTA (no sintetico, no tautologico).

Diferencia clave con bench/elastic_bench.py: alli el target era un mapeo de
rango 6 CONSTRUIDO a mano -> la redundancia estaba garantizada por diseño.
Aqui se entrena un transformer REAL (ZCodeBERT, 4 capas, attention+MLP
factorizados) en Masked-LM sobre codigo estructurado. La unica "trampa" es
la que comete un practicante real: SOBRE-aprovisionar el rango (rank=48 en
un modelo cuya config small() usa 16). Si hay redundancia explotable es
EMERGENTE del entrenamiento, no plantada. Pregunta honesta: ¿ElasticRank
recupera memoria en un modelo real, sin que se lo pongamos facil, y sin
degradar la calidad (loss/accuracy de MLM)?

3 configs, mismo seed/datos/init:
  baseline = ElasticRank OFF
  v1       = espectral + actividad
  v2       = + redundancia funcional (leverage del Gram)

Memoria reportada de forma HONESTA: parametros entrenables TOTALES del
modelo (no solo la parte factorizada), porque embeddings/decoder/LayerNorm
NO se duermen — ese es el numero real que veria el usuario.

Sin descargas (codigo sintetico estructurado). CPU-viable.
Uso: python -m bench.elastic_bench_real [--epochs 12 --seed 42]
Salida: tabla + bench/results/elastic_bench_real.json
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
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mztrain import ZTrainConfig, RankSchedule, ZTrainEngine  # noqa: E402
from mztrain.layers import ZFactorizedLinear  # noqa: E402
from mztrain.data import (  # noqa: E402
    CodeTokenizer, SyntheticCodeGenerator, CodeDataset, MLMDataset,
)
from mztrain.models import ZCodeBERTConfig, ZCodeBERTForMLM  # noqa: E402

VOCAB = 2000
SEQLEN = 64
HIDDEN = 128
LAYERS = 4
HEADS = 4
INTER = 512
OVERPROVISIONED_RANK = 48     # override con --rank
N_SNIPPETS = 320
BATCH = 16


def make_loaders(seed: int):
    gen = SyntheticCodeGenerator(seed=seed)
    snippets = gen.generate(N_SNIPPETS)
    tok = CodeTokenizer(vocab_size=VOCAB, max_length=SEQLEN)
    base = CodeDataset(snippets, tok, max_length=SEQLEN)
    mlm = MLMDataset(base, mask_prob=0.15, seed=seed)
    n_tr = int(0.8 * len(mlm))
    n_va = len(mlm) - n_tr
    tr, va = random_split(
        mlm, [n_tr, n_va], generator=torch.Generator().manual_seed(seed)
    )
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(tr, batch_size=BATCH, shuffle=True, generator=g),
        DataLoader(va, batch_size=BATCH),
    )


def build_model(seed: int) -> ZCodeBERTForMLM:
    torch.manual_seed(seed)
    cfg = ZCodeBERTConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, intermediate_size=INTER,
        max_position_embeddings=SEQLEN, rank=OVERPROVISIONED_RANK,
        use_activation_checkpointing=False,
    )
    return ZCodeBERTForMLM(cfg)


def loss_fn(model, batch):
    masked_ids, attn_mask, labels = batch
    loss, _ = model(masked_ids, attention_mask=attn_mask, labels=labels)
    return loss


@torch.no_grad()
def _old_coherence(module) -> torch.Tensor:
    """Metrica de redundancia ANTERIOR (Gram con magnitud + ridge global
    escalado por mean(diag)). Se recomputa aqui solo para el diagnostico:
    comparar old vs new sobre el modelo entrenado y mostrar si el ridge
    global cegaba a las direcciones redundantes de baja energia."""
    r = int(module.S.numel())
    if r < 2:
        return torch.zeros(r)
    U = module.U.detach().float()
    S = module.S.detach().float()
    V = module.V.detach().float()
    G = (U.t() @ U) * (V @ V.t()) * (S.unsqueeze(0) * S.unsqueeze(1))
    if not torch.isfinite(G).all():
        return torch.zeros(r)
    diag = torch.diagonal(G)
    lam = 1e-4 * float(diag.mean().clamp_min(1e-12))
    try:
        Ginv = torch.linalg.inv(G + lam * torch.eye(r))
    except Exception:
        return torch.zeros(r)
    lev = (diag * torch.diagonal(Ginv)).clamp_min(1.0)
    return (1.0 - 1.0 / lev).clamp(0.0, 1.0)


@torch.no_grad()
def signal_diagnostic(engine) -> List[Dict[str, Any]]:
    """Para cada capa factorizada: distribucion de los 3 signals + la
    coherencia ANTIGUA, para ver empiricamente si habia redundancia que la
    metrica vieja se perdia por escala."""
    ec = engine.elastic_rank
    if ec is None:
        return []

    def pct(t):
        t = t.float()
        return [round(float(t.min()), 4), round(float(t.median()), 4),
                round(float(t.quantile(0.9)), 4), round(float(t.max()), 4)]

    rows = []
    for name, module in ec._iter_layers(engine.model):
        st = ec._states.get(name)
        if st is None:
            continue
        new_coh = st.redund_ema
        old_coh = _old_coherence(module)
        rows.append({
            "layer": name,
            "rank": int(module.S.numel()),
            "spectral[min/p50/p90/max]": pct(st.spectral_ema),
            "update[min/p50/p90/max]": pct(st.update_ema),
            "coh_new[min/p50/p90/max]": pct(new_coh),
            "coh_old[min/p50/p90/max]": pct(old_coh),
            "n_coh_new>0.9": int((new_coh > 0.9).sum()),
            "n_coh_old>0.9": int((old_coh > 0.9).sum()),
            "n_v1_dead(spec<.03&upd<.2)": int(
                ((st.spectral_ema < 0.03) & (st.update_ema < 0.2)).sum()),
        })
    return rows


def active_factor_params(model) -> int:
    return sum(
        m.U.numel() + m.S.numel() + m.V.numel()
        for m in model.modules() if isinstance(m, ZFactorizedLinear)
    )


def trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def mlm_accuracy(model, loader, device) -> float:
    model.eval()
    ok = tot = 0
    for batch in loader:
        masked_ids, attn, labels = [b.to(device) for b in batch]
        _, logits = model(masked_ids, attention_mask=attn, labels=labels)
        pred = logits.argmax(dim=-1)
        m = labels != -100
        ok += (pred[m] == labels[m]).sum().item()
        tot += int(m.sum().item())
    return ok / max(tot, 1)


def run_probe(seed: int, epochs: int, device) -> List[Dict[str, Any]]:
    """Experimento decisivo: ¿la loss-sensitivity por direccion es
    heterogenea entre capas en un transformer real? Modo diagnostico puro
    (no duerme/compacta nada)."""
    train_loader, val_loader = make_loaders(seed)
    model = build_model(seed)
    out_json = str(ROOT / "bench" / "results" /
                   "loss_sensitivity_zcodebert.json")
    cfg = ZTrainConfig(
        initial_rank=OVERPROVISIONED_RANK,
        max_rank=OVERPROVISIONED_RANK,
        rank_schedule=RankSchedule.CONSTANT,
        min_params_to_factorize=4096,
        compress_optimizer_states=True,
        checkpoint_activations=False,
        use_amp=False,
        learning_rate=5e-4,
        weight_decay=1e-2,
        log_interval=10_000,
        use_elastic_rank=True,
        elastic_rank_min_per_layer=8,
        elastic_rank_probe_loss_sensitivity=True,
        elastic_rank_probe_interval_epochs=max(1, epochs // 4),
        elastic_rank_probe_max_dirs_per_layer=16,
        elastic_rank_probe_output=out_json,
    )
    engine = ZTrainEngine(model, cfg, device=device)
    engine.train(train_loader, val_loader, loss_fn,
                 epochs=epochs, early_stopping_patience=10_000)
    return engine.elastic_rank._probe_history


def run(label: str, elastic: bool, redundancy: bool, seed: int,
        epochs: int, device) -> Dict[str, Any]:
    train_loader, val_loader = make_loaders(seed)
    model = build_model(seed)

    steps_per_epoch = max(1, len(train_loader))
    cfg_kw = dict(
        initial_rank=OVERPROVISIONED_RANK,
        max_rank=OVERPROVISIONED_RANK,
        rank_schedule=RankSchedule.CONSTANT,
        min_params_to_factorize=4096,
        compress_optimizer_states=True,        # realista: encima de la compresion
        checkpoint_activations=False,
        use_amp=False,
        learning_rate=5e-4,
        weight_decay=1e-2,
        log_interval=10_000,
    )
    if elastic:
        cfg_kw.update(
            use_elastic_rank=True,
            elastic_rank_check_interval=max(1, steps_per_epoch // 2),
            elastic_rank_ema_beta=0.5,
            elastic_rank_score_quantile=1.0,
            elastic_rank_sleep_spectral_threshold=3e-2,
            elastic_rank_sleep_update_threshold=2e-1,
            elastic_rank_sleep_patience_checks=2,
            elastic_rank_min_age_checks=2,
            elastic_rank_post_growth_grace_checks=0,
            elastic_rank_post_refactor_grace_checks=0,
            elastic_rank_min_per_layer=8,
            elastic_rank_compact_min_dirs=2,
            elastic_rank_prune_after_epochs=4,
            elastic_rank_use_redundancy_signal=redundancy,
            elastic_rank_redundancy_threshold=0.9,
            elastic_rank_redundancy_patience_checks=2,
        )
    engine = ZTrainEngine(model, ZTrainConfig(**cfg_kw),
                          device=device)

    init_total = trainable_params(engine.model)
    init_fact = active_factor_params(engine.model)
    traj: List[int] = []

    def cb(eng, epoch, metrics):
        traj.append(trainable_params(eng.model))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    summary = engine.train(
        train_loader, val_loader, loss_fn,
        epochs=epochs, early_stopping_patience=10_000, callbacks=[cb],
    )
    wall = time.time() - t0

    diag = signal_diagnostic(engine) if elastic else None
    final_total = trainable_params(engine.model)
    final_fact = active_factor_params(engine.model)
    val_loss = summary["val_losses"][-1]
    val_acc = mlm_accuracy(engine.model, val_loader, device)
    es = summary.get("elastic_rank_stats") or {}
    opt_mb = engine.optimizer.get_memory_stats().get("state_memory_mb", 0.0)

    return {
        "label": label,
        "seed": seed,
        "epochs": epochs,
        "wall_time_s": round(wall, 1),
        "init_trainable_params": init_total,
        "final_trainable_params": final_total,
        "init_factor_params": init_fact,
        "final_factor_params": final_fact,
        "total_param_reduction_pct": round(
            100.0 * (init_total - final_total) / max(init_total, 1), 2),
        "factor_param_reduction_pct": round(
            100.0 * (init_fact - final_fact) / max(init_fact, 1), 2),
        "adam_state_mb_proxy": round(2 * final_total * 4 / (1024 * 1024), 4),
        "opt_state_mb_compressed": round(float(opt_mb), 4),
        "val_mlm_loss": round(float(val_loss), 5),
        "val_mlm_accuracy": round(val_acc, 5),
        "elastic_slept": es.get("total_slept"),
        "elastic_pruned": es.get("total_pruned"),
        "elastic_max_coherence": es.get("max_coherence_ema"),
        "trainable_trajectory": traj,
        "diag": diag,
    }


def main() -> None:
    global OVERPROVISIONED_RANK
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=OVERPROVISIONED_RANK,
                    help="rango factorizado sobre-aprovisionado (max=hidden)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--probe", action="store_true",
                    help="experimento decisivo: loss-sensitivity por "
                         "direccion (modo diagnostico, no muta nada)")
    args = ap.parse_args()
    OVERPROVISIONED_RANK = args.rank
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[real-bench] device={device} seed={args.seed} "
          f"epochs={args.epochs}")
    print(f"[real-bench] ZCodeBERT {LAYERS}L h{HIDDEN} MLM sobre codigo; "
          f"rank SOBRE-aprovisionado={OVERPROVISIONED_RANK} "
          f"(small() usa 16)\n")

    if args.probe:
        print("[real-bench] EXPERIMENTO DECISIVO: loss-sensitivity por "
              "direccion (modo diagnostico, no muta nada)\n")
        hist = run_probe(args.seed, args.epochs, device)
        if not hist:
            print("  (sin registros de probe)")
            return
        r = hist[-1]  # ultimo epoch sondeado (modelo mas entrenado)
        g = r["global"]
        gs = r["global_spearman"]
        print(f"  epoch sondeado={r['epoch']}  loss_base={r['loss_base']}  "
              f"dirs={r['n_directions']}")
        print(f"  rel_delta global: median={g['median']} p10={g['p10']} "
              f"p90={g['p90']} max={g['max']} cv={g['cv']} "
              f"p90/p10={g['p90_p10_ratio']}")
        print(f"  Spearman(rel_delta, proxy): spectral={gs['spectral']} "
              f"update={gs['update']} coherence={gs['coherence']}")
        print(f"\n  VEREDICTO: {r['verdict']['reading']}")
        print(f"  proxies predictivos (|rho|>=0.5): "
              f"{r['verdict']['proxies_predictive']}")
        print("\n  -- por capa (rel_delta p90 | rho spectral) --")
        for L in r["layers"]:
            print(f"  {L['layer']:<38} p90={L['rel_delta']['p90']:<10} "
                  f"ratio={L['rel_delta']['p90_p10_ratio']:<8} "
                  f"rho_spec={L['spearman_vs_spectral']}")
        print(f"\n[real-bench] JSON -> bench/results/"
              f"loss_sensitivity_zcodebert.json")
        return

    base = run("baseline", False, False, args.seed, args.epochs, device)
    print(f"  baseline (OFF)              done ({base['wall_time_s']}s)")
    v1 = run("v1", True, False, args.seed, args.epochs, device)
    print(f"  v1 (espectral+actividad)    done ({v1['wall_time_s']}s)")
    v2 = run("v2", True, True, args.seed, args.epochs, device)
    print(f"  v2 (+ redundancia)          done ({v2['wall_time_s']}s)\n")

    def cell(v):
        return f"{v:,.4f}" if isinstance(v, float) else f"{v:,}"

    def row(name, b, x1, x2):
        print(f"  {name:<32}{cell(b):<13}{cell(x1):<13}{cell(x2):<13}")

    print("=" * 74)
    print(f"  {'Metrica':<32}{'baseline':<13}{'v1':<13}{'v2':<13}")
    print("-" * 74)
    row("Params entrenables TOTALES", base["final_trainable_params"],
        v1["final_trainable_params"], v2["final_trainable_params"])
    row("  reduccion total %", base["total_param_reduction_pct"],
        v1["total_param_reduction_pct"], v2["total_param_reduction_pct"])
    row("Params factorizados", base["final_factor_params"],
        v1["final_factor_params"], v2["final_factor_params"])
    row("  reduccion factor %", base["factor_param_reduction_pct"],
        v1["factor_param_reduction_pct"], v2["factor_param_reduction_pct"])
    row("Adam state MB (proxy 2x)", base["adam_state_mb_proxy"],
        v1["adam_state_mb_proxy"], v2["adam_state_mb_proxy"])
    row("Val MLM loss (lower=better)", base["val_mlm_loss"],
        v1["val_mlm_loss"], v2["val_mlm_loss"])
    row("Val MLM accuracy (higher=bet)", base["val_mlm_accuracy"],
        v1["val_mlm_accuracy"], v2["val_mlm_accuracy"])
    row("Wall time (s)", base["wall_time_s"],
        v1["wall_time_s"], v2["wall_time_s"])
    print("=" * 74)
    for tag, r in (("v1", v1), ("v2", v2)):
        print(f"  {tag}: slept={r['elastic_slept']} "
              f"pruned={r['elastic_pruned']} "
              f"max_coherence={r['elastic_max_coherence']} | "
              f"trainable {r['init_trainable_params']:,} -> "
              f"{r['final_trainable_params']:,}")

    print("\n  -- DIAGNOSTICO v2 (¿habia redundancia que la metrica vieja "
          "se perdia?) --")
    for d in (v2.get("diag") or [])[:6]:
        print(f"  {d['layer']} r{d['rank']}: "
              f"spec={d['spectral[min/p50/p90/max]']} "
              f"upd={d['update[min/p50/p90/max]']}")
        print(f"      coh_new={d['coh_new[min/p50/p90/max]']} "
              f"coh_old={d['coh_old[min/p50/p90/max]']} "
              f"| new>0.9={d['n_coh_new>0.9']} old>0.9={d['n_coh_old>0.9']} "
              f"v1_dead={d['n_v1_dead(spec<.03&upd<.2)']}")

    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "torch_version": torch.__version__,
        "model": {"arch": "ZCodeBERT", "layers": LAYERS, "hidden": HIDDEN,
                  "heads": HEADS, "vocab": VOCAB, "seqlen": SEQLEN,
                  "overprovisioned_rank": OVERPROVISIONED_RANK},
        "baseline": base, "v1": v1, "v2": v2,
    }
    p = ROOT / "bench" / "results" / "elastic_bench_real.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[real-bench] saved -> {p}")


if __name__ == "__main__":
    main()
