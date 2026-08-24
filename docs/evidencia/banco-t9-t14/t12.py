"""T12 - corrige la linea base de T9-B: referencia CON rampa a LR 3e-4.

Celdas nuevas:
  R_warm_lo = desde cero + rampa al inicio       -> define la linea base fuerte a 3e-4
  M_warm_lo = morph + rampa al inicio Y tras la cirugia (simetrico con la referencia)
Los morphs de T9-B se reutilizan para recalcular su ratio contra la nueva Q'/T_R'.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import (Clock, build, dump, load_data, make_opt, n_params,
                  train_steps, val_bpc_full)
from mztrain.elastic_shape import GrowthEvent, LrWarmup, apply_event

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "t12_results.json")
LR_LO = 3e-4
WARMUP_STEPS, WARMUP_FLOOR, NOISE = 200, 0.1, 1e-3
R_STEPS, PHASE1_STEPS, PHASE2_CAP, EVAL_EVERY, EV_R = 4000, 2000, 4000, 250, 500
DEGEN_THR = 3.20
D_SMALL, D_BIG, LAYERS, HEADS = 192, 384, 6, 6


def _state():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {"experiment": "T12", "protocol": {
        "lr": LR_LO, "wd": 0.01, "batch": 16, "seq": 256, "R_steps": R_STEPS,
        "phase1_steps": PHASE1_STEPS, "phase2_cap": PHASE2_CAP,
        "warmup": [WARMUP_STEPS, WARMUP_FLOOR], "noise_scale": NOISE,
        "degen_thr": DEGEN_THR, "endpoint": "F_192@384",
        "celdas_reutilizadas_de_T9B": ["R/S1 (sin rampa)", "M1/S1 (rampa solo tras cirugia)"],
    }, "runs": {}}


def _save(st):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump(RESULTS, st)


def probe_batch(val, n=8, seq=256):
    return torch.stack([val[i * seq:(i + 1) * seq] for i in range(n)])


def run_R_warm(name, seed, st, train, val, V):
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    m = build(V, D_BIG, LAYERS, HEADS, D_SMALL, "cuda", seed)
    o = make_opt(m, lr=LR_LO)
    warm = LrWarmup(o, WARMUP_STEPS, WARMUP_FLOOR)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    clk, curve = Clock(), []
    for done in range(0, R_STEPS, EV_R):
        train_steps(m, o, train, g, EV_R, clk, warmup=warm, amp=False)
        b = val_bpc_full(m, val)
        curve.append([done + EV_R, round(clk.s, 3), round(b, 6)])
        print(f"  {name} step={done + EV_R} t={clk.s:7.1f} bpc={b:.4f}", flush=True)
    final = curve[-1][2]
    st["runs"][name] = {
        "kind": "R_warm_lo", "seed": seed, "lr": LR_LO, "warmup_inicial": True,
        "params": n_params(m), "final_bpc": final, "train_s": round(clk.s, 3),
        "degenerado": final > DEGEN_THR, "curve": curve,
    }
    print(f"  {name} FIN bpc={final:.4f} t={clk.s:.1f}", flush=True)
    _save(st)
    del m, o
    torch.cuda.empty_cache()


def run_M_warm(name, seed, st, train, val, V, Q, T_R):
    """Morph con rampa TAMBIEN al inicio de la fase densa (simetrico con R_warm_lo)."""
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    m = build(V, D_SMALL, LAYERS, HEADS, None, "cuda", seed)
    o = make_opt(m, lr=LR_LO)
    warm = LrWarmup(o, WARMUP_STEPS, WARMUP_FLOOR)  # rampa inicial
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242 + seed)
    px = probe_batch(val)
    clk, curve, events = Clock(), [], []
    hit_s = hit_step = None
    for done in range(0, PHASE1_STEPS, EVAL_EVERY):
        train_steps(m, o, train, g, EVAL_EVERY, clk, warmup=warm, amp=False)
        b = val_bpc_full(m, val)
        curve.append([done + EVAL_EVERY, round(clk.s, 3), round(b, 6)])
    pre = curve[-1][2]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ev = GrowthEvent(step=0, factorize=True, new_d=D_BIG, noise_scale=NOISE)
    o, rep = apply_event(m, o, ev, probe_x=px, generator=gg)
    torch.cuda.synchronize()
    clk.s += time.perf_counter() - t0
    warm = LrWarmup(o, WARMUP_STEPS, WARMUP_FLOOR)  # rampa post-cirugia (como T9-B)
    post = val_bpc_full(m, val)
    events.append({"to_d": m.d, "params": n_params(m),
                   "drift_rel": rep.get("logits_drift_rel"), "bpc_pre": pre,
                   "bpc_post": post, "bpc_rel_delta": (post - pre) / pre,
                   "train_s": round(clk.s, 3)})
    print(f"  {name} CIRUGIA -> d={m.d} drift={rep.get('logits_drift_rel'):.4f} "
          f"bpc {pre:.4f}->{post:.4f}", flush=True)
    for done in range(0, PHASE2_CAP, EVAL_EVERY):
        train_steps(m, o, train, g, EVAL_EVERY, clk, warmup=warm, amp=False)
        b = val_bpc_full(m, val)
        step = PHASE1_STEPS + done + EVAL_EVERY
        curve.append([step, round(clk.s, 3), round(b, 6)])
        print(f"  {name} step={step} t={clk.s:7.1f} bpc={b:.4f}", flush=True)
        if hit_s is None and b <= Q:
            hit_s, hit_step = round(clk.s, 3), step
            print(f"  {name} >>> alcanza Q'={Q:.4f} en {hit_s}s (paso {step})", flush=True)
    final = curve[-1][2]
    st["runs"][name] = {
        "kind": "M_warm_lo", "seed": seed, "lr": LR_LO, "warmup_inicial": True,
        "params": n_params(m), "final_bpc": final, "train_s": round(clk.s, 3),
        "degenerado": final > DEGEN_THR, "Q": Q, "T_R": T_R, "T_M": hit_s,
        "T_M_step": hit_step, "ratio": (hit_s / T_R) if (hit_s and T_R) else None,
        "events": events, "curve": curve,
    }
    print(f"  {name} FIN bpc={final:.4f} T_M={hit_s}", flush=True)
    _save(st)
    del m, o
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    train, val, V = load_data("cuda")
    st = _state()

    for s in seeds:
        run_R_warm(f"R_warm_lo/s{s}", s, st, train, val, V)

    ok = [st["runs"][f"R_warm_lo/s{s}"] for s in seeds
          if not st["runs"][f"R_warm_lo/s{s}"]["degenerado"]]
    Q = sum(r["final_bpc"] for r in ok) / len(ok)
    T_R = sum(r["train_s"] for r in ok) / len(ok)
    st["linea_base_fuerte_3e4"] = {"Q": Q, "T_R": T_R, "n": len(ok),
                                   "bpcs": [r["final_bpc"] for r in ok]}
    print(f"\n[LINEA BASE FUERTE 3e-4] Q'={Q:.4f} T_R'={T_R:.1f} ({len(ok)}/{len(seeds)} sanas)",
          flush=True)
    print(f"[T9-B publico] Q=2.8801 T_R=208.7 -> la rampa mejora la referencia en "
          f"{2.8801 - Q:+.4f} BPC", flush=True)
    _save(st)

    # recalculo de los morphs de T9-B contra la linea base fuerte
    with open(os.path.join(HERE, "t9_results.json"), encoding="utf-8") as f:
        t9 = json.load(f)["runs"]
    rec = {}
    for s in seeds:
        c = t9[f"M1/S1/s{s}"]["curve"]
        hit = next((t for _, t, b in c if b <= Q), None)
        rec[f"M1/S1/s{s}"] = {"T_M_contra_Q_fuerte": hit,
                              "ratio": (hit / T_R) if hit else None}
        print(f"  T9-B morph s{s}: T_M'={hit} -> ratio {hit / T_R:.3f}" if hit
              else f"  T9-B morph s{s}: NO alcanza Q'", flush=True)
    hits = [v["T_M_contra_Q_fuerte"] for v in rec.values() if v["T_M_contra_Q_fuerte"]]
    if hits:
        r = sum(hits) / len(hits) / T_R
        print(f"\n[T9-B RECALCULADO] ratio = {r:.3f} ({len(hits)}/{len(seeds)} alcanzan Q')"
              f"  [publicado: 0.574]", flush=True)
        st["t9b_recalculado"] = {"detalle": rec, "ratio": r, "n_hit": len(hits)}
    _save(st)

    for s in seeds:
        run_M_warm(f"M_warm_lo/s{s}", s, st, train, val, V, Q, T_R)


if __name__ == "__main__":
    main()
