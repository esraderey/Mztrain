"""T10 - el claim bajo LR ajustado (1.2e-3) y la hipotesis de estabilidad.

Mismo banco y protocolo que T9; la unica variable manipulada es el LR.
Preregistro: PREREGISTRO-T10-lr-estabilidad.md. Volcado atomico reanudable.
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
RESULTS = os.path.join(HERE, "t10_results.json")
LR_HI = 1.2e-3
WARMUP_STEPS, WARMUP_FLOOR, NOISE = 200, 0.1, 1e-3
R_STEPS, PHASE1_STEPS, PHASE2_CAP, EVAL_EVERY, EV_R = 4000, 2000, 4000, 250, 500
DEGEN_THR = 3.20
D_SMALL, D_BIG, LAYERS, HEADS = 192, 384, 6, 6


def _state():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {
        "experiment": "T10",
        "protocol": {
            "data": "char-WikiText-2 (vocab train+val = 1118)", "batch": 16,
            "seq": 256, "lr": LR_HI, "wd": 0.01, "R_steps": R_STEPS,
            "phase1_steps": PHASE1_STEPS, "phase2_cap": PHASE2_CAP,
            "eval_every": EVAL_EVERY, "warmup": [WARMUP_STEPS, WARMUP_FLOOR],
            "noise_scale": NOISE, "degen_thr": DEGEN_THR,
            "endpoint": "F_192@384 (L=6, h=6)",
        },
        "runs": {},
    }


def _save(st):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump(RESULTS, st)


def probe_batch(val, n=8, seq=256):
    return torch.stack([val[i * seq:(i + 1) * seq] for i in range(n)])


def run_R(name, seed, lr, rank, st, train, val, V):
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    m = build(V, D_BIG, LAYERS, HEADS, rank, "cuda", seed)
    o = make_opt(m, lr=lr)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    clk, curve, nan = Clock(), [], False
    torch.cuda.reset_peak_memory_stats()
    try:
        for done in range(0, R_STEPS, EV_R):
            train_steps(m, o, train, g, EV_R, clk, amp=False)
            b = val_bpc_full(m, val)
            curve.append([done + EV_R, round(clk.s, 3), round(b, 6)])
            print(f"  {name} step={done + EV_R} t={clk.s:7.1f} bpc={b:.4f}", flush=True)
    except RuntimeError as e:
        nan = True
        print(f"  {name} ABORTADO: {e}", flush=True)
    final = curve[-1][2] if curve else float("inf")
    st["runs"][name] = {
        "kind": "R", "seed": seed, "lr": lr, "rank": rank, "params": n_params(m),
        "final_bpc": final, "train_s": round(clk.s, 3), "nan": nan,
        "degenerado": bool(nan or final > DEGEN_THR), "curve": curve,
    }
    print(f"  {name} FIN bpc={final:.4f} degenerado={st['runs'][name]['degenerado']}", flush=True)
    _save(st)
    del m, o
    torch.cuda.empty_cache()


def run_M(name, seed, lr, st, train, val, V, Q=None, T_R=None):
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    m = build(V, D_SMALL, LAYERS, HEADS, None, "cuda", seed)
    o = make_opt(m, lr=lr)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242 + seed)
    px = probe_batch(val)
    clk, curve, events, nan = Clock(), [], [], False
    torch.cuda.reset_peak_memory_stats()
    hit_s = hit_step = bpc_at_TR = None
    try:
        for done in range(0, PHASE1_STEPS, EVAL_EVERY):
            train_steps(m, o, train, g, EVAL_EVERY, clk, amp=False)
            b = val_bpc_full(m, val)
            curve.append([done + EVAL_EVERY, round(clk.s, 3), round(b, 6)])
        pre = curve[-1][2]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ev = GrowthEvent(step=0, factorize=True, new_d=D_BIG, noise_scale=NOISE)
        o, rep = apply_event(m, o, ev, probe_x=px, generator=gg)
        torch.cuda.synchronize()
        surg = time.perf_counter() - t0
        clk.s += surg
        warm = LrWarmup(o, WARMUP_STEPS, WARMUP_FLOOR)
        post = val_bpc_full(m, val)
        events.append({
            "to_d": m.d, "params": n_params(m), "drift_rel": rep.get("logits_drift_rel"),
            "bpc_pre": pre, "bpc_post": post, "bpc_rel_delta": (post - pre) / pre,
            "surgery_s": round(surg, 3), "train_s": round(clk.s, 3),
        })
        print(f"  {name} CIRUGIA -> d={m.d} drift={rep.get('logits_drift_rel'):.4f} "
              f"bpc {pre:.4f}->{post:.4f}", flush=True)
        for done in range(0, PHASE2_CAP, EVAL_EVERY):
            train_steps(m, o, train, g, EVAL_EVERY, clk, warmup=warm, amp=False)
            b = val_bpc_full(m, val)
            step = PHASE1_STEPS + done + EVAL_EVERY
            curve.append([step, round(clk.s, 3), round(b, 6)])
            print(f"  {name} step={step} t={clk.s:7.1f} bpc={b:.4f}", flush=True)
            if Q is not None and hit_s is None and b <= Q:
                hit_s, hit_step = round(clk.s, 3), step
                print(f"  {name} >>> alcanza Q={Q:.4f} en {hit_s}s (paso {step})", flush=True)
            if T_R is not None and bpc_at_TR is None and clk.s >= T_R:
                bpc_at_TR = b
    except RuntimeError as e:
        nan = True
        print(f"  {name} ABORTADO: {e}", flush=True)
    final = curve[-1][2] if curve else float("inf")
    st["runs"][name] = {
        "kind": "M", "seed": seed, "lr": lr, "params": n_params(m), "final_bpc": final,
        "train_s": round(clk.s, 3), "nan": nan,
        "degenerado": bool(nan or final > DEGEN_THR),
        "Q": Q, "T_R": T_R, "T_M": hit_s, "T_M_step": hit_step,
        "ratio": (hit_s / T_R) if (hit_s and T_R) else None,
        "bpc_at_T_R": bpc_at_TR, "events": events, "curve": curve,
    }
    print(f"  {name} FIN bpc={final:.4f} T_M={hit_s} degenerado="
          f"{st['runs'][name]['degenerado']}", flush=True)
    _save(st)
    del m, o
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--control-seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    csesds = [int(x) for x in a.control_seeds.split(",")]
    train, val, V = load_data("cuda")
    st = _state()

    # control diagnostico: replica la inestabilidad de T6 a rango bajo?
    for s in csesds:
        run_R(f"CTRL_F64/s{s}", s, LR_HI, 64, st, train, val, V)

    for s in seeds:
        run_R(f"R_hi/s{s}", s, LR_HI, D_SMALL, st, train, val, V)

    ok = [st["runs"][f"R_hi/s{s}"] for s in seeds
          if f"R_hi/s{s}" in st["runs"] and not st["runs"][f"R_hi/s{s}"]["degenerado"]]
    Q = sum(r["final_bpc"] for r in ok) / len(ok)
    T_R = sum(r["train_s"] for r in ok) / len(ok)
    degen = [s for s in seeds if st["runs"][f"R_hi/s{s}"]["degenerado"]]
    print(f"[R_hi] Q={Q:.4f} T_R={T_R:.1f} sanos={len(ok)}/{len(seeds)} degenerados={degen}",
          flush=True)
    st["ref_R_hi"] = {"Q": Q, "T_R": T_R, "n_ok": len(ok), "degenerados": degen}
    _save(st)

    for s in seeds:
        run_M(f"M_hi/s{s}", s, LR_HI, st, train, val, V, Q, T_R)


if __name__ == "__main__":
    main()
