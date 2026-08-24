"""T9 - las tres banderas que T8 dejo abiertas: ESCALA, growths ENCADENADOS, bf16.

Protocolo del arco (T4/T6/T8): char-WT2, batch 16, seq 256, AdamW lr 3e-4 wd 0.01
sin schedule, fp32 salvo el brazo C (autocast bf16, params fp32, EVAL SIEMPRE fp32).
Reloj de TRAIN puro (la cirugia cuenta). Dump atomico incremental, reanudable.
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

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t9_results.json")
WARMUP_STEPS, WARMUP_FLOOR, NOISE = 200, 0.1, 1e-3
R_STEPS, PHASE1_STEPS, PHASE2_CAP, EVAL_EVERY = 4000, 2000, 4000, 250
STALL_THR = 3.20
EV_R = 500  # cadencia de eval de R (descriptiva: no toca ni Q ni el reloj de train)

SCALES = {
    "S1": dict(d_small=192, d_big=384, layers=6, heads=6, mid=288),
    "S2": dict(d_small=384, d_big=768, layers=8, heads=8, mid=576),
}


def _state():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {
        "experiment": "T9",
        "protocol": {
            "data": "char-WikiText-2 (vocab train+val = 1118)",
            "batch": 16, "seq": 256, "lr": 3e-4, "wd": 0.01,
            "R_steps": R_STEPS, "phase1_steps": PHASE1_STEPS,
            "phase2_cap": PHASE2_CAP, "eval_every": EVAL_EVERY,
            "warmup": [WARMUP_STEPS, WARMUP_FLOOR], "noise_scale": NOISE,
            "stall_thr": STALL_THR,
        },
        "runs": {},
    }


def _save(st):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump(RESULTS, st)


def probe_batch(val, n=8, seq=256):
    return torch.stack([val[i * seq:(i + 1) * seq] for i in range(n)])


def run_R(name, scale, seed, amp, train, val, V, st, steps=R_STEPS):
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    c = SCALES[scale]
    m = build(V, c["d_big"], c["layers"], c["heads"], c["d_small"], "cuda", seed)
    o = make_opt(m)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    clk, curve = Clock(), []
    torch.cuda.reset_peak_memory_stats()
    for done in range(0, steps, EV_R):
        train_steps(m, o, train, g, EV_R, clk, amp=amp)
        b = val_bpc_full(m, val)
        curve.append([done + EV_R, round(clk.s, 3), round(b, 6)])
        print(f"  {name} step={done + EV_R} t={clk.s:7.1f} bpc={b:.4f}", flush=True)
    st["runs"][name] = {
        "kind": "R", "scale": scale, "seed": seed, "amp": amp,
        "params": n_params(m), "final_bpc": curve[-1][2],
        "train_s": round(clk.s, 3), "curve": curve,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
    }
    _save(st)
    del m, o
    torch.cuda.empty_cache()


def run_M(name, scale, seed, amp, chained, train, val, V, st, Q=None, T_R=None,
          phase1=PHASE1_STEPS, cap=PHASE2_CAP):
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    c = SCALES[scale]
    m = build(V, c["d_small"], c["layers"], c["heads"], None, "cuda", seed)
    o = make_opt(m)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242 + seed)
    px = probe_batch(val)
    clk, curve, events = Clock(), [], []
    torch.cuda.reset_peak_memory_stats()

    if chained:
        stages = [(phase1 // 2, dict(factorize=True, new_d=c["mid"])),
                  (phase1 // 2, dict(new_d=c["d_big"]))]
    else:
        stages = [(phase1, dict(factorize=True, new_d=c["d_big"]))]

    warm, base = None, 0
    for n_steps, ev_kw in stages:
        for done in range(0, n_steps, EVAL_EVERY):
            train_steps(m, o, train, g, EVAL_EVERY, clk, warmup=warm, amp=amp)
            b = val_bpc_full(m, val)
            curve.append([base + done + EVAL_EVERY, round(clk.s, 3), round(b, 6)])
        base += n_steps
        pre = curve[-1][2]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ev = GrowthEvent(step=0, noise_scale=NOISE, **ev_kw)
        o, rep = apply_event(m, o, ev, probe_x=px, generator=gg)
        torch.cuda.synchronize()
        surg_s = time.perf_counter() - t0
        clk.s += surg_s  # la cirugia cuenta como reloj de train
        warm = LrWarmup(o, WARMUP_STEPS, WARMUP_FLOOR)
        post = val_bpc_full(m, val)
        events.append({
            "to_d": m.d, "params": n_params(m),
            "drift_rel": rep.get("logits_drift_rel"),
            "bpc_pre": pre, "bpc_post": post,
            "bpc_rel_delta": (post - pre) / pre,
            "surgery_s": round(surg_s, 3), "opt": rep["optimizer"],
            "train_s": round(clk.s, 3),
        })
        print(f"  {name} CIRUGIA -> d={m.d} drift={rep.get('logits_drift_rel'):.4f} "
              f"bpc {pre:.4f}->{post:.4f} ({surg_s:.2f}s)", flush=True)

    hit_s = hit_step = bpc_at_TR = None
    for done in range(0, cap, EVAL_EVERY):
        train_steps(m, o, train, g, EVAL_EVERY, clk, warmup=warm, amp=amp)
        b = val_bpc_full(m, val)
        step = base + done + EVAL_EVERY
        curve.append([step, round(clk.s, 3), round(b, 6)])
        print(f"  {name} step={step} t={clk.s:7.1f} bpc={b:.4f}", flush=True)
        if Q is not None and hit_s is None and b <= Q:
            hit_s, hit_step = round(clk.s, 3), step
            print(f"  {name} >>> alcanza Q={Q:.4f} en {hit_s}s (paso {step})", flush=True)
        if T_R is not None and bpc_at_TR is None and clk.s >= T_R:
            bpc_at_TR = b
    st["runs"][name] = {
        "kind": "M", "scale": scale, "seed": seed, "amp": amp, "chained": chained,
        "params": n_params(m), "final_bpc": curve[-1][2], "train_s": round(clk.s, 3),
        "Q": Q, "T_R": T_R, "T_M": hit_s, "T_M_step": hit_step,
        "ratio": (hit_s / T_R) if (hit_s and T_R) else None,
        "bpc_at_T_R": bpc_at_TR, "events": events, "curve": curve,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
    }
    _save(st)
    del m, o
    torch.cuda.empty_cache()


def q_and_TR(st, tag, seeds):
    """Q y T_R de la referencia EXCLUYENDO seeds estancados (regla preregistrada:
    un R en la meseta infla Q y favoreceria al morph; nunca se excluye un M)."""
    ok = [st["runs"][f"R/{tag}/s{s}"] for s in seeds
          if f"R/{tag}/s{s}" in st["runs"]
          and st["runs"][f"R/{tag}/s{s}"]["final_bpc"] <= STALL_THR]
    if not ok:
        return None, None, []
    Q = sum(r["final_bpc"] for r in ok) / len(ok)
    T_R = sum(r["train_s"] for r in ok) / len(ok)
    return Q, T_R, [r["seed"] for r in ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    train, val, V = load_data("cuda")
    st = _state()

    if a.arm == "B":       # S1 fp32: R + morph simple (replica T8) + encadenado
        tag = "S1"
        for s in seeds:
            run_R(f"R/{tag}/s{s}", "S1", s, False, train, val, V, st)
        Q, T_R, used = q_and_TR(st, tag, seeds)
        print(f"[{tag}] Q={Q:.4f} T_R={T_R:.1f} seeds_validos={used}", flush=True)
        st["ref_" + tag] = {"Q": Q, "T_R": T_R, "seeds": used}
        _save(st)
        for s in used:
            run_M(f"M1/{tag}/s{s}", "S1", s, False, False, train, val, V, st, Q, T_R)
            run_M(f"M2/{tag}/s{s}", "S1", s, False, True, train, val, V, st, Q, T_R)

    elif a.arm == "C":     # S1 bf16: R + morph simple
        tag = "S1_bf16"
        for s in seeds:
            run_R(f"R/{tag}/s{s}", "S1", s, True, train, val, V, st)
        Q, T_R, used = q_and_TR(st, tag, seeds)
        print(f"[{tag}] Q={Q:.4f} T_R={T_R:.1f} seeds_validos={used}", flush=True)
        st["ref_" + tag] = {"Q": Q, "T_R": T_R, "seeds": used}
        _save(st)
        for s in used:
            run_M(f"M1/{tag}/s{s}", "S1", s, True, False, train, val, V, st, Q, T_R)

    else:                  # A: escala S2 fp32
        tag = "S2"
        for s in seeds:
            run_R(f"R/{tag}/s{s}", "S2", s, False, train, val, V, st)
        Q, T_R, used = q_and_TR(st, tag, seeds)
        print(f"[{tag}] Q={Q:.4f} T_R={T_R:.1f} seeds_validos={used}", flush=True)
        st["ref_" + tag] = {"Q": Q, "T_R": T_R, "seeds": used}
        _save(st)
        for s in used:
            run_M(f"M1/{tag}/s{s}", "S2", s, False, False, train, val, V, st, Q, T_R)


if __name__ == "__main__":
    main()
