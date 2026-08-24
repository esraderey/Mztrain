"""T11 - aislar el mecanismo de la proteccion: fase densa o rampa de LR?

Completa el 2x2 de T10 con las dos celdas que faltan:
  R_warm   = referencia desde cero CON rampa de LR al inicio
  M_nowarm = morph SIN rampa tras la cirugia
Las otras dos celdas se reutilizan de t10_results.json (declarado en el preregistro).
Mismo banco, mismas semillas, mismo protocolo, LR 1.2e-3.
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
RESULTS = os.path.join(HERE, "t11_results.json")
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
        "experiment": "T11",
        "protocol": {
            "data": "char-WikiText-2 (vocab train+val = 1118)", "batch": 16, "seq": 256,
            "lr": LR_HI, "wd": 0.01, "R_steps": R_STEPS, "phase1_steps": PHASE1_STEPS,
            "phase2_cap": PHASE2_CAP, "warmup": [WARMUP_STEPS, WARMUP_FLOOR],
            "noise_scale": NOISE, "degen_thr": DEGEN_THR,
            "celdas_reutilizadas_de_T10": ["R_hi (sin rampa)", "M_hi (con rampa)"],
        },
        "runs": {},
    }


def _save(st):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump(RESULTS, st)


def probe_batch(val, n=8, seq=256):
    return torch.stack([val[i * seq:(i + 1) * seq] for i in range(n)])


def modo_fallo(curve, thr=DEGEN_THR):
    """Distingue los dos modos que identifico T10: meseta nunca rota vs colapso
    posterior a una ruptura. Descriptivo, no entra en ninguna regla."""
    bpcs = [b for _, _, b in curve]
    if bpcs[-1] <= thr:
        return "sano"
    return "colapso_post_ruptura" if min(bpcs) <= thr else "meseta_no_rota"


def run_R_warm(name, seed, st, train, val, V):
    """Referencia desde cero CON rampa de LR al inicio."""
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    m = build(V, D_BIG, LAYERS, HEADS, D_SMALL, "cuda", seed)
    o = make_opt(m, lr=LR_HI)
    warm = LrWarmup(o, WARMUP_STEPS, WARMUP_FLOOR)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    clk, curve, nan = Clock(), [], False
    try:
        for done in range(0, R_STEPS, EV_R):
            train_steps(m, o, train, g, EV_R, clk, warmup=warm, amp=False)
            b = val_bpc_full(m, val)
            curve.append([done + EV_R, round(clk.s, 3), round(b, 6)])
            print(f"  {name} step={done + EV_R} t={clk.s:7.1f} bpc={b:.4f}", flush=True)
    except RuntimeError as e:
        nan = True
        print(f"  {name} ABORTADO: {e}", flush=True)
    final = curve[-1][2] if curve else float("inf")
    st["runs"][name] = {
        "kind": "R_warm", "seed": seed, "lr": LR_HI, "warmup": True,
        "params": n_params(m), "final_bpc": final, "train_s": round(clk.s, 3),
        "nan": nan, "degenerado": bool(nan or final > DEGEN_THR),
        "modo": modo_fallo(curve) if curve else "abortado", "curve": curve,
    }
    r = st["runs"][name]
    print(f"  {name} FIN bpc={final:.4f} degenerado={r['degenerado']} modo={r['modo']}",
          flush=True)
    _save(st)
    del m, o
    torch.cuda.empty_cache()


def run_M_nowarm(name, seed, st, train, val, V, Q, T_R):
    """Morph SIN rampa tras la cirugia."""
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    m = build(V, D_SMALL, LAYERS, HEADS, None, "cuda", seed)
    o = make_opt(m, lr=LR_HI)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242 + seed)
    px = probe_batch(val)
    clk, curve, events, nan = Clock(), [], [], False
    hit_s = hit_step = None
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
        clk.s += time.perf_counter() - t0
        # SIN LrWarmup: esa es la variable manipulada
        post = val_bpc_full(m, val)
        events.append({
            "to_d": m.d, "params": n_params(m), "drift_rel": rep.get("logits_drift_rel"),
            "bpc_pre": pre, "bpc_post": post, "bpc_rel_delta": (post - pre) / pre,
            "train_s": round(clk.s, 3),
        })
        print(f"  {name} CIRUGIA (sin rampa) -> d={m.d} "
              f"drift={rep.get('logits_drift_rel'):.4f} bpc {pre:.4f}->{post:.4f}", flush=True)
        for done in range(0, PHASE2_CAP, EVAL_EVERY):
            train_steps(m, o, train, g, EVAL_EVERY, clk, amp=False)
            b = val_bpc_full(m, val)
            step = PHASE1_STEPS + done + EVAL_EVERY
            curve.append([step, round(clk.s, 3), round(b, 6)])
            print(f"  {name} step={step} t={clk.s:7.1f} bpc={b:.4f}", flush=True)
            if hit_s is None and b <= Q:
                hit_s, hit_step = round(clk.s, 3), step
                print(f"  {name} >>> alcanza Q={Q:.4f} en {hit_s}s (paso {step})", flush=True)
    except RuntimeError as e:
        nan = True
        print(f"  {name} ABORTADO: {e}", flush=True)
    final = curve[-1][2] if curve else float("inf")
    st["runs"][name] = {
        "kind": "M_nowarm", "seed": seed, "lr": LR_HI, "warmup": False,
        "params": n_params(m), "final_bpc": final, "train_s": round(clk.s, 3),
        "nan": nan, "degenerado": bool(nan or final > DEGEN_THR),
        "modo": modo_fallo(curve) if curve else "abortado",
        "Q": Q, "T_R": T_R, "T_M": hit_s, "T_M_step": hit_step,
        "ratio": (hit_s / T_R) if (hit_s and T_R) else None,
        "events": events, "curve": curve,
    }
    r = st["runs"][name]
    print(f"  {name} FIN bpc={final:.4f} T_M={hit_s} degenerado={r['degenerado']}", flush=True)
    _save(st)
    del m, o
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    train, val, V = load_data("cuda")
    st = _state()

    # Q y T_R vienen de T10 (mismas referencias sanas); declarado en el preregistro
    with open(os.path.join(HERE, "t10_results.json"), encoding="utf-8") as f:
        t10 = json.load(f)
    Q, T_R = t10["ref_R_hi"]["Q"], t10["ref_R_hi"]["T_R"]
    st["Q_de_T10"], st["T_R_de_T10"] = Q, T_R
    print(f"[T11] Q={Q:.4f} T_R={T_R:.1f} (heredados de T10)", flush=True)
    _save(st)

    for s in seeds:
        run_R_warm(f"R_warm/s{s}", s, st, train, val, V)
    for s in seeds:
        run_M_nowarm(f"M_nowarm/s{s}", s, st, train, val, V, Q, T_R)

    # resumen del 2x2
    rw = [st["runs"][f"R_warm/s{s}"] for s in seeds if f"R_warm/s{s}" in st["runs"]]
    mn = [st["runs"][f"M_nowarm/s{s}"] for s in seeds if f"M_nowarm/s{s}" in st["runs"]]
    p_rw = sum(r["degenerado"] for r in rw)
    p_mn = sum(r["degenerado"] for r in mn)
    print(f"\n[2x2] R sin rampa: 4/6 (T10) | R CON rampa: {p_rw}/{len(rw)}", flush=True)
    print(f"[2x2] M SIN rampa: {p_mn}/{len(mn)} | M con rampa: 0/6 (T10)", flush=True)
    st["resumen_2x2"] = {"R_sin_rampa_T10": 4, "R_con_rampa": p_rw,
                         "M_sin_rampa": p_mn, "M_con_rampa_T10": 0,
                         "n_R_warm": len(rw), "n_M_nowarm": len(mn)}
    _save(st)


if __name__ == "__main__":
    main()
