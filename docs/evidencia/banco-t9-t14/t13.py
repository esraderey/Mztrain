"""T13 - promediar los pesos justo ANTES de la cirugia.

Comparacion PAREADA e intercalada por semilla (ctl/s0, avg/s0, ctl/s1, ...):
T12 midio que el reloj deriva hasta 10.6% entre sesiones por termica, y el
intercalado la neutraliza. Preregistro: PREREGISTRO-T13-promedio-precirugia.md.
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
RESULTS = os.path.join(HERE, "t13_results.json")
LR = 1.2e-3
Q_HEREDADA = 2.4536          # linea base fuerte de T11 (valor de calidad, no reloj)
WARMUP_STEPS, WARMUP_FLOOR, NOISE = 200, 0.1, 1e-3
PHASE1, PHASE2_CAP, EVAL_EVERY = 2000, 4000, 250
K_SNAP = 100                 # separacion entre los dos snapshots promediados
D_SMALL, D_BIG, LAYERS, HEADS = 192, 384, 6, 6


def _state():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {"experiment": "T13", "protocol": {
        "lr": LR, "Q_heredada_de_T11": Q_HEREDADA, "phase1": PHASE1,
        "phase2_cap": PHASE2_CAP, "warmup": [WARMUP_STEPS, WARMUP_FLOOR],
        "K_snapshots": K_SNAP, "noise_scale": NOISE,
        "diseno": "pareado, intercalado por semilla (anti deriva termica, T12)",
    }, "runs": {}}


def _save(st):
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump(RESULTS, st)


def probe_batch(val, n=8, seq=256):
    return torch.stack([val[i * seq:(i + 1) * seq] for i in range(n)])


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def promediar(m, a, b):
    """theta <- (a+b)/2 en los tensores flotantes; el resto se conserva."""
    s = {k: ((a[k] + v) / 2 if v.dtype.is_floating_point and k in a else v)
         for k, v in b.items()}
    m.load_state_dict(s)


def run(name, seed, promedio, st, train, val, V):
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    m = build(V, D_SMALL, LAYERS, HEADS, None, "cuda", seed)
    o = make_opt(m, lr=LR)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242 + seed)
    px = probe_batch(val)
    clk, curve = Clock(), []

    # --- fase densa, con snapshot en PHASE1-K y en PHASE1
    hecho = 0
    est_prev = None
    for objetivo in list(range(EVAL_EVERY, PHASE1 - K_SNAP + 1, EVAL_EVERY)):
        train_steps(m, o, train, g, objetivo - hecho, clk)
        hecho = objetivo
        b = val_bpc_full(m, val)
        curve.append([hecho, round(clk.s, 3), round(b, 6)])
    train_steps(m, o, train, g, (PHASE1 - K_SNAP) - hecho, clk)
    hecho = PHASE1 - K_SNAP
    est_prev = snap(m)
    train_steps(m, o, train, g, K_SNAP, clk)
    hecho = PHASE1
    bpc_pre_prom = val_bpc_full(m, val)

    # --- el promedio (cuenta en el reloj)
    bpc_post_prom = bpc_pre_prom
    if promedio:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        promediar(m, est_prev, snap(m))
        torch.cuda.synchronize()
        clk.s += time.perf_counter() - t0
        bpc_post_prom = val_bpc_full(m, val)
        print(f"  {name} PROMEDIO  {bpc_pre_prom:.4f} -> {bpc_post_prom:.4f} "
              f"({bpc_post_prom - bpc_pre_prom:+.4f})", flush=True)
    curve.append([hecho, round(clk.s, 3), round(bpc_post_prom, 6)])

    # --- cirugia
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o, rep = apply_event(m, o, GrowthEvent(step=0, factorize=True, new_d=D_BIG,
                                           noise_scale=NOISE),
                         probe_x=px, generator=gg)
    torch.cuda.synchronize()
    clk.s += time.perf_counter() - t0
    warm = LrWarmup(o, WARMUP_STEPS, WARMUP_FLOOR)
    bpc_post_cir = val_bpc_full(m, val)
    print(f"  {name} CIRUGIA -> d={m.d} deriva={rep.get('logits_drift_rel'):.4f} "
          f"bpc {bpc_post_prom:.4f}->{bpc_post_cir:.4f}", flush=True)

    # --- fase ancha
    hit_s = hit_step = None
    for done in range(0, PHASE2_CAP, EVAL_EVERY):
        train_steps(m, o, train, g, EVAL_EVERY, clk, warmup=warm)
        b = val_bpc_full(m, val)
        step = PHASE1 + done + EVAL_EVERY
        curve.append([step, round(clk.s, 3), round(b, 6)])
        if hit_s is None and b <= Q_HEREDADA:
            hit_s, hit_step = round(clk.s, 3), step
            print(f"  {name} >>> alcanza Q={Q_HEREDADA} en {hit_s}s (paso {step})", flush=True)
    st["runs"][name] = {
        "kind": "M_avg" if promedio else "M_ctl", "seed": seed, "promedio": promedio,
        "params": n_params(m), "final_bpc": curve[-1][2], "train_s": round(clk.s, 3),
        "bpc_pre_promedio": bpc_pre_prom, "bpc_post_promedio": bpc_post_prom,
        "delta_promedio": bpc_post_prom - bpc_pre_prom,
        "deriva_cirugia": rep.get("logits_drift_rel"), "bpc_post_cirugia": bpc_post_cir,
        "Q": Q_HEREDADA, "T_M": hit_s, "T_M_step": hit_step, "curve": curve,
    }
    print(f"  {name} FIN bpc={curve[-1][2]:.4f} T_M={hit_s}", flush=True)
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
    for s in seeds:                      # INTERCALADO: ctl y avg juntos por semilla
        run(f"M_ctl/s{s}", s, False, st, train, val, V)
        run(f"M_avg/s{s}", s, True, st, train, val, V)

    pares = []
    for s in seeds:
        c, v = st["runs"].get(f"M_ctl/s{s}"), st["runs"].get(f"M_avg/s{s}")
        if c and v and c["T_M"] and v["T_M"]:
            pares.append((s, c["T_M"], v["T_M"], v["T_M"] - c["T_M"]))
    if pares:
        mejoran = sum(1 for _, _, _, d in pares if d < 0)
        mc = sum(p[1] for p in pares) / len(pares)
        gan = -sum(p[3] for p in pares) / len(pares) / mc
        print(f"\n[PAREADO] {len(pares)} semillas | mejoran {mejoran}/{len(pares)} | "
              f"ganancia relativa media {gan * 100:+.2f}%", flush=True)
        for s, c, v, d in pares:
            print(f"   s{s}: ctl {c:7.1f}s  avg {v:7.1f}s  delta {d:+7.1f}s", flush=True)
        st["pareado"] = {"n": len(pares), "mejoran": mejoran, "ganancia_rel": gan,
                         "detalle": pares}
        _save(st)


if __name__ == "__main__":
    main()
