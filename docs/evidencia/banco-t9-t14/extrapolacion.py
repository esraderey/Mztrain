"""Falsacion barata de la premisa del "aprendizaje predictivo":

Si en el paso t tomo la direccion reciente d = theta_t - theta_{t-K} y salto a
theta_t + alpha*d, ¿mejora la perdida? ¿Y cuantos pasos de entrenamiento REAL
vale ese salto?

Se mide en tres puntos con dinamicas distintas: fase densa, justo despues de la
cirugia (el transitorio, donde la trayectoria deberia ser mas predecible) y ya
estabilizado. Metrica: BPC de validacion completo, fp32.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import Clock, build, dump, load_data, make_opt, train_steps, val_bpc_full
from mztrain.elastic_shape import GrowthEvent, LrWarmup, apply_event

LR = 1.2e-3
K = 100                       # ventana de la direccion
ALPHAS = (0.5, 1.0, 2.0, 4.0)
HORIZONTES = (25, 50, 100, 200, 400)   # pasos reales contra los que se compara


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def cargar(m, s):
    m.load_state_dict(s)


def extrapolar(m, viejo, nuevo, alpha):
    """theta <- theta_nuevo + alpha*(theta_nuevo - theta_viejo), solo en floats."""
    s = {}
    for k, v in nuevo.items():
        if v.dtype.is_floating_point and k in viejo:
            s[k] = v + alpha * (v - viejo[k])
        else:
            s[k] = v
    cargar(m, s)


def prueba(nombre, m, o, train, val, g, out):
    """En el estado actual: mide la extrapolacion y la compara con seguir entrenando."""
    theta_prev = None
    clk = Clock()
    train_steps(m, o, train, g, K, clk)          # avanza K para tener direccion
    theta_prev = None
    est_t_menos_K = snap(m)
    train_steps(m, o, train, g, K, clk)
    est_t = snap(m)
    est_opt = copy.deepcopy(o.state_dict())

    base = val_bpc_full(m, val)
    r = {"nombre": nombre, "bpc_base": base, "extrapolacion": {}, "entrenamiento_real": {}}
    print(f"\n== {nombre} ==  BPC en theta_t: {base:.4f}", flush=True)

    # (a) que da saltar en la direccion reciente
    for a in ALPHAS:
        extrapolar(m, est_t_menos_K, est_t, a)
        b = val_bpc_full(m, val)
        r["extrapolacion"][str(a)] = b
        print(f"   extrapolar alpha={a:<4} -> {b:.4f}  ({b - base:+.4f})", flush=True)

    # (b) que da seguir entrenando de verdad
    cargar(m, est_t)
    o.load_state_dict(est_opt)
    hecho = 0
    for h in HORIZONTES:
        train_steps(m, o, train, g, h - hecho, Clock())
        hecho = h
        b = val_bpc_full(m, val)
        r["entrenamiento_real"][str(h)] = b
        print(f"   entrenar {h:4d} pasos      -> {b:.4f}  ({b - base:+.4f})", flush=True)

    mejor_a = min(r["extrapolacion"], key=lambda k: r["extrapolacion"][k])
    mejor_b = r["extrapolacion"][mejor_a]
    equiv = None
    for h in HORIZONTES:
        if r["entrenamiento_real"][str(h)] <= mejor_b:
            equiv = h
            break
    r["mejor_alpha"] = float(mejor_a)
    r["pasos_equivalentes"] = equiv
    print(f"   -> mejor extrapolacion alpha={mejor_a} vale "
          f"{'<%d' % HORIZONTES[0] if equiv == HORIZONTES[0] else equiv} pasos reales"
          if equiv else "   -> la extrapolacion NO alcanza ni 25 pasos reales", flush=True)
    out["pruebas"].append(r)
    dump("extrapolacion_results.json", out)
    # deja el modelo donde estaba, con su optimizador
    cargar(m, est_t)
    o.load_state_dict(est_opt)


def main():
    train, val, V = load_data("cuda")
    out = {"lr": LR, "K": K, "alphas": list(ALPHAS), "horizontes": list(HORIZONTES),
           "pruebas": []}
    m = build(V, 192, 6, 6, None, "cuda", 0)
    o = make_opt(m, lr=LR)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000)

    train_steps(m, o, train, g, 800, Clock())
    prueba("fase densa (paso ~1000)", m, o, train, val, g, out)

    train_steps(m, o, train, g, 2000 - 1000 - 400, Clock())
    px = torch.stack([val[i * 256:(i + 1) * 256] for i in range(8)])
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242)
    o, rep = apply_event(m, o, GrowthEvent(step=0, factorize=True, new_d=384,
                                           noise_scale=1e-3), probe_x=px, generator=gg)
    print(f"\n[cirugia aplicada: deriva {rep.get('logits_drift_rel'):.4f}]", flush=True)
    prueba("transitorio post-cirugia", m, o, train, val, g, out)

    train_steps(m, o, train, g, 1000, Clock())
    prueba("estabilizado (fase ancha)", m, o, train, val, g, out)
    print("\nlisto", flush=True)


if __name__ == "__main__":
    main()
