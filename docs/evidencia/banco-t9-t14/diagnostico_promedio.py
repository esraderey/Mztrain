"""Diagnostico: ¿QUE mata al promedio pre-cirugia?

Hipotesis a contrastar en el mismo punto del entrenamiento (paso 2000, denso 192):
  H1 "velocidad cero": el promedio esta fuera de la trayectoria y el optimizador
      tiene que reconstruir el momento.  -> predice gradiente MENOR en el promedio.
  H2 "colapso del espectro": promediar cancela las direcciones que oscilan, que son
      las de valores singulares pequenos; al factorizar, esas direcciones nacen con
      S diminuto y el gradiente de U,V escala con S -> nacen casi congeladas.
      -> predice cola del espectro MAS pequena y rango efectivo MENOR en el promedio.

Se miden ambas cosas, y ademas el efecto de un promedio SUAVE (alpha=0.25), que la
sonda previa sugirio que conserva casi toda la ganancia (-0.0385 contra -0.0431).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import Clock, build, get_batch, load_data, make_opt, train_steps, val_bpc_full
from mztrain.elastic_shape import GrowthEvent, apply_event

LR, K = 1.2e-3, 100
PHASE1 = 2000


def snap(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def mezcla(a, b, w):
    """(1-w)*b + w*a  -> w=0.5 es la media; w=0.25 es el promedio suave."""
    return {k: ((1 - w) * v + w * a[k] if v.dtype.is_floating_point and k in a else v)
            for k, v in b.items()}


def espectro(sd):
    """Estadisticos del espectro de las matrices densas de los bloques."""
    out = []
    for k, v in sd.items():
        if v.ndim == 2 and "blocks" in k and v.shape[0] > 8:
            s = torch.linalg.svdvals(v.float())
            energia = s.pow(2).sum()
            # rango efectivo (participation ratio de la energia)
            p = s.pow(2) / energia
            r_eff = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
            out.append({
                "capa": k, "s_max": float(s[0]), "s_min": float(s[-1]),
                "cond": float(s[0] / s[-1].clamp_min(1e-12)),
                "cola_10pct": float(s[int(0.9 * len(s)):].pow(2).sum() / energia),
                "r_eff": r_eff, "fro": float(energia.sqrt()),
            })
    return out


def resumen(e, nombre):
    n = len(e)
    print(f"  {nombre:26s} s_min medio {sum(x['s_min'] for x in e) / n:.5f} | "
          f"cond media {sum(x['cond'] for x in e) / n:8.1f} | "
          f"rango efectivo medio {sum(x['r_eff'] for x in e) / n:6.2f} | "
          f"energia cola-10% {sum(x['cola_10pct'] for x in e) / n * 100:.4f}% | "
          f"||W||_F {sum(x['fro'] for x in e) / n:.3f}", flush=True)


def grad_norm(m, train, g, n=8):
    """Norma media del gradiente sobre n lotes fijos (sin actualizar nada)."""
    tot = 0.0
    gg = torch.Generator(device="cuda")
    gg.manual_seed(31337)
    for _ in range(n):
        x, y = get_batch(train, gg)
        m.zero_grad(set_to_none=True)
        _, loss = m(x, y)
        loss.backward()
        s = sum(p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None)
        tot += float(s.sqrt())
    m.zero_grad(set_to_none=True)
    return tot / n


def main():
    train, val, V = load_data("cuda")
    m = build(V, 192, 6, 6, None, "cuda", 0)
    o = make_opt(m, lr=LR)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000)
    train_steps(m, o, train, g, PHASE1 - K, Clock())
    est_prev = snap(m)
    train_steps(m, o, train, g, K, Clock())
    est_t = snap(m)

    print("=== EN EL PUNTO PRE-CIRUGIA (denso d=192, paso 2000) ===", flush=True)
    variantes = [("original (theta_t)", 0.0), ("promedio suave (w=0.25)", 0.25),
                 ("promedio (w=0.50)", 0.5)]
    datos = {}
    for nombre, w in variantes:
        sd = mezcla(est_prev, est_t, w) if w else est_t
        m.load_state_dict(sd)
        b = val_bpc_full(m, val)
        gn = grad_norm(m, train, g)
        e = espectro(sd)
        datos[nombre] = {"bpc": b, "grad": gn, "sd": sd}
        print(f"\n  [{nombre}]  BPC {b:.4f}   ||grad|| {gn:.4f}", flush=True)
        resumen(e, "espectro:")

    print("\n=== TRAS LA CIRUGIA (factorizado d=384) ===", flush=True)
    for nombre, _ in variantes:
        mm = build(V, 192, 6, 6, None, "cuda", 0)
        mm.load_state_dict(datos[nombre]["sd"])
        oo = make_opt(mm, lr=LR)
        oo, rep = apply_event(mm, oo, GrowthEvent(step=0, factorize=True, new_d=384,
                                                  noise_scale=1e-3))
        b = val_bpc_full(mm, val)
        gn = grad_norm(mm, train, g)
        # gradiente separado por factor
        gs = {"U": 0.0, "S": 0.0, "V": 0.0}
        for blk in mm.blocks:
            for lay in (blk.qkv, blk.proj, blk.fc1, blk.fc2):
                for k in gs:
                    p = getattr(lay, k)
                    if p.grad is not None:
                        gs[k] += float(p.grad.pow(2).sum())
        gs = {k: v ** 0.5 for k, v in gs.items()}
        s_all = torch.cat([blk.qkv.S.detach().abs().flatten() for blk in mm.blocks])
        print(f"  [{nombre:24s}] BPC {b:.4f}  ||grad|| {gn:.4f}  "
              f"gU {gs['U']:.4f} gS {gs['S']:.4f} gV {gs['V']:.4f}  "
              f"S_qkv: min {float(s_all.min()):.5f} mediana {float(s_all.median()):.4f}",
              flush=True)
        del mm, oo
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
