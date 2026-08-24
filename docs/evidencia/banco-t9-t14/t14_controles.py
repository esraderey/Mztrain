"""Controles de T14: ¿es el nulo un artefacto del tratamiento de los datos?

Candidato 1 (truncado no representativo): YA DESCARTADO sin GPU — los 120M
caracteres son el 22.3% del corpus, cubren 6958 articulos (23.3%, proporcional) y
la distancia de variacion total frente al trozo siguiente es 0.0027.

Aqui van los dos que necesitan GPU:

  C2 - VOCABULARIO. WT-103 tiene 2989 caracteres frente a 1118 de WT-2: 1871 clases
       extra casi nunca observadas en el softmax. Si eso encarece el BPC, estaria
       enmascarando una mejora real. Control: entrenar en WT-2 con el vocabulario de
       WT-103, y comparar contra el mismo entrenamiento con vocabulario propio.

  C3 - LA PREMISA. Se asumio que WT-2 estaba saturado por hacer 2.26 epocas. Se
       comprueba directamente midiendo BPC de ENTRENAMIENTO contra BPC de
       VALIDACION: si estan pegados, no habia memorizacion y por tanto no habia
       nada que la ampliacion del corpus pudiera arreglar. El nulo seria real.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import Clock, build, dump, make_opt, n_params
from t14 import EVAL_EVERY, PHASE1, PHASE2, cargar, pasos, val_bpc
from mztrain.elastic_shape import GrowthEvent, LrWarmup, apply_event

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "t14_controles.json")


def morph(V, lr, seed, train, val):
    """Mismo protocolo que T14: 2000 densos -> cirugia -> 4000 anchos."""
    m = build(V, 192, 6, 6, None, "cuda", seed)
    o = make_opt(m, lr=lr)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242 + seed)
    pasos(m, o, train, g, PHASE1, Clock())
    px = torch.stack([val[i * 256:(i + 1) * 256] for i in range(8)]).long()
    o, _ = apply_event(m, o, GrowthEvent(step=0, factorize=True, new_d=384,
                                         noise_scale=1e-3), probe_x=px, generator=gg)
    warm = LrWarmup(o, 200, 0.1)
    pasos(m, o, train, g, PHASE2, Clock(), warm)
    return m


def main():
    out = {"controles": {}}
    tr103, va103, V103 = cargar("wt103")
    tr2, va2, V2 = cargar("wt2")

    print("=== C2: el vocabulario mas grande, ¿encarece el BPC? ===", flush=True)
    print("  se entrena en WT-2 (mismos datos) con cada vocabulario", flush=True)
    for etiqueta, V in (("vocab propio de WT-2 (1118)", V2),
                        ("vocab de WT-103 (2989)", V103)):
        m = morph(V, 1.2e-3, 0, tr2, va2)
        b = val_bpc(m, va2)
        out["controles"][f"C2/{etiqueta}"] = {"bpc": b, "params": n_params(m), "vocab": V}
        print(f"  {etiqueta:30s} BPC {b:.4f}  ({n_params(m) / 1e6:.2f}M params)", flush=True)
        del m
        torch.cuda.empty_cache()
        dump(RESULTS, out)

    print("\n=== C3: ¿estaba WT-2 saturado? (train vs val) ===", flush=True)
    for corpus, tr, va, V, epocas in (("wt2", tr2, va2, V2, 2.26),
                                      ("wt103", tr103, va103, V103, 0.20)):
        m = morph(V, 1.2e-3, 0, tr, va)
        b_val = val_bpc(m, va)
        # BPC de entrenamiento sobre el mismo numero de ventanas, desde el inicio
        b_tr = val_bpc(m, tr[:len(va)])
        out["controles"][f"C3/{corpus}"] = {"bpc_val": b_val, "bpc_train": b_tr,
                                            "gap": b_val - b_tr, "epocas": epocas}
        print(f"  {corpus:6s} ({epocas:.2f} epocas)  train {b_tr:.4f} | val {b_val:.4f} | "
              f"brecha {b_val - b_tr:+.4f}", flush=True)
        del m
        torch.cuda.empty_cache()
        dump(RESULTS, out)
    print("\n  Una brecha pequena = sin memorizacion = el corpus NO limitaba.", flush=True)


if __name__ == "__main__":
    main()
