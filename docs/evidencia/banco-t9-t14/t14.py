"""T14 - ¿limitaba el corpus, o los parametros rinden poco de verdad?

2x2 (escala x LR) sobre char-WikiText-103 truncado a 120M caracteres (0.20 epocas
con el presupuesto del arco, frente a 2.26 epocas en WT-2), mas la celda que falta
en WT-2 para que las dos tablas sean comparables.

Preregistro: PREREGISTRO-T14-datos-vs-escala.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import Clock, build, dump, make_opt, n_params, val_bpc_full
from mztrain.elastic_shape import GrowthEvent, LrWarmup, apply_event

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "t14_results.json")
BATCH, SEQ, LN2 = 16, 256, math.log(2.0)
PHASE1, PHASE2, EVAL_EVERY = 2000, 4000, 500
WARMUP_STEPS, WARMUP_FLOOR, NOISE = 200, 0.1, 1e-3
MAX_CHARS = 120_000_000
ESCALAS = {"S1": (192, 384, 6, 6), "S2": (384, 768, 8, 8)}


# ---------------------------------------------------------------- datos
def cargar(corpus: str):
    """Devuelve (train_int16_gpu, val_int16_gpu, vocab). El corpus se guarda en
    int16 en la GPU (a 120M caracteres son 240 MB) y cada lote se convierte a long:
    guardarlo ya en int64 costaria 960 MB de VRAM sin ninguna ganancia."""
    cache = os.path.join(HERE, f"{corpus}_char.pt")
    if os.path.exists(cache):
        blob = torch.load(cache, map_location="cpu")
    else:
        import datasets
        nombre = "wikitext-2-raw-v1" if corpus == "wt2" else "wikitext-103-raw-v1"
        ds = datasets.load_dataset("wikitext", nombre)
        tr = "".join(ds["train"]["text"])[:MAX_CHARS]
        va = "".join(ds["validation"]["text"])
        cps_tr = np.frombuffer(tr.encode("utf-32-le"), dtype=np.uint32)
        cps_va = np.frombuffer(va.encode("utf-32-le"), dtype=np.uint32)
        uniq = np.unique(np.concatenate([np.unique(cps_tr), np.unique(cps_va)]))
        blob = {"train": torch.from_numpy(np.searchsorted(uniq, cps_tr).astype(np.int16)),
                "val": torch.from_numpy(np.searchsorted(uniq, cps_va).astype(np.int16)),
                "vocab": int(len(uniq))}
        torch.save(blob, cache)
    return (blob["train"].cuda(), blob["val"].cuda(), int(blob["vocab"]))


def lote(data, gen, batch=BATCH, seq=SEQ):
    ix = torch.randint(len(data) - seq - 1, (batch,), generator=gen, device=data.device)
    x = torch.stack([data[i:i + seq] for i in ix]).long()
    y = torch.stack([data[i + 1:i + 1 + seq] for i in ix]).long()
    return x, y


# La metrica de evaluacion es UNA SOLA en todo el arco: la de bank. La version que
# vivia aqui topaba en 4000 ventanas frente a las 4461 del conjunto completo, y esa
# divergencia contamino las comparaciones entre experimentos (cribado 2026-08-22).
val_bpc = val_bpc_full


def pasos(model, opt, data, gen, n, clk, warm=None):
    model.train()
    with clk:
        for _ in range(n):
            x, y = lote(data, gen)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if warm is not None:
                warm.step()
            if float(loss) != float(loss):
                raise RuntimeError("NaN")


# ---------------------------------------------------------------- run
def _state():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {"experiment": "T14", "protocol": {
        "corpus_grande": f"char-WikiText-103 truncado a {MAX_CHARS} caracteres",
        "batch": BATCH, "seq": SEQ, "phase1": PHASE1, "phase2": PHASE2,
        "warmup": [WARMUP_STEPS, WARMUP_FLOOR], "condicion": "morph"}, "runs": {}}


def run(name, corpus, escala, lr, seed, st, datos):
    if name in st["runs"]:
        print(f"[skip] {name}", flush=True)
        return
    train, val, V = datos[corpus]
    d_small, d_big, L, H = ESCALAS[escala]
    m = build(V, d_small, L, H, None, "cuda", seed)
    o = make_opt(m, lr=lr)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242 + seed)
    clk, curva = Clock(), []
    for hecho in range(0, PHASE1, EVAL_EVERY):
        pasos(m, o, train, g, EVAL_EVERY, clk)
        curva.append([hecho + EVAL_EVERY, round(clk.s, 2), round(val_bpc(m, val), 6)])
    px = torch.stack([val[i * SEQ:(i + 1) * SEQ] for i in range(8)]).long()
    o, rep = apply_event(m, o, GrowthEvent(step=0, factorize=True, new_d=d_big,
                                           noise_scale=NOISE),
                         probe_x=px, generator=gg)
    warm = LrWarmup(o, WARMUP_STEPS, WARMUP_FLOOR)
    for hecho in range(0, PHASE2, EVAL_EVERY):
        pasos(m, o, train, g, EVAL_EVERY, clk, warm)
        b = val_bpc(m, val)
        curva.append([PHASE1 + hecho + EVAL_EVERY, round(clk.s, 2), round(b, 6)])
        print(f"  {name} paso={PHASE1 + hecho + EVAL_EVERY} bpc={b:.4f}", flush=True)
    st["runs"][name] = {"corpus": corpus, "escala": escala, "lr": lr, "seed": seed,
                        "params": n_params(m), "vocab": V,
                        "final_bpc": curva[-1][2], "train_s": round(clk.s, 2),
                        "deriva": rep.get("logits_drift_rel"), "curva": curva}
    print(f"  {name} FIN bpc={curva[-1][2]:.4f}  ({clk.s:.0f}s)", flush=True)
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump(RESULTS, st)
    del m, o
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default="")
    a = ap.parse_args()
    st = _state()
    datos = {}
    for c in ("wt103", "wt2"):
        datos[c] = cargar(c)
        tr, va, V = datos[c]
        print(f"{c}: train {len(tr):,} car | val {len(va):,} | vocab {V} | "
              f"{6000 * BATCH * SEQ / len(tr):.2f} epocas en 6000 pasos", flush=True)

    plan = [("wt103", "S1", 3e-4, 0), ("wt103", "S1", 3e-4, 1),
            ("wt103", "S1", 1.2e-3, 0), ("wt103", "S1", 1.2e-3, 1),
            ("wt103", "S2", 3e-4, 0), ("wt103", "S2", 1.2e-3, 0),
            ("wt2", "S2", 1.2e-3, 0)]          # la celda que faltaba en WT-2
    for corpus, esc, lr, s in plan:
        nombre = f"{corpus}/{esc}/lr{lr:g}/s{s}"
        if a.solo and a.solo not in nombre:
            continue
        run(nombre, corpus, esc, lr, s, st, datos)


if __name__ == "__main__":
    main()
