"""T13 - Tabla de capacidad REAL en la RTX 4060 (8 GB), con vocabulario de 50k BPE.

Mide, para las formas de la familia GPT-2, que se puede ENTRENAR de verdad:
densoo factorizado, fp32 o bf16, VRAM pico, tokens/s y si cae en el acantilado
WDDM (Windows empieza a derramar a memoria compartida: no hay OOM, se hunde el
rendimiento). Vocab 50257 = el coste real de embeddings y logits.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import Clock, dump
from mztrain.elastic_shape import GPT, dense_lin, fact_lin

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "capacity_results.json")

VOCAB = 50257          # BPE real (GPT-2), no el de caracteres
BATCH, SEQ = 8, 256    # 2048 tokens/paso; con acumulacion se agranda sin coste de VRAM
STEPS, WARM = 5, 2
CLIFF_GB = 7.0         # umbral practico de la 4060 bajo WDDM (T5)

# (nombre, d, capas, cabezas) - formas de la familia GPT-2
FORMAS = [
    ("GPT-2 small  (124M)", 768, 12, 12),
    ("GPT-2 medium (355M)", 1024, 24, 16),
    ("GPT-2 large  (774M)", 1280, 36, 20),
    ("GPT-2 XL    (1558M)", 1600, 48, 25),
]


def params_teoricos(d, L, vocab, rank=None):
    """Cuenta exacta de parametros del GPT del arco (head atado, bias=False)."""
    emb = vocab * d + SEQ * d
    if rank is None:
        bloque = d * 3 * d + d * d + d * 4 * d + 4 * d * d          # qkv, proj, fc1, fc2
    else:
        def f(i, o):
            return o * rank + rank + rank * i
        bloque = f(d, 3 * d) + f(d, d) + f(d, 4 * d) + f(4 * d, d)
    ln = 2 * d * L * 2 + 2 * d
    return emb + bloque * L + ln


def probe(nombre, d, L, h, rank, amp):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    etiqueta = (f"{nombre} | {'denso' if rank is None else f'fact r=d/{d // rank}'} "
                f"| {'bf16' if amp else 'fp32'}")
    res = {"forma": nombre, "d": d, "layers": L, "heads": h, "rank": rank,
           "amp": amp, "params": None, "params_densos_equiv": params_teoricos(d, L, VOCAB),
           "vram_gb": None, "s_por_paso": None, "tokens_s": None, "estado": None}
    try:
        lin = dense_lin if rank is None else fact_lin(rank)
        m = GPT(VOCAB, SEQ, d, L, h, lin).to("cuda")
        res["params"] = sum(p.numel() for p in m.parameters())
        o = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
        g = torch.Generator(device="cuda")
        g.manual_seed(0)
        x = torch.randint(VOCAB, (BATCH, SEQ), generator=g, device="cuda")
        y = torch.randint(VOCAB, (BATCH, SEQ), generator=g, device="cuda")

        def paso():
            if amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = m(x, y)
            else:
                _, loss = m(x, y)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()

        for _ in range(WARM):
            paso()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            paso()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / STEPS
        res["s_por_paso"] = round(dt, 4)
        res["tokens_s"] = int(BATCH * SEQ / dt)
        res["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        res["estado"] = "acantilado" if res["vram_gb"] > CLIFF_GB else "ok"
        del m, o, x, y
    except torch.cuda.OutOfMemoryError:
        res["estado"] = "OOM"
    except RuntimeError as e:
        res["estado"] = "OOM" if "out of memory" in str(e).lower() else f"error: {e}"
    torch.cuda.empty_cache()
    p = res["params"]
    print(f"  {etiqueta:44s} {'-' if p is None else f'{p / 1e6:7.1f}M'}  "
          f"vram={res['vram_gb']}  {res['tokens_s']} tok/s  [{res['estado']}]", flush=True)
    return res


def main():
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "vocab": VOCAB, "batch": BATCH, "seq": SEQ, "cliff_gb": CLIFF_GB,
           "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2),
           "probes": []}
    print(f"{out['gpu']} | {out['vram_total_gb']} GB | torch {out['torch']} | "
          f"vocab {VOCAB} | batch {BATCH} x seq {SEQ}\n", flush=True)
    for nombre, d, L, h in FORMAS:
        for rank in (None, d // 2, d // 6):
            for amp in (False, True):
                out["probes"].append(probe(nombre, d, L, h, rank, amp))
                dump(RESULTS, out)
        print(flush=True)
    print("listo", flush=True)


if __name__ == "__main__":
    main()
