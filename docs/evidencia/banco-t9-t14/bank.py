"""Banco T9 - reconstruccion del harness del arco empirico (protocolo T4/T6/T8).

char-WikiText-2, batch 16, seq 256, AdamW lr 3e-4 wd 0.01 sin schedule,
bias=False, embedding atado, heads fijos. Metrica primaria val_bpc_full
(todo el val set, ventanas no solapadas de 256, SIEMPRE fp32).
Reloj de TRAIN puro: los evals no cuentan.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "wt2_char.pt")

BATCH, SEQ = 16, 256
LR, WD = 3e-4, 0.01
LN2 = math.log(2.0)


# ---------------------------------------------------------------- data
def load_data(device: str = "cuda"):
    if os.path.exists(CACHE):
        blob = torch.load(CACHE, map_location="cpu")
    else:
        import datasets
        ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
        tr = "".join(ds["train"]["text"])
        va = "".join(ds["validation"]["text"])
        chars = sorted(set(tr) | set(va))
        stoi = {c: i for i, c in enumerate(chars)}
        blob = {
            "train": torch.tensor([stoi[c] for c in tr], dtype=torch.int16),
            "val": torch.tensor([stoi[c] for c in va], dtype=torch.int16),
            "vocab": len(chars),
        }
        torch.save(blob, CACHE)
    train = blob["train"].to(device=device, dtype=torch.long)
    val = blob["val"].to(device=device, dtype=torch.long)
    return train, val, int(blob["vocab"])


def get_batch(data: torch.Tensor, gen: torch.Generator, batch: int = BATCH,
              seq: int = SEQ):
    ix = torch.randint(len(data) - seq - 1, (batch,), generator=gen,
                       device=data.device)
    x = torch.stack([data[i:i + seq] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + seq] for i in ix])
    return x, y


@torch.no_grad()
def val_bpc_full(model, val: torch.Tensor, seq: int = SEQ, chunk: int = 32) -> float:
    """BPC sobre TODO el val set: ventanas secuenciales no solapadas.
    Siempre fp32 (sin autocast) para comparabilidad exacta con T4/T6/T8."""
    was_training = model.training
    model.eval()
    n = (len(val) - 1) // seq
    tot_nats, tot_tok = 0.0, 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # .long() es no-op si val ya es int64; permite que un corpus guardado en
        # int16 (T14: 120M caracteres = 240 MB en vez de 960) use ESTA misma metrica
        # en vez de una copia divergente. El cribado del 2026-08-22 encontro que la
        # copia de T14 topaba en 4000 ventanas frente a las 4461 de aqui.
        idx = torch.stack([val[i * seq:(i + 1) * seq] for i in range(s, e)]).long()
        tgt = torch.stack([val[i * seq + 1:(i + 1) * seq + 1] for i in range(s, e)]).long()
        _, loss = model(idx, tgt)
        ntok = idx.numel()
        tot_nats += float(loss) * ntok
        tot_tok += ntok
    if was_training:
        model.train()
    return tot_nats / tot_tok / LN2


# ---------------------------------------------------------------- model
def build(vocab: int, d: int, layers: int, heads: int, rank: Optional[int],
          device: str = "cuda", seed: int = 0):
    from mztrain.elastic_shape import GPT, dense_lin, fact_lin
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    lin = dense_lin if rank is None else fact_lin(rank)
    return GPT(vocab, SEQ, d, layers, heads, lin).to(device)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def make_opt(model, lr: float = LR, wd: float = WD):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


# ---------------------------------------------------------------- loop
@dataclass
class Curve:
    step: List[int] = field(default_factory=list)
    train_s: List[float] = field(default_factory=list)
    bpc: List[float] = field(default_factory=list)

    def add(self, step: int, train_s: float, bpc: float) -> None:
        self.step.append(step)
        self.train_s.append(round(train_s, 3))
        self.bpc.append(round(bpc, 6))


class Clock:
    """Reloj de train puro: acumula solo el tiempo dentro de train_steps."""
    def __init__(self) -> None:
        self.s = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        torch.cuda.synchronize()
        self.s += time.perf_counter() - self._t


def train_steps(model, opt, data, gen, n: int, clock: Clock,
                warmup: Optional[object] = None, amp: bool = False) -> float:
    """n pasos de train. Devuelve la ultima loss (nats). NaN aborta."""
    model.train()
    last = float("nan")
    with clock:
        for _ in range(n):
            x, y = get_batch(data, gen)
            if amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if warmup is not None:
                warmup.step()
            last = float(loss)
            if last != last:
                raise RuntimeError("NaN en train")
    return last


def dump(path: str, obj) -> None:
    """Dump atomico incremental (patron del arco)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
