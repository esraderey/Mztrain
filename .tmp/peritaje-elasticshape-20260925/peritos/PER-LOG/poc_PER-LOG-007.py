"""PER-LOG-007: LrWarmup valida steps pero no floor; floor NaN/negativo produce
LR NaN/negativo (ascenso de gradiente) en silencio.

Propiedad: LrWarmup(opt, steps, floor) con floor no finito o < 0 lanza
ValueError, igual que steps <= 0; nunca deja g["lr"] no finito o negativo.
Exit 0 = defecto no reproducido; AssertionError = defecto confirmado.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import mztrain.elastic_shape as shape  # noqa: E402


def fresh():
    p = nn.Parameter(torch.ones(3))
    return p, torch.optim.AdamW([p], lr=1e-3, weight_decay=0.0)


# Control negativo: steps invalido se rechaza; floor valido funciona.
_, opt = fresh()
try:
    shape.LrWarmup(opt, steps=0)
    raise SystemExit("control roto: steps=0 aceptado")
except ValueError:
    pass
_, opt = fresh()
w = shape.LrWarmup(opt, steps=2, floor=0.1)
assert math.isclose(opt.param_groups[0]["lr"], 1e-4)
print("[control OK] steps=0 rechazado, floor=0.1 correcto")

accepted = []
for bad in (float("nan"), -0.5):
    p, opt = fresh()
    try:
        shape.LrWarmup(opt, steps=10, floor=bad)
    except ValueError:
        continue
    lr = opt.param_groups[0]["lr"]
    p.grad = torch.ones_like(p)
    opt.step()
    accepted.append((bad, lr, p.detach().tolist()))
    print(f"[floor={bad}] aceptado: lr={lr} param tras un paso={p.detach().tolist()}")

assert not accepted, f"PER-LOG-007 CONFIRMADO: floor invalido aceptado en silencio: {accepted}"
print("PER-LOG-007 no reproducido")
