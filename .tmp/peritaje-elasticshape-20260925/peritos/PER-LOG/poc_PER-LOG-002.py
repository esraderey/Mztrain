"""PER-LOG-002: apply_event descarta en silencio lr/weight_decay explicitos
cuando el evento no llega a migrar (factorize redundante, new_d == d).

Propiedad (SAFETY: "un override explicito afecta a todos"): si
apply_event(..., lr=L, weight_decay=W) devuelve accepted=True, todos los
grupos del optimizer devuelto tienen lr == L y weight_decay == W.
Exit 0 = defecto no reproducido; AssertionError = defecto confirmado.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))

import torch  # noqa: E402

import mztrain.elastic_shape as shape  # noqa: E402

LR, WD = 5e-4, 0.05


def setup():
    torch.manual_seed(1)
    model = shape.GPT(64, 16, 24, 1, 2, shape.fact_lin(8))  # ya factorizado
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randint(0, 64, (2, 16))
    opt.zero_grad()
    model(x, x)[1].backward()
    opt.step()
    return model, opt


def check(event):
    model, opt = setup()
    new, report = shape.apply_event(model, opt, event, lr=LR, weight_decay=WD)
    assert report["accepted"]
    return [(g["lr"], g["weight_decay"]) for g in new.param_groups], report["optimizer"]


# Control negativo: evento que migra (deepen) aplica el override.
groups, log = check(shape.GrowthEvent(1, add_layers=1))
assert all(g == (LR, WD) for g in groups), groups
print(f"[control OK] add_layers=1 -> grupos {groups}")

failures = []
for name, ev in [
    ("factorize redundante", shape.GrowthEvent(1, factorize=True)),
    ("new_d == d", shape.GrowthEvent(1, new_d=24)),
]:
    groups, log = check(ev)
    print(f"[{name}] grupos={groups} reporte={log}")
    if not all(g == (LR, WD) for g in groups):
        failures.append((name, groups))

assert not failures, (
    f"PER-LOG-002 CONFIRMADO: accepted=True pero el override lr={LR}/wd={WD} no se aplico: {failures}"
)
print("PER-LOG-002 no reproducido")
