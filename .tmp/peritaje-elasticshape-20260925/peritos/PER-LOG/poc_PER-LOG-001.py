"""PER-LOG-001: LrWarmup reutiliza un _mzshape_base_lr caduco.

Propiedad: tras un growth aceptado, LrWarmup(new_opt, n) termina (done) con
lr == lr vigente justo antes del growth. Falla si entre growths el LR cambio
(decaimiento externo / step decay) despues de un warmup ya TERMINADO: la base
persistente nunca se borra y el nuevo warmup rampa hasta el pico viejo.
Exit 0 = defecto no reproducido; AssertionError = defecto confirmado.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))

import torch  # noqa: E402

import mztrain.elastic_shape as shape  # noqa: E402


def setup():
    torch.manual_seed(0)
    model = shape.GPT(64, 16, 24, 1, 2, shape.fact_lin(8))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 16))
    return model, opt, x


def train(model, opt, x, n):
    for _ in range(n):
        opt.zero_grad()
        model(x, x)[1].backward()
        opt.step()


def run(decay_between_growths):
    model, opt, x = setup()
    w1 = shape.LrWarmup(opt, steps=5)
    for _ in range(5):
        train(model, opt, x, 1)
        w1.step()
    assert w1.done and math.isclose(opt.param_groups[0]["lr"], 1e-3)
    if decay_between_growths:
        # scheduler externo / step decay escribe g["lr"] entre growths
        for g in opt.param_groups:
            g["lr"] = 1e-4
        train(model, opt, x, 3)
    pre_growth_lr = opt.param_groups[0]["lr"]
    new, report = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=36, noise_scale=0.0))
    assert report["accepted"]
    w2 = shape.LrWarmup(new, steps=5)
    for _ in range(5):
        train(model, new, x, 1)
        w2.step()
    assert w2.done
    return pre_growth_lr, new.param_groups[0]["lr"], new.param_groups[0].get("_mzshape_base_lr")


# Control negativo: sin cambio de LR entre growths el warmup vuelve al LR previo.
pre, final, base = run(decay_between_growths=False)
assert math.isclose(final, pre), f"control roto: {final} != {pre}"
print(f"[control OK] sin decaimiento: lr previo={pre:g} lr tras warmup={final:g}")

# Defecto: con decaimiento entre growths el warmup rampa a la base caduca.
pre, final, base = run(decay_between_growths=True)
print(f"[defecto?] lr previo={pre:g} lr tras warmup={final:g} _mzshape_base_lr={base}")
assert math.isclose(final, pre, rel_tol=1e-9), (
    f"PER-LOG-001 CONFIRMADO: LrWarmup termina en lr={final:g} ({final / pre:.1f}x el LR "
    f"vigente {pre:g}); usa _mzshape_base_lr={base} caduco de un warmup ya terminado"
)
print("PER-LOG-001 no reproducido")
