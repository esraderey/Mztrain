"""PER-LOG-003: un candidato no finito producido por el ruido funcional de
widen_factorized_linear se propaga como ValueError en vez de rechazo.

Propiedad (SAFETY "Rollback y manejo de fallos": "Al exceder un limite o
encontrar un candidato no finito, el resultado contiene accepted=False,
rolled_back=True ... y el optimizer devuelto es el ORIGINAL"): con un
noise_scale finito valido cuyo ruido desborda el dtype, apply_event devuelve
(opt_original, report) con accepted=False; no lanza.
Nota: el mismo desborde en widen_embedding NO lanza (se detecta como no
finito y se rechaza); la ruta de capas factorizadas lanza antes de llegar ahi.
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
    torch.manual_seed(2)
    model = shape.GPT(64, 16, 24, 1, 2, shape.fact_lin(8))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 16))
    opt.zero_grad()
    model(x, x)[1].backward()
    opt.step()
    return model, opt


# Control negativo: ruido normal -> aceptado.
model, opt = setup()
new, report = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=36, noise_scale=1e-3))
assert report["accepted"] and new is not opt
print("[control OK] noise_scale=1e-3 aceptado")

# Defecto: noise_scale finito (pasa la validacion) pero no representable en fp32.
NOISE = 1e300
assert math.isfinite(NOISE) and NOISE >= 0
model, opt = setup()
params = list(model.parameters())
try:
    returned, report = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=36, noise_scale=NOISE))
except ValueError as error:
    restored = model.d == 24 and all(a is b for a, b in zip(params, model.parameters()))
    raise AssertionError(
        f"PER-LOG-003 CONFIRMADO: candidato no finito propago ValueError({error!r}) en vez de "
        f"accepted=False/rolled_back=True (modelo restaurado={restored})"
    )
assert returned is opt and report["accepted"] is False and report["rolled_back"] is True, report
print("PER-LOG-003 no reproducido:", report.get("rejection_reason"))
