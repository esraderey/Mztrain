"""PER-API-010: GrowthEvent(noise_scale=None) hace que apply_event reviente
con un TypeError generico ("must be real number, not NoneType") en vez del
ValueError claro "noise_scale debe ser finito y >= 0" que usan todas las
demas validaciones de la funcion para el mismo campo (nan, negativo).

Ubicacion: src/mztrain/elastic_shape.py:577-578
    if not math.isfinite(ev.noise_scale) or ev.noise_scale < 0:
        raise ValueError("noise_scale debe ser finito y >= 0")
`math.isfinite(None)` lanza TypeError ANTES de llegar al `raise`
ValueError, así que un `GrowthEvent(noise_scale=None)` (valor
explicitamente listado como caso limite a probar) no obtiene el mensaje
de validacion disenado para ese mismo campo.

La falla ocurre antes de `_shape_transaction` (no hay mutacion), así que
no es un problema de integridad -- es de observabilidad: el mensaje no
identifica la causa real ("noise_scale invalido").

Control negativo: noise_scale=0.0 (valido) no lanza nada indebido.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from mztrain.elastic_shape import GPT, GrowthEvent, apply_event, dense_lin

VOCAB, SEQ, D, HEADS = 64, 16, 24, 2

model = GPT(VOCAB, SEQ, D, layers=1, heads=HEADS, lin=dense_lin)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
ev = GrowthEvent(step=0, factorize=True, new_d=48, noise_scale=None)

raised_clear_valueerror = False
raised_typeerror = False
try:
    apply_event(model, opt, ev)
except ValueError as e:
    raised_clear_valueerror = "noise_scale" in str(e)
except TypeError:
    raised_typeerror = True

assert raised_clear_valueerror, (
    "DEFECTO PER-API-010: GrowthEvent(noise_scale=None) no produce el ValueError "
    f"'noise_scale debe ser finito y >= 0' (raised_typeerror={raised_typeerror}); "
    "math.isfinite(None) revienta antes con TypeError('must be real number, not "
    "NoneType'), que no identifica el campo invalido."
)

# --- control negativo: noise_scale=0.0 es valido --------------------------
model2 = GPT(VOCAB, SEQ, D, layers=1, heads=HEADS, lin=dense_lin)
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
ev2 = GrowthEvent(step=0, factorize=True, new_d=48, noise_scale=0.0)
_, report = apply_event(model2, opt2, ev2)
assert report["accepted"] is True

print("PER-API-010: OK (si ves esto sin AssertionError arriba, el defecto existe)")
