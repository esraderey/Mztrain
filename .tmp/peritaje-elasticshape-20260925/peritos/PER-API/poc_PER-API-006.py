"""PER-API-006: LrWarmup no valida que `floor` este en [0, 1]. Su
proposito documentado es "Rampa lineal floor->1.0 ... absorbe el
transitorio de LN residual y del estado Adam aproximado" (docstring de la
clase), es decir, EMPEZAR bajo y subir a 1.0 para amortiguar un salto
brusco tras un growth. Con floor > 1 el comportamiento se INVIERTE: el LR
arranca en floor*base (un PICO por encima de la base) y desciende hacia
1.0*base, exactamente lo opuesto de un warmup protector, sin ninguna
excepcion ni aviso.

Ubicacion: src/mztrain/elastic_shape.py:650-661
    def __init__(self, opt, steps, floor=0.1):
        if steps <= 0:
            raise ValueError("steps > 0")
        ...  # floor: SIN validacion de rango
        self._apply(floor)

Control negativo: floor=0.1 (dentro de [0,1]) produce el arranque bajo
esperado.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from mztrain.elastic_shape import LrWarmup


def make_opt(lr=1e-3):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


# --- caso adverso: floor > 1 --------------------------------------------
opt = make_opt(lr=1e-3)
warmup = LrWarmup(opt, steps=10, floor=2.0)  # fuera de [0,1], no validado
base_lr = opt.param_groups[0]["_mzshape_base_lr"]
lr_at_start = opt.param_groups[0]["lr"]

assert lr_at_start <= base_lr, (
    f"DEFECTO PER-API-006: LrWarmup(steps=10, floor=2.0) no fue rechazado pese a "
    f"floor fuera de [0,1]; produce un LR inicial ({lr_at_start}) MAYOR que la "
    f"base ({base_lr}) -- un pico, no una rampa protectora -- exactamente lo "
    "opuesto del proposito documentado de 'absorber el transitorio'."
)

# --- control negativo: floor=0.1 (valido) arranca bajo, como se espera ---
opt2 = make_opt(lr=1e-3)
warmup2 = LrWarmup(opt2, steps=10, floor=0.1)
base_lr2 = opt2.param_groups[0]["_mzshape_base_lr"]
lr_at_start2 = opt2.param_groups[0]["lr"]
assert lr_at_start2 <= base_lr2
assert abs(lr_at_start2 - base_lr2 * 0.1) < 1e-12

print("PER-API-006: OK (si ves esto sin AssertionError arriba, el defecto existe)")
