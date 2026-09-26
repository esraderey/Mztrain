"""PER-API-008: _require_adamw (elastic_shape.py) usa `type(opt) is not
torch.optim.AdamW` (chequeo de tipo EXACTO), no `isinstance`. Cualquier
subclase de AdamW -- incluso una trivial, sin logica adicional -- es
rechazada con el mismo TypeError generico que un optimizer totalmente
distinto. docs/ELASTICSHAPE-SAFETY.md dice "Se admite torch.optim.AdamW
estandar", sin aclarar que "estandar" excluye tambien a las subclases.

Ubicacion: src/mztrain/elastic_shape.py:346-348
    def _require_adamw(opt):
        if type(opt) is not torch.optim.AdamW:
            raise TypeError(...)

Esto es una discrepancia de fidelidad documental (o, como minimo, una
ambiguedad no aclarada), no necesariamente un bug de logica: puede ser
deliberado (evitar comportamientos no verificados de una subclase
desconocida). Se reporta como hallazgo de observabilidad/documentacion.

Control negativo: torch.optim.AdamW puro es aceptado.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from mztrain.elastic_shape import GPT, GrowthEvent, apply_event, dense_lin

VOCAB, SEQ, D, HEADS = 64, 16, 24, 2


class MyAdamW(torch.optim.AdamW):
    """Subclase trivial: no anade ni cambia ningun comportamiento."""

    pass


model = GPT(VOCAB, SEQ, D, layers=1, heads=HEADS, lin=dense_lin)
opt = MyAdamW(model.parameters(), lr=1e-3)

raised_typeerror = False
try:
    ev = GrowthEvent(step=0, factorize=True, new_d=48)
    apply_event(model, opt, ev)
except TypeError:
    raised_typeerror = True

assert not raised_typeerror, (
    "DEFECTO/fidelidad PER-API-008: una subclase TRIVIAL de torch.optim.AdamW "
    "(sin logica adicional) fue rechazada por _require_adamw via "
    "'type(opt) is not torch.optim.AdamW' (chequeo estricto, no isinstance). "
    "SAFETY.md dice 'Se admite torch.optim.AdamW estandar' sin aclarar que las "
    "subclases -- incluso funcionalmente identicas -- quedan excluidas."
)

# --- control negativo: AdamW puro se acepta -------------------------------
model2 = GPT(VOCAB, SEQ, D, layers=1, heads=HEADS, lin=dense_lin)
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
ev2 = GrowthEvent(step=0, factorize=True, new_d=48)
_, report = apply_event(model2, opt2, ev2)
assert report["accepted"] is True

print("PER-API-008: OK (si ves esto sin AssertionError arriba, el defecto existe)")
