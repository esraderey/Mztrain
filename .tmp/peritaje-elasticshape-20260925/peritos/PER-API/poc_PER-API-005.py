"""PER-API-005: pad_adam_entry no valida explicitamente la presencia de la
clave 'step'; la accede con entry["step"] (linea marcada como "requerido")
pero no envuelve el acceso en una comprobacion con mensaje propio. Un
`entry` sin 'step' produce un KeyError('step') crudo, no el tipo de
ValueError descriptivo que el resto del modulo usa consistentemente para
precondiciones violadas (p.ej. "shapes incompatibles en los momentos
AdamW").

Ubicacion: src/mztrain/shape_ops.py:341
    entry["step"]  # requerido; el contador se clona por valor

Control negativo: entry con 'step' presente funciona normalmente.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from mztrain.shape_ops import pad_adam_entry

entry_sin_step = {
    "exp_avg": torch.zeros(4),
    "exp_avg_sq": torch.zeros(4),
}

raised_clear_valueerror = False
raised_keyerror = False
try:
    pad_adam_entry(entry_sin_step, (6,))
except ValueError:
    raised_clear_valueerror = True
except KeyError:
    raised_keyerror = True

assert raised_clear_valueerror, (
    "DEFECTO PER-API-005: pad_adam_entry({'exp_avg':..., 'exp_avg_sq':...}) sin "
    f"'step' no produce un ValueError explicativo (raised_keyerror={raised_keyerror}); "
    "falla con un KeyError('step') que no documenta la precondicion violada, a "
    "diferencia de otras validaciones del mismo modulo (p.ej. 'shapes "
    "incompatibles en los momentos AdamW')."
)

# --- control negativo: con 'step' presente, funciona normalmente ---------
entry_ok = {
    "exp_avg": torch.zeros(4),
    "exp_avg_sq": torch.zeros(4),
    "step": torch.tensor(5.0),
}
out = pad_adam_entry(entry_ok, (6,))
assert out["exp_avg"].shape == (6,)
assert float(out["step"]) == 5.0

print("PER-API-005: OK (si ves esto sin AssertionError arriba, el defecto existe)")
