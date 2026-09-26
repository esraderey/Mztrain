"""PER-API-004: pad_state_tensor no valida el TIPO de `new_shape`. Si se le
pasa un tensor en vez de una tupla/lista de enteros de Python, las
comprobaciones de precondicion (ndim, crecimiento monotono) pasan igual
(comparaciones elemento a elemento sobre tensores 0-D funcionan), pero la
funcion revienta mas abajo con un TypeError interno de `torch.zeros`, no
con el ValueError de validacion que el resto de la funcion usa
consistentemente para entradas invalidas.

Ubicacion: src/mztrain/shape_ops.py:298-322
    for k, (old_d, new_d) in enumerate(zip(t.shape, new_shape, strict=True)):
        if new_d < old_d:
            raise ValueError(...)
    out = torch.zeros(*new_shape, device=t.device, dtype=t.dtype)  # <-- revienta aqui

Control negativo: new_shape como tupla de ints funciona normalmente.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from mztrain.shape_ops import pad_state_tensor

t = torch.randn(4)
new_shape_tensor = torch.tensor([6])  # 1 sola dimension, coherente con t.ndim == 1

raised_value_error = False
raised_type_error = False
try:
    pad_state_tensor(t, new_shape_tensor)
except ValueError:
    raised_value_error = True
except TypeError:
    raised_type_error = True

assert raised_value_error, (
    "DEFECTO PER-API-004: pad_state_tensor(new_shape=torch.tensor([6])) no produce "
    f"el ValueError de validacion esperado (raised_type_error={raised_type_error}); "
    "el codigo no rechaza explicitamente un new_shape que no sea una secuencia de "
    "enteros de Python, y falla mas tarde con un TypeError interno de torch.zeros "
    "que no explica la causa real (tipo invalido de new_shape)."
)

# --- control negativo: tupla de enteros funciona correctamente -----------
out = pad_state_tensor(t, (6,))
assert out.shape == (6,)
assert torch.equal(out[:4], t)
assert torch.equal(out[4:], torch.zeros(2))

print("PER-API-004: OK (si ves esto sin AssertionError arriba, el defecto existe)")
