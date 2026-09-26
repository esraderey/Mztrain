"""PER-API-009: widen_layernorm no valida que `new_dim` sea un entero. Un
`new_dim` float (p.ej. 6.5) pasa la comprobacion `new_dim < old_dim` (la
comparacion numerica funciona con floats), pero revienta mas abajo al
construir `nn.LayerNorm(new_dim, ...)`, con un TypeError interno de
PyTorch ("'float' object is not iterable") que no menciona en absoluto
que el problema es el tipo de `new_dim`.

Ubicacion: src/mztrain/shape_ops.py:200-226
    if new_dim < old_dim:
        raise ValueError("v1 solo crece: new_dim >= dim actual")
    dmap = _validate_map(dim_map, old_dim, new_dim, "dim_map")
    ...
    new = nn.LayerNorm(new_dim, eps=ln.eps, elementwise_affine=ln.elementwise_affine)

Ninguna mutacion ocurre antes del crash (new es un objeto nuevo, `ln`
original no se toca), asi que no hay corrupcion de estado -- es un
hallazgo de observabilidad (mensaje de error que no describe la causa
real), no de integridad.

Control negativo: new_dim entero funciona normalmente.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch.nn as nn

from mztrain.shape_ops import widen_layernorm

ln = nn.LayerNorm(4)

raised_clear_valueerror = False
raised_typeerror = False
try:
    widen_layernorm(ln, new_dim=6.5)
except ValueError:
    raised_clear_valueerror = True
except TypeError:
    raised_typeerror = True

assert raised_clear_valueerror, (
    "DEFECTO PER-API-009: widen_layernorm(new_dim=6.5) (float, no entero) no "
    f"produce el ValueError esperado de validacion (raised_typeerror="
    f"{raised_typeerror}); revienta con un TypeError interno de nn.LayerNorm "
    "que no dice nada sobre el tipo invalido de new_dim."
)

# --- control negativo: new_dim entero funciona ----------------------------
ln2 = nn.LayerNorm(4)
new_ln = widen_layernorm(ln2, new_dim=6)
assert new_ln.normalized_shape == (6,)

print("PER-API-009: OK (si ves esto sin AssertionError arriba, el defecto existe)")
