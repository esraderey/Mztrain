"""PER-API-001: _validate_map (shape_ops.py) acepta mapas de indices con
dtype float y los trunca silenciosamente en vez de rechazarlos.

Ubicacion: src/mztrain/shape_ops.py:29
    m = torch.as_tensor(m, dtype=torch.long)
No hay chequeo de que `m` sea ya un tensor/lista de enteros: cualquier
float se trunca hacia cero (comportamiento de cast de PyTorch), sin
excepcion, antes de correr las validaciones de duplicados/rango.

Se ataca via widen_layernorm (llama a _validate_map internamente), que
es parte de la API publica exportada en __init__.py.

Control negativo: el mismo dim_map como enteros funciona normalmente.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn as nn

from mztrain.shape_ops import widen_layernorm


def build_ln():
    ln = nn.LayerNorm(4)
    with torch.no_grad():
        ln.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        ln.bias.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    return ln


# --- caso adverso: dim_map con valores NO enteros -----------------------
ln = build_ln()
dim_map_float = torch.tensor([0.1, 1.1, 2.1, 3.1])  # claramente no son indices validos

raised = False
try:
    widen_layernorm(ln, new_dim=6, dim_map=dim_map_float)
except (ValueError, TypeError):
    raised = True

assert raised, (
    "DEFECTO PER-API-001: widen_layernorm(dim_map=[0.1,1.1,2.1,3.1]) (floats, no "
    "enteros) fue ACEPTADO sin excepcion. _validate_map hace "
    "torch.as_tensor(m, dtype=torch.long) que trunca los floats hacia cero en vez "
    "de exigir un dtype entero, produciendo un mapa '[0,1,2,3]' distinto del que "
    "el caller escribio, sin ningun aviso."
)

# --- control negativo: mapa entero real, debe funcionar sin problema -----
ln2 = build_ln()
dim_map_int = torch.tensor([0, 1, 2, 3], dtype=torch.long)
new_ln = widen_layernorm(ln2, new_dim=6, dim_map=dim_map_int)
assert torch.equal(new_ln.weight.data[:4], ln2.weight.data), "control negativo roto"
assert torch.equal(new_ln.bias.data[:4], ln2.bias.data), "control negativo roto"

print("PER-API-001: OK (si ves esto sin AssertionError arriba, el defecto existe)")
