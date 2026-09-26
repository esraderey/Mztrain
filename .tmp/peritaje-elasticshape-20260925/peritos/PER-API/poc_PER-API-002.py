"""PER-API-002: scale_output_rows (shape_ops.py) solo valida que `factor`
sea finito, no que sea > 0. Su unico uso documentado (SPEC #4, docstring de
la funcion) es la correccion EXACTA de escala SDPA, que siempre es
sqrt(head_dim'/head_dim) > 0. Un factor 0 o negativo produce un estado
"imposible" para ese contrato (invierte signos o anula filas de U) sin
ninguna excepcion ni aviso.

Ubicacion: src/mztrain/shape_ops.py:246-254
    if not math.isfinite(factor):
        raise ValueError("factor no finito")
    ...
    layer.U.data[idx, :] *= factor

Control negativo: factor positivo (2.0) escala sin cambiar signos.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from mztrain.layers import ZFactorizedLinear
from mztrain.shape_ops import scale_output_rows


def fresh_layer():
    torch.manual_seed(0)
    layer = ZFactorizedLinear(8, 8, rank=4, bias=True, init_method="random")
    with torch.no_grad():
        layer.U.data.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) + 1.0)
        layer.bias.data.copy_(torch.arange(8, dtype=torch.float32) + 1.0)
    return layer


# --- caso adverso: factor negativo ---------------------------------------
layer = fresh_layer()
U_before = layer.U.data.clone()
scale_output_rows(layer, rows=[0, 1, 2], factor=-2.5)
sign_flipped = (torch.sign(layer.U.data[:3]) != torch.sign(U_before[:3])).any().item()

assert not sign_flipped, (
    "DEFECTO PER-API-002: scale_output_rows acepto factor=-2.5 (negativo) sin "
    "lanzar excepcion, invirtiendo el signo de las filas U afectadas. El unico "
    "uso documentado de esta primitiva (correccion de escala SDPA) exige un "
    "factor estrictamente positivo (sqrt(...)); la funcion solo valida "
    "'finito', permitiendo un estado matematicamente imposible para su "
    "contrato declarado."
)

# --- control negativo: factor positivo se comporta como se espera --------
layer2 = fresh_layer()
U_before2 = layer2.U.data.clone()
scale_output_rows(layer2, rows=[0, 1, 2], factor=2.0)
assert torch.allclose(layer2.U.data[:3], U_before2[:3] * 2.0)
assert (torch.sign(layer2.U.data[:3]) == torch.sign(U_before2[:3])).all()

print("PER-API-002: OK (si ves esto sin AssertionError arriba, el defecto existe)")
