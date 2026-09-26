"""PER-MAT-003: en parametros bf16 las operaciones declaradas EXACTAS
(dense_to_factorized, correccion SDPA con c irracional) introducen error
~u_bf16 del orden de o MAYOR que el ruido deliberado (noise_scale=1e-3), y
dense_to_factorized deja S en bf16 (widen_factorized_linear si respeta S fp32).

Derivacion: redondeo bf16 (8 bits de significando) -> error relativo RMS por
entrada ~ 2^-7/sqrt(12)/1.44 ~ 1.6e-3; tres factores independientes (U, S, V)
-> ||U'S'V' - W||/||W|| ~ sqrt(3)*1.6e-3 ~ 2.7e-3 > 1e-3.
scale_output_rows con c = sqrt(1.5) (crecimiento 1.5x) redondea c*U en bf16:
error relativo ~1.6e-3 en el bloque q; con c = 2 (4x) es exacto (potencia de 2).
Control negativo: fp32 (error < 1e-5, S fp32) y c = 2.
"""
import copy
import math
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))

import torch  # noqa: E402

from mztrain.layers import ZFactorizedLinear  # noqa: E402
from mztrain.shape_ops import dense_to_factorized, scale_output_rows  # noqa: E402


def rel(fact, W):
    Wr = (fact.U.detach().double() * fact.S.detach().double()) @ fact.V.detach().double()
    return ((Wr - W.double()).norm() / W.double().norm()).item()


torch.manual_seed(0)
lin = torch.nn.Linear(48, 144, bias=False)
f32 = dense_to_factorized(copy.deepcopy(lin))
lin_bf = copy.deepcopy(lin).to(torch.bfloat16)
fbf = dense_to_factorized(lin_bf)
e32, ebf = rel(f32, lin.weight.detach()), rel(fbf, lin_bf.weight.detach())
print(f"dense_to_factorized fp32: rel={e32:.2e}  S.dtype={f32.S.dtype}")
print(f"dense_to_factorized bf16: rel={ebf:.2e}  S.dtype={fbf.S.dtype}  (_reconstruction_error={fbf._reconstruction_error:.2e})")
assert e32 < 1e-5 and f32.S.dtype == torch.float32  # control negativo

# scale_output_rows: c irracional en bf16
layer = ZFactorizedLinear(16, 48, rank=8, bias=True, init_method="svd").to(torch.bfloat16)
with torch.no_grad():
    layer.bias.normal_()
for c, label in ((2.0, "c=2 (4x, control)"), (math.sqrt(1.5), "c=sqrt(1.5) (1.5x)")):
    lay = copy.deepcopy(layer)
    ref = lay.U.detach().double()[:16] * c
    scale_output_rows(lay, torch.arange(16), c)
    err = ((lay.U.detach().double()[:16] - ref).norm() / ref.norm()).item()
    print(f"scale_output_rows {label}: rel err filas q = {err:.2e}")
    if c == 2.0:
        assert err == 0.0

bad = []
if fbf.S.dtype != torch.float32:
    bad.append(f"S en {fbf.S.dtype} (regla S fp32; widen la respeta)")
if ebf >= 1e-3:
    bad.append(f"conversion 'exacta' con error {ebf:.2e} >= noise_scale por defecto 1e-3")
assert not bad, "; ".join(bad)
