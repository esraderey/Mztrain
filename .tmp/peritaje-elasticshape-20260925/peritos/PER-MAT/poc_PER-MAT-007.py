"""PER-MAT-007: ZFactorizedLinear.reconstruct_weight ignora wake_gate, asi que
no devuelve la W_efectiva del forward (U diag(S*gate) V) que el propio
contrato de ruido y _gated_s definen.

layers.py:404-407  return (self.U * self.S.unsqueeze(0)) @ self.V
forward:           h = F.linear(x, V) * _gated_s()  -> y = x (U diag(S*g) V)^T
Diferencia: sum_i (1 - g_i) S_i u_i v_i^T. Consumidores: engine._defactorize_recursive
(export_full_model) y refactorize_model (tras el cual reset_after_refactorize
pone gate=1): una direccion a mitad de mini-warmup salta de g*S a S.
Control negativo: gate = 1 -> identicos.
"""
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))

import torch  # noqa: E402

from mztrain.layers import ZFactorizedLinear  # noqa: E402

torch.manual_seed(0)
x = torch.randn(5, 8, dtype=torch.float64)


def gap(gate):
    layer = ZFactorizedLinear(8, 6, rank=3, bias=False).double()
    layer.wake_gate = torch.tensor(gate, dtype=torch.float64)
    with torch.no_grad():
        y_fwd = layer(x)
        y_rec = x @ layer.reconstruct_weight().T
        return ((y_fwd - y_rec).norm() / y_fwd.norm().clamp_min(1e-300)).item()


g1 = gap([1.0, 1.0, 1.0])
g2 = gap([1.0, 0.5, 0.0])
print(f"gate=1          : rel gap forward vs reconstruct_weight = {g1:.2e}")
print(f"gate=[1,.5,0]   : rel gap forward vs reconstruct_weight = {g2:.2e}")
assert g1 < 1e-12  # control negativo
assert g2 < 1e-12, f"reconstruct_weight != W_efectiva del forward (gap {g2:.3f})"
