"""PER-MAT-001: el presupuesto funcional de ruido de qkv deja de valer tras
scale_output_rows (correccion SDPA) en widen_gpt.

Afirmacion atacada (SAFETY "Nuevo contrato del ruido"):
    ||delta_U diag(S*gate) V||_F = noise_scale * ||U diag(S*gate) V||_F
widen_gpt ruidea las filas nuevas de qkv y DESPUES multiplica TODO el bloque q
(incluidas las filas nuevas ruidosas) por c = sqrt(hd2/hd). Derivacion:
    ||dW'||^2 = c^2 a^2 + b^2  con  a^2 + b^2 = (ns ||W||)^2
    => ratio realizado = sqrt((c^2 a^2 + b^2)/(a^2 + b^2))  in (1, c]
    esperado con ruido isotropo (a^2 ~ 1/3): sqrt((c^2+2)/3)  (c=2 -> ~1.41)
Control negativo: la misma primitiva sin correccion SDPA cumple el contrato.
Ejecutar: python poc_PER-MAT-001.py  (falla la ultima asercion = hallazgo)
"""
import math
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))

import torch  # noqa: E402

from mztrain.elastic_shape import GPT, fact_lin, qkv_out_map, widen_gpt  # noqa: E402
from mztrain.shape_ops import widen_factorized_linear  # noqa: E402


def eff(layer):
    return (layer.U.detach().double() * layer._gated_s().detach().double()) @ layer.V.detach().double()


torch.manual_seed(0)
ns = 1e-2
d, new_d, heads = 8, 32, 2  # hd 4 -> 16, c = 2
c = math.sqrt((new_d // heads) / (d // heads))
model = GPT(64, 16, d, 1, heads, fact_lin(4)).double()
old_qkv = model.blocks[0].qkv
W_old = eff(old_qkv)
omap = qkv_out_map(d, new_d, heads)
fresh = torch.ones(3 * new_d, dtype=torch.bool)
fresh[omap] = False
is_q = torch.arange(3 * new_d) < new_d

# control negativo: primitiva aislada (sin correccion SDPA) -> contrato exacto
ctrl = widen_factorized_linear(
    old_qkv, new_d, 3 * new_d, in_map=torch.arange(d), out_map=omap,
    noise_scale=ns, generator=torch.Generator().manual_seed(7),
)
r_ctrl = (eff(ctrl)[fresh].norm() / W_old.norm()).item() / ns
print(f"control (primitiva sola): ratio/ns = {r_ctrl:.12f}")
assert abs(r_ctrl - 1.0) < 1e-9, "control: la primitiva aislada deberia cumplir el contrato"

widen_gpt(model, new_d, noise_scale=ns, generator=torch.Generator().manual_seed(7))
W_new = eff(model.blocks[0].qkv)
realized = (W_new[fresh].norm() / W_old.norm()).item() / ns
a = (W_new[fresh & is_q].norm() / c).item()
b = W_new[fresh & ~is_q].norm().item()
pred = math.sqrt((c * c * a * a + b * b) / (a * a + b * b))
print(f"c = {c:.4f}; fraccion q del ruido a^2/(a^2+b^2) = {a*a/(a*a+b*b):.3f}")
print(f"widen_gpt qkv: ratio realizado/ns = {realized:.6f}  (prediccion {pred:.6f}; esperado ~{math.sqrt((c*c+2)/3):.3f})")
print(f"respecto a ||W_new||: {(W_new[fresh].norm() / W_new[~fresh].norm()).item() / ns:.6f}")
assert abs(realized - pred) < 1e-9, "la derivacion c^2 a^2 + b^2 deberia ser exacta"
# afirmacion atacada: el contrato de norma se sostiene en la capa qkv resultante
assert abs(realized - 1.0) < 1e-6, f"CONTRATO ROTO: ||dW||/||W|| = {realized:.4f}*noise_scale en qkv tras la correccion SDPA"
