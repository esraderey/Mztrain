"""PER-MAT-008: afirmaciones de SPEC-elasticshape-v1 internamente inconsistentes.

(a) SPEC §4: "por sqrt(d_h'/d_h) en un solo lado (q O k, nunca ambos: bilinealidad)".
    Por bilinealidad, escalar q y k por c^(1/2) CADA UNO da c*q.k: tambien exacto.
    Lo unico prohibido es c en ambos (c^2). "nunca ambos" es falso tal cual.
(b) SPEC §1: "y_viejo identico bit a bit (misma suma, mismos terminos)" contradice el
    criterio de la misma SPEC ("la reasociacion FP impide atol=0 — diffs ~2e-7")
    y el docstring del modulo ("salvo reasociacion FP del GEMM"). Se mide e imprime.
Control negativo: c en ambos lados rompe la identidad.
"""
import math
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mztrain.layers import ZFactorizedLinear  # noqa: E402
from mztrain.shape_ops import widen_factorized_linear  # noqa: E402

torch.manual_seed(0)
hd, hd2 = 4, 16
c = math.sqrt(hd2 / hd)
q = torch.randn(1, 2, 16, hd, dtype=torch.float64)
k = torch.randn(1, 2, 16, hd, dtype=torch.float64)
v = torch.randn(1, 2, 16, hd, dtype=torch.float64)
pad = lambda t: torch.cat([t, torch.zeros(*t.shape[:-1], hd2 - hd, dtype=t.dtype)], -1)  # noqa: E731
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
one_side = F.scaled_dot_product_attention(pad(c * q), pad(k), pad(v), is_causal=True)[..., :hd]
both_sqrt = F.scaled_dot_product_attention(pad(math.sqrt(c) * q), pad(math.sqrt(c) * k), pad(v), is_causal=True)[..., :hd]
both_c = F.scaled_dot_product_attention(pad(c * q), pad(c * k), pad(v), is_causal=True)[..., :hd]
e1, e2, e3 = ((t - ref).abs().max().item() for t in (one_side, both_sqrt, both_c))
print(f"q*c: {e1:.1e} | q*sqrt(c), k*sqrt(c): {e2:.1e} | q*c, k*c (control): {e3:.1e}")
assert e1 < 1e-12 and e3 > 1e-3  # controles

# (b) medicion (informativa) de bit-exactitud en fp32 con in grande
lay = ZFactorizedLinear(512, 512, rank=64, bias=False)
x = torch.randn(64, 512)
wide = widen_factorized_linear(lay, 768, 768)
with torch.no_grad():
    y0 = lay(x)
    y1 = wide(torch.cat([x, torch.zeros(64, 256)], -1))[:, :512]
print(f"(b) widen fp32, noise=0: max|y_viejo - y_nuevo| = {(y0 - y1).abs().max().item():.2e} "
      f"(bit a bit: {torch.equal(y0, y1)})")

# afirmacion atacada (a): escalar ambos lados no puede ser exacto
assert e2 > 1e-6, f"SPEC §4 FALSA: escalar q y k por sqrt(c) es exacto (err {e2:.1e})"
