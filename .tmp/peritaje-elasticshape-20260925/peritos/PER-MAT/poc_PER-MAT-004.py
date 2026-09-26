"""PER-MAT-004: widen_layernorm(variance_compensation=True) no compensa eps;
el residuo NO es solo el shift de media declarado.

Derivacion (x en R^d, zero-pad a d', k = d/d'):
  mu' = k mu ; var' = k sigma^2 + k(1-k) mu^2
  LN'(old) = gamma sqrt(k) [(x-mu) + (1-k)mu] / sqrt(var' + eps)
           = gamma [(x-mu) + (1-k)mu] / sqrt(sigma^2 + (1-k)mu^2 + eps/k)
Con mu = 0 (residuo declarado nulo) queda eps/k = eps*d'/d en lugar de eps:
  error relativo = 1 - sqrt((sigma^2+eps)/(sigma^2+eps*d'/d)).
Con sigma^2 = eps y d'=2d: 1 - sqrt(2/3) = 18%. Correccion exacta: eps' = eps*d/d'.
Control negativo: eps' = eps*d/d' -> exacto (1e-12) ; sigma^2 = 1 -> ~1e-5.
"""
import math
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))

import torch  # noqa: E402

from mztrain.shape_ops import widen_layernorm  # noqa: E402

torch.manual_seed(0)
d, dn, eps = 8, 16, 1e-5
ln = torch.nn.LayerNorm(d, eps=eps).double()
with torch.no_grad():
    ln.weight.uniform_(0.5, 1.5)
    ln.bias.normal_()


def centered(var):
    x = torch.randn(32, d, dtype=torch.float64)
    x = x - x.mean(-1, keepdim=True)  # mu = 0 exacto: el residuo declarado desaparece
    return x / x.std(-1, unbiased=False, keepdim=True) * math.sqrt(var)


def err(new, x):
    xp = torch.cat([x, torch.zeros(x.shape[0], dn - d, dtype=x.dtype)], -1)
    with torch.no_grad():
        y_old, y_new = ln(x), new(xp)[:, :d]
        return ((y_new - y_old).norm() / (y_old - ln.bias).norm()).item()


new = widen_layernorm(ln, dn, variance_compensation=True)
fixed = widen_layernorm(ln, dn, variance_compensation=True)
fixed.eps = eps * d / dn

e_big = err(new, centered(1.0))
e_fix = err(fixed, centered(eps))
e_bad = err(new, centered(eps))
print(f"sigma^2=1   , eps sin compensar: rel err = {e_big:.2e}")
print(f"sigma^2=eps , eps' = eps*d/d'   : rel err = {e_fix:.2e}")
print(f"sigma^2=eps , eps sin compensar: rel err = {e_bad:.4f}  (prediccion {1 - math.sqrt(2 / 3):.4f})")
assert e_fix < 1e-12  # control: la compensacion exacta de eps existe
assert e_big < 1e-4  # control: con sigma^2 >> eps el termino es despreciable
# afirmacion atacada: con mu=0 el unico residuo declarado es nulo -> deberia ser exacto
assert e_bad < 1e-6, f"RESIDUO NO DECLARADO por eps: {e_bad:.3f}"
