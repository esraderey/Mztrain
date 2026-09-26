"""PER-MAT-002: en fp16 la calibracion funcional depende del gauge de los
factores (underflow a subnormales / cero) y NO cumple "tolerancia del dtype".

Afirmaciones atacadas (shape_ops._functional_noise docstring):
  "La magnitud funcional no depende de std(U) ni de su gauge"
  "Se calcula en alta precision; el resultado tiene tolerancia del dtype."
Derivacion: el ruido final por entrada es ~ noise_scale * std(U) * sqrt(out/count).
Con gauge U*g, S/g (mismo W) y fp16, para g*noise_scale < ~6e-8 (subnormal
minimo fp16) las entradas se cuantizan en pasos de 5.96e-8 o se anulan: el
cociente realizado ||dW||/||W|| se aparta de noise_scale mucho mas que
eps_fp16 = 2^-10 ~ 9.8e-4. Solo se comprueba isfinite, no underflow.
Control negativo: mismo experimento en fp32 (y fp16 con gauge 1) cumple.
"""
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))

import torch  # noqa: E402

from mztrain.layers import ZFactorizedLinear  # noqa: E402
from mztrain.shape_ops import widen_factorized_linear  # noqa: E402


def build(dtype, gauge):
    torch.manual_seed(1)
    layer = ZFactorizedLinear(16, 16, rank=4, bias=False, init_method="random")
    with torch.no_grad():
        layer.U.normal_()
        layer.V.normal_()
        layer.S.fill_(1.0)
        layer.U.mul_(gauge)  # mismo W: U*g, S/g
        layer.S.div_(gauge)
    layer = layer.to(dtype)
    layer.S.data = layer.S.data.float()  # regla del proyecto: S fp32
    return layer


def ratio(layer, ns):
    W = (layer.U.double() * layer._gated_s().double()) @ layer.V.double()
    new = widen_factorized_linear(layer, 16, 32, noise_scale=ns, generator=torch.Generator().manual_seed(3))
    Wn = (new.U.double() * new._gated_s().double()) @ new.V.double()
    zeros = (new.U[16:] == 0).float().mean().item()
    return (Wn[16:].norm() / W.norm()).item() / ns, zeros


TOL_FP16 = 5 * 2.0 ** -10  # tolerancia generosa "del dtype"
sweep = [1e-3, 1e-4, 4e-5, 1e-5]

for ns in sweep:  # control negativo 1: fp32, gauge extremo -> exacto
    r, _ = ratio(build(torch.float32, 1e-3), ns)
    print(f"fp32 gauge=1e-3 ns={ns:.0e}: ratio={r:.8f}")
    assert abs(r - 1) < 1e-5

r, _ = ratio(build(torch.float16, 1.0), 1e-3)  # control negativo 2: fp16 sin gauge
print(f"fp16 gauge=1    ns=1e-03: ratio={r:.6f}")
assert abs(r - 1) < TOL_FP16

bad = []
for ns in sweep:
    r, z = ratio(build(torch.float16, 1e-3), ns)
    print(f"fp16 gauge=1e-3 ns={ns:.0e}: ratio={r:.6f}  fraccion de ruido = 0: {z:.2f}")
    if abs(r - 1) >= TOL_FP16:
        bad.append((ns, r, z))
assert not bad, f"CALIBRACION ROTA en fp16 (dependiente del gauge): {bad}"
