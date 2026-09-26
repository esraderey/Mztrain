"""PER-MAT-006: en la ruta de parametros bf16 los logits de la sonda YA estan
redondeados a bf16; convertirlos a fp64 no evita que el redondeo domine KL.

Afirmacion atacada (elastic_shape._probe_metrics):
  "# FP64 evita que redondeos bf16/fp32 dominen KL o diferencias de loss."
Derivacion: logits en [32, 64) -> ulp bf16 h = 2^(5-7) = 0.25. Para una deriva
real |delta| << h, Q(a+delta)-Q(a) salta h con probabilidad ~|delta|/h, luego
E[delta_q^2] ~ |delta| h >> delta^2: KL_medido/KL_real ~ h/|delta| (x100-x1000).
Control negativo: los mismos logits en fp32 (ulp ~3.8e-6) reproducen el KL real.
"""
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))

import torch  # noqa: E402

import mztrain.elastic_shape as es  # noqa: E402

torch.manual_seed(0)
before = 40.0 + 3.0 * torch.randn(8, 16, 64, dtype=torch.float64)  # offset comun: softmax-invariante
after = before + 1e-3 * torch.randn_like(before)  # deriva real pequena
targets = torch.randint(0, 64, (8, 16))

true = es._probe_metrics(before, after, targets)
m32 = es._probe_metrics(before.float(), after.float(), targets)
mbf = es._probe_metrics(before.bfloat16(), after.bfloat16(), targets)
for name, m in (("fp64", true), ("fp32", m32), ("bf16", mbf)):
    print(f"{name}: kl={m['kl_div']:.3e}  loss_delta={m['loss_delta']:+.3e}")

assert abs(m32["kl_div"] / true["kl_div"] - 1) < 0.05  # control negativo
# afirmacion atacada: el computo fp64 impide que el redondeo bf16 domine KL
assert mbf["kl_div"] < 10 * true["kl_div"], (
    f"REDONDEO DOMINA: KL bf16 = {mbf['kl_div']:.2e} vs real {true['kl_div']:.2e} "
    f"(x{mbf['kl_div'] / true['kl_div']:.0f})"
)
