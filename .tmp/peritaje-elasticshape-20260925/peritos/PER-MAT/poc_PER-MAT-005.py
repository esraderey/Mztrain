"""PER-MAT-005: el estado Adam de gamma (LN) queda en una escala incoherente
tras variance_compensation, por el MISMO argumento que justifica _rescale_q_state.

Derivacion (mu=0, eps->0, k=d/d'): gamma' = sqrt(k) gamma, n' = n/sqrt(k)
  y = gamma' n' = gamma n (funcion preservada)
  dL/dgamma'_i = dL/dy_i * n'_i = (1/sqrt(k)) dL/dgamma_i = sqrt(d'/d) dL/dgamma_i
=> coherente seria m *= sqrt(d'/d), v *= d'/d (q usa m/=c, v/=c^2). Sin ello,
durante ~[10, 1000] pasos m ya refleja g' pero v sigue en g^2: el paso Adam de
gamma es sqrt(d'/d) veces el estacionario, y relativo a |gamma'| es d'/d veces
(1.5x, 2x, 4x la anchura -> 1.5x, 2x, 4x). Bias de LN: gradiente invariante.
Control negativo: gradiente de beta (ratio 1) y funcion preservada.
Segunda parte: apply_event copia el exp_avg de lnf.weight SIN reescalar,
mientras si reescala las filas q.
"""
import math
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB / "src"))

import torch  # noqa: E402

import mztrain.elastic_shape as es  # noqa: E402
from mztrain.shape_ops import widen_layernorm  # noqa: E402

torch.manual_seed(0)
d, dn = 8, 16
ln = torch.nn.LayerNorm(d, eps=1e-12).double()
with torch.no_grad():
    ln.weight.uniform_(0.5, 1.5)
    ln.bias.normal_()
x = torch.randn(64, d, dtype=torch.float64)
x = x - x.mean(-1, keepdim=True)
w_out = torch.randn(d, dtype=torch.float64)

(ln(x) * w_out).sum().backward()
g_gamma, g_beta = ln.weight.grad.clone(), ln.bias.grad.clone()

new = widen_layernorm(ln, dn, variance_compensation=True)
xp = torch.cat([x, torch.zeros(64, dn - d, dtype=x.dtype)], -1)
w_out_p = torch.cat([w_out, torch.zeros(dn - d, dtype=x.dtype)])
y_new = new(xp)
(y_new * w_out_p).sum().backward()
with torch.no_grad():
    fn_err = ((y_new[:, :d] - ln(x)).norm() / ln(x).norm()).item()
r_gamma = (new.weight.grad[:d].norm() / g_gamma.norm()).item()
r_beta = (new.bias.grad[:d].norm() / g_beta.norm()).item()
print(f"funcion preservada: rel err {fn_err:.2e}")
print(f"||dL/dgamma'|| / ||dL/dgamma|| = {r_gamma:.6f}  (sqrt(d'/d) = {math.sqrt(dn / d):.6f})")
print(f"||dL/dbeta'||  / ||dL/dbeta||  = {r_beta:.6f}")
assert fn_err < 1e-9 and abs(r_beta - 1) < 1e-9  # controles

# parte 2: el optimizer migrado no reescala gamma (si reescala q)
torch.manual_seed(123)
model = es.GPT(23, 6, 8, 1, 2, es.fact_lin(4))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
xi = torch.randint(0, 23, (2, 6))
model(xi, xi)[1].backward()
opt.step()
old_m = opt.state[model.lnf.weight]["exp_avg"].clone()
new_opt, _ = es.apply_event(model, opt, es.GrowthEvent(1, new_d=16, noise_scale=0))
new_m = new_opt.state[model.lnf.weight]["exp_avg"][:8]
print(f"exp_avg(lnf.weight) viejo vs migrado: max|diff| = {(new_m - old_m).abs().max().item():.2e} (coherente seria x{math.sqrt(2):.3f})")

# afirmacion atacada (implicita en la politica de migracion): el gradiente futuro
# de gamma conserva la escala del estado migrado sin reescalar
assert abs(r_gamma - 1) < 1e-6, f"ESCALA INCOHERENTE: el gradiente de gamma escala x{r_gamma:.4f} y m/v no se reescalan"
