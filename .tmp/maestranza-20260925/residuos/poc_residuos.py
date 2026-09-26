"""PoC del director para los residuos del peritaje (2026-09-26). Cada bloque imprima REPRODUCE/NO."""
import sys, math, warnings
sys.path.insert(0, "D:/mztrain/src")
import torch, torch.nn as nn
from mztrain.layers import ZFactorizedLinear
from mztrain.elastic_shape import GPT, fact_lin, GrowthEvent, apply_event, LrWarmup, widen_gpt, migrate_optimizer
from mztrain.shape_ops import dense_to_factorized
torch.manual_seed(0)
out = []
def rep(i, cond, msg): out.append((i, cond)); print(("REPRODUCE" if cond else "NO       "), i, msg)

# R1: export a denso (engine.py:1557 / zcodebert.py:518 usan reconstruct_weight) con gates parciales
lay = ZFactorizedLinear(8, 6, rank=3, bias=False).double()
lay.wake_gate.copy_(torch.tensor([1.0, 0.5, 0.0]))
x = torch.randn(4, 8, dtype=torch.float64)
lin = nn.Linear(8, 6, bias=False).double(); lin.weight.data = lay.reconstruct_weight()   # mismo codigo que el export
gap = float((lin(x) - lay(x)).norm() / lay(x).norm())
rep("R1 export!=forward con wake_gate parcial", gap > 1e-6, f"gap rel={gap:.3f}")

# R2: grow_rank(preserve_weights=False) re-SVD y reaplica gates viejos a la base nueva
lay2 = ZFactorizedLinear(8, 6, rank=3, bias=False)
lay2.wake_gate.copy_(torch.tensor([1.0, 0.5, 0.0]))
lay2.grow_rank(5, preserve_weights=False)
rep("R2 gates viejos sobre base SVD nueva tras grow_rank(preserve=False)", not bool((lay2.wake_gate == 1).all()), f"wake_gate={lay2.wake_gate.tolist()}")

# R3: warmup abandonado + cambio externo de LR -> el siguiente warmup vuelve al pico viejo
m = GPT(64, 16, 24, 1, 2, fact_lin(12)); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
w = LrWarmup(opt, steps=10); [w.step() for _ in range(3)]           # abandonado a mitad
for g in opt.param_groups: g["lr"] = 1e-5                            # decaimiento externo
x = torch.randint(0, 64, (2, 16))
opt2, r = apply_event(m, opt, GrowthEvent(1, new_d=48), probe_x=x)
w2 = LrWarmup(opt2, steps=2); w2.step(); w2.step()
rep("R3 warmup abandonado: LR final != LR vigente", abs(opt2.param_groups[0]["lr"] - 1e-5) > 1e-12, f"lr final={opt2.param_groups[0]['lr']:.2e} (vigente 1e-5)")
# R3b: reanudacion desde checkpoint a mitad de warmup (claves en param_groups serializadas)
m = GPT(64, 16, 24, 1, 2, fact_lin(12)); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
w = LrWarmup(opt, steps=10); [w.step() for _ in range(3)]
sd = opt.state_dict(); opt_b = torch.optim.AdamW(m.parameters(), lr=1e-3); opt_b.load_state_dict(sd)
for g in opt_b.param_groups: g["lr"] = 1e-5
w3 = LrWarmup(opt_b, steps=2); w3.step(); w3.step()
rep("R3b tras load_state_dict + LR externo: warmup vuelve a la base vieja", abs(opt_b.param_groups[0]["lr"] - 1e-5) > 1e-12, f"lr final={opt_b.param_groups[0]['lr']:.2e}")

# R4: eps reescalado no viaja en state_dict -> reconstruir desde config da otra funcion
torch.manual_seed(1)
m = GPT(64, 16, 8, 1, 2, fact_lin(4)).double()
widen_gpt(m, 16, noise_scale=0.0); widen_gpt.__wrapped__  # noqa: B018 (solo comprobar que existe)
m._mzshape_pending_recs = False
m2 = GPT(64, 16, 16, 1, 2, fact_lin(4), ln_eps=getattr(m, "ln_eps", 1e-5)).double(); m2.load_state_dict(m.state_dict())
with torch.no_grad():
    d = float((m(x)[0] - m2(x)[0]).norm() / m(x)[0].norm())
rep("R4 eps de LN no reconstruible desde la configuracion del GPT", m.lnf.eps != m2.lnf.eps or d > 1e-12, f"eps crecido={m.lnf.eps:.2e} reconstruido={m2.lnf.eps:.2e} drift logits={d:.2e}")

# R5: dense_to_factorized bf16 -> S bf16 (y forward mixto imposible sin autocast)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    f = dense_to_factorized(nn.Linear(16, 32, bias=False).to(torch.bfloat16))
rep("R5 S en bf16 tras dense_to_factorized bf16", f.S.dtype == torch.bfloat16, f"S.dtype={f.S.dtype}")
lay3 = ZFactorizedLinear(8, 6, rank=4, bias=False).to(torch.bfloat16); lay3.S.data = lay3.S.data.float()
try: lay3(torch.randn(2, 8).to(torch.bfloat16)); mixed_ok = True
except RuntimeError: mixed_ok = False
rep("R5b forward U/V bf16 + S fp32 sin autocast falla", not mixed_ok, "")
print("\n", sum(c for _, c in out), "/", len(out), "reproducen")
