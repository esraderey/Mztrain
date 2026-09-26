import sys, math, warnings, copy
sys.path.insert(0, "D:/mztrain/src")
import torch, torch.nn as nn
import mztrain.elastic_shape as shape
from mztrain.shape_ops import NoiseError, widen_factorized_linear, widen_layernorm
from mztrain.layers import ZFactorizedLinear

def mo(seed=0, d=8, bias=False, r=4):
    torch.manual_seed(seed)
    if bias:
        lin = lambda i,o: ZFactorizedLinear(i,o,rank=r,bias=True,init_method="svd")
    else:
        lin = shape.fact_lin(r)
    m = shape.GPT(23, 6, d, 1, 2, lin)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randint(0, 23, (2, 6))
    opt.zero_grad(); m(x, x)[1].backward(); opt.step()
    return m, opt, x

print("== (i) warmup abandonado + cambio externo")
m, opt, x = mo()
w1 = shape.LrWarmup(opt, steps=10)
for _ in range(3): w1.step()
# abandona w1, cambio externo LR
for g in opt.param_groups: g["lr"] = 1e-5
new, rep = shape.apply_event(m, opt, shape.GrowthEvent(1, new_d=12, noise_scale=0.0))
w2 = shape.LrWarmup(new, steps=2)
w2.step(); w2.step()
print("  final lr", new.param_groups[0]["lr"], "(vigente 1e-5, base caduca 1e-3)", "base key now:", "_mzshape_base_lr" in new.param_groups[0])

print("== (i) G4-B: w1 activo, growth, w2 reusa base")
m, opt, x = mo()
w1 = shape.LrWarmup(opt, steps=10)
for _ in range(3): w1.step()
new, rep = shape.apply_event(m, opt, shape.GrowthEvent(1, new_d=12, noise_scale=0.0))
w2 = shape.LrWarmup(new, steps=4)
for _ in range(4): w2.step()
print("  final", new.param_groups[0]["lr"], "expect 1e-3")

print("== (ii) combos override")
def groups(o): return [(g["lr"], g["weight_decay"]) for g in o.param_groups]
for label, dense, ev in [
    ("factorize n_conv>0 sin new_d", True, shape.GrowthEvent(1, factorize=True)),
    ("factorize n_conv>0 new_d==d", True, shape.GrowthEvent(1, factorize=True, new_d=8)),
    ("factorize redundante", False, shape.GrowthEvent(1, factorize=True)),
    ("new_d==d", False, shape.GrowthEvent(1, new_d=8)),
    ("vacio", False, shape.GrowthEvent(1)),
    ("add_layers", False, shape.GrowthEvent(1, add_layers=1)),
    ("fact+widen+deepen", True, shape.GrowthEvent(1, factorize=True, new_d=12, add_layers=1)),
]:
    for kw in ({"lr":5e-4,"weight_decay":0.05}, {"weight_decay":0.05}, {"lr":5e-4}):
        torch.manual_seed(0)
        mdl = shape.GPT(23, 6, 8, 1, 2, shape.dense_lin if dense else shape.fact_lin(4))
        op = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=0.01)
        xx = torch.randint(0, 23, (2, 6)); op.zero_grad(); mdl(xx, xx)[1].backward(); op.step()
        st0 = copy.deepcopy(op.state_dict())
        nw, rp = shape.apply_event(mdl, op, ev, probe_x=xx, max_kl=10.0, **kw)
        exp = (kw.get("lr",1e-3), kw.get("weight_decay",0.01))
        ok = all(g == exp for g in groups(nw))
        # state preserved for untouched params in no-surgery cases
        same_state = len(nw.state) == len(op.state) if not dense else None
        print(f"  {label:30s} {str(kw):40s} ok={ok} newopt={nw is not op} state_n={len(nw.state)}/{len(op.state)} orig_lr={op.param_groups[0]['lr']} rep={rp['optimizer']}")

print("== (ii) override + rechazo por calidad en evento sin cirugia -> opt original")
m, opt, x = mo()
nw, rp = shape.apply_event(m, opt, shape.GrowthEvent(1, factorize=True), lr=5e-4, probe_x=x, max_kl=-0.0)
print("  accepted", rp["accepted"], "is orig", nw is opt, rp.get("kl_div"))

print("== (ii) override con warmup activo (base) sin cirugia")
m, opt, x = mo()
w = shape.LrWarmup(opt, steps=10); w.step()
nw, rp = shape.apply_event(m, opt, shape.GrowthEvent(1, new_d=8), lr=2e-4)
print("  groups", nw.param_groups[0]["lr"], nw.param_groups[0].get("_mzshape_base_lr"))

print("== (ii) migrate_optimizer(model,opt,[]) sin pendiente")
m, opt, x = mo()
o2 = shape.migrate_optimizer(m, opt, [], 5e-4, 0.05); print("  ok", o2.param_groups[0]["lr"])
