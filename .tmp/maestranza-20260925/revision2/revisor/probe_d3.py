import sys, math, importlib.util, warnings, logging
sys.path.insert(0, "D:/mztrain/src")
logging.disable(logging.CRITICAL); warnings.simplefilter("ignore")
import torch
from mztrain import elastic_shape as shape
from torch.optim.lr_scheduler import _update_param_group_val

def mk(lr=1e-3):
    torch.manual_seed(123)
    m = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4))
    return m, torch.optim.AdamW(m.parameters(), lr=lr)
def fin(opt, steps=2):
    w = shape.LrWarmup(opt, steps)
    for _ in range(steps): w.step()
    return opt.param_groups[0]["lr"], sorted(k for k in opt.param_groups[0] if k.startswith("_mz"))
out = {}
# a) warmup active + growth, no external
m, o = mk(); w = shape.LrWarmup(o, 10); [w.step() for _ in range(3)]
n, r = shape.apply_event(m, o, shape.GrowthEvent(1, new_d=12)); out["a_active_growth"] = fin(n)
# b) warmup finished -> keys
m, o = mk(); out["b_finished"] = fin(o)
# e) apply_event lr override with active warmup
m, o = mk(); w = shape.LrWarmup(o, 10); [w.step() for _ in range(3)]
n, r = shape.apply_event(m, o, shape.GrowthEvent(1, new_d=12), lr=5e-4); out["e_override"] = (n.param_groups[0]["lr"], fin(n))
# e2) override equals the last warmup-written lr (coincidence)
m, o = mk(); w = shape.LrWarmup(o, 10); [w.step() for _ in range(3)]
cur = o.param_groups[0]["lr"]
n, r = shape.apply_event(m, o, shape.GrowthEvent(1, new_d=12), lr=cur); out["e2_override_eq_written"] = (cur, fin(n))
# f) two consecutive LrWarmup same opt no external change
m, o = mk(); w = shape.LrWarmup(o, 10); [w.step() for _ in range(3)]; out["f_two_consecutive"] = fin(o)
# g) external equals last written
m, o = mk(); w = shape.LrWarmup(o, 10); [w.step() for _ in range(3)]
o.param_groups[0]["lr"] = float(o.param_groups[0]["lr"]); out["g_coincidence"] = fin(o)
# h) legacy checkpoint (HEAD code): only _mzshape_base_lr mid-warmup
m, o = mk(); w = shape.LrWarmup(o, 10); [w.step() for _ in range(3)]
o.param_groups[0].pop("_mzshape_warmup_lr"); out["h_legacy_mid_warmup"] = fin(o)
# i) tensor lr + torch scheduler-style in-place write
torch.manual_seed(0)
m = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4))
o = torch.optim.AdamW(m.parameters(), lr=torch.tensor(1e-3))
w = shape.LrWarmup(o, 10); [w.step() for _ in range(3)]
_update_param_group_val(o.param_groups[0], "lr", 1e-5)   # external in-place write (LRScheduler path)
w2 = shape.LrWarmup(o, 2); w2.step(); w2.step()
out["i_tensor_lr_inplace_external"] = float(o.param_groups[0]["lr"])
# j) multiple param groups, one touched externally
torch.manual_seed(0)
m = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4))
ps = list(m.parameters())
o = torch.optim.AdamW([{"params": ps[:3], "lr": 1e-3}, {"params": ps[3:], "lr": 2e-3}])
w = shape.LrWarmup(o, 10); [w.step() for _ in range(3)]
o.param_groups[1]["lr"] = 7e-6
w2 = shape.LrWarmup(o, 2); w2.step(); w2.step()
out["j_groups"] = [g["lr"] for g in o.param_groups]
# k) abandoned warmup never followed by another: keys linger
m, o = mk(); w = shape.LrWarmup(o, 10); w.step()
out["k_lingering_keys"] = sorted(k for k in o.param_groups[0] if k.startswith("_mz"))
for k, v in out.items(): print(k, v)
