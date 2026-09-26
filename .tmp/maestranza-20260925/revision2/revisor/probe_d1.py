import sys, importlib.util, copy, warnings
sys.path.insert(0, "D:/mztrain/src")
import torch, torch.nn as nn
from mztrain import layers as NEW
spec = importlib.util.spec_from_file_location("layers_head", "D:/mztrain/.tmp/maestranza-20260925/revision2/revisor/layers_head.py")
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)

def pair(cls_new, cls_old, *a, **k):
    torch.manual_seed(0); n = cls_new(*a, **k)
    torch.manual_seed(0); o = cls_old(*a, **k)
    o.load_state_dict(n.state_dict()); return n, o

res = {}
# 1) bitwise equality single dtype, gate=1: forward and reconstruct
for dt in (torch.float32, torch.float64, torch.bfloat16):
    for name in ("ZFactorizedLinear", "ZSparseFactorizedLinear"):
        kw = dict(rank=4, bias=True) if name == "ZFactorizedLinear" else dict(rank=4, bias=True, sparse_density=0.2)
        n, o = pair(getattr(NEW, name), getattr(OLD, name), 16, 12, **kw)
        n = n.to(dt); o = o.to(dt)
        x = torch.randn(5, 16).to(dt)
        with torch.no_grad():
            res[f"fwd_eq_{name}_{dt}"] = torch.equal(n(x), o(x))
            res[f"rec_eq_{name}_{dt}"] = torch.equal(n.reconstruct_weight(), o.reconstruct_weight())
# 2) autocast CPU bf16 with fp32 params
n, o = pair(NEW.ZFactorizedLinear, OLD.ZFactorizedLinear, 16, 12, rank=4, bias=True)
x = torch.randn(5, 16)
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    res["cpu_autocast_fp32params_eq"] = torch.equal(n(x), o(x))
# 3) autocast CUDA bf16 fp32 params and bf16 U/V + S fp32
if torch.cuda.is_available():
    nc, oc = n.cuda(), o.cuda(); xc = x.cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        res["cuda_autocast_fp32params_eq"] = torch.equal(nc(xc), oc(xc))
    nb, ob = copy.deepcopy(nc).to(torch.bfloat16), copy.deepcopy(oc).to(torch.bfloat16)
    nb.S.data = nc.S.data.float().clone(); ob.S.data = oc.S.data.float().clone()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        res["cuda_autocast_mixed_eq"] = torch.equal(nb(xc), ob(xc))
    # mixed without autocast: old errors, new works
    try:
        with torch.no_grad(): ob(xc.bfloat16()); res["old_mixed_noautocast"] = "ok"
    except Exception as e: res["old_mixed_noautocast"] = "ERR " + type(e).__name__
    with torch.no_grad(): res["new_mixed_noautocast_dtype"] = str(nb(xc.bfloat16()).dtype)
    # cpu tensor under CUDA autocast region (device mismatch of autocast)
    nbc = copy.deepcopy(nb).cpu()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        res["cpu_mixed_in_cuda_autocast"] = str(nbc(x.bfloat16()).dtype)
# 4) export to dense with mixed dtype (engine-like)
m = NEW.ZFactorizedLinear(16, 12, rank=4, bias=True).to(torch.bfloat16); m.S.data = m.S.data.float()
with torch.no_grad(): m.wake_gate.copy_(torch.tensor([1., .5, 0., 1.]))
lin = nn.Linear(16, 12, bias=True, device=m.U.device); lin.weight.data = m.reconstruct_weight(); lin.bias.data = m.bias.data.clone()
xb = torch.randn(5, 16).bfloat16()
with torch.no_grad():
    y1, y2 = lin(xb).float(), m(xb).float()
res["export_mixed_weight_dtype"] = str(lin.weight.dtype)
res["export_mixed_rel"] = float((y1 - y2).norm() / y2.norm())
# fp16 U with fp32 S
m16 = NEW.ZFactorizedLinear(16, 12, rank=4).half(); m16.S.data = m16.S.data.float()
res["rec_fp16_dtype"] = str(m16.reconstruct_weight().dtype)
# fp64 S with fp32 U
m64 = NEW.ZFactorizedLinear(16, 12, rank=4); m64.S.data = m64.S.data.double()
res["rec_u32_s64_dtype"] = str(m64.reconstruct_weight().dtype)
with torch.no_grad(): res["fwd_u32_s64_dtype"] = str(m64(torch.randn(2, 16)).dtype)
# 5) gate size mismatch guard
mg = NEW.ZFactorizedLinear(16, 12, rank=4); mg.wake_gate = torch.full((3,), 0.5)
res["rec_guard_mismatch_ungated"] = torch.equal(mg.reconstruct_weight(), (mg.U * mg.S.unsqueeze(0)) @ mg.V)
# 6) refactorize continuity with partial gates + engine-like reset
from mztrain.refactorize import refactorize_model
for cls in (NEW.ZFactorizedLinear, NEW.ZSparseFactorizedLinear):
    torch.manual_seed(1)
    kw = {} if cls is NEW.ZFactorizedLinear else dict(sparse_density=0.2)
    L = cls(16, 12, rank=4, bias=True, **kw)
    with torch.no_grad(): L.wake_gate.copy_(torch.tensor([1., .5, 0., .25]))
    xx = torch.randn(6, 16)
    with torch.no_grad(): yb = L(xx)
    opt = torch.optim.AdamW(L.parameters(), lr=1e-3)
    refactorize_model(L, opt, 4, (cls,))
    L.sleep_mask = torch.zeros(4, dtype=torch.bool); L.wake_gate = torch.ones(4)  # reset_after_refactorize
    with torch.no_grad(): ya = L(xx)
    res[f"refactorize_continuity_rel_{cls.__name__}"] = float((ya - yb).norm() / yb.norm())
    # old semantics: ungated SVD then reset -> jump
# old refactorize jump for comparison: use OLD class via same function (refactorize uses module.reconstruct_weight)
torch.manual_seed(1)
Lo = OLD.ZFactorizedLinear(16, 12, rank=4, bias=True)
with torch.no_grad(): Lo.wake_gate.copy_(torch.tensor([1., .5, 0., .25]))
xx = torch.randn(6, 16)
with torch.no_grad(): yb = Lo(xx)
refactorize_model(Lo, torch.optim.AdamW(Lo.parameters()), 4, (OLD.ZFactorizedLinear,))
Lo.sleep_mask = torch.zeros(4, dtype=torch.bool); Lo.wake_gate = torch.ones(4)
with torch.no_grad(): ya = Lo(xx)
res["refactorize_continuity_rel_OLD"] = float((ya - yb).norm() / yb.norm())
# 7) sparse grow_rank preserve False with partial gates keeps function
torch.manual_seed(2)
S = NEW.ZSparseFactorizedLinear(16, 12, rank=3, bias=False, sparse_density=0.2)
with torch.no_grad(): S.sparse_values.normal_(); S.wake_gate.copy_(torch.tensor([1., .5, 0.]))
xx = torch.randn(6, 16)
with torch.no_grad(): yb = S(xx)
S.grow_rank(5)
with torch.no_grad(): ya = S(xx)
res["sparse_grow_rel"] = float((ya - yb).norm() / yb.norm())
res["sparse_grow_gates"] = S.wake_gate.tolist()
# 8) meta device
try:
    mm = NEW.ZFactorizedLinear(16, 12, rank=4, device="meta") if False else None
except Exception as e: pass
for k, v in res.items(): print(k, v)
