import sys, math, warnings, copy, importlib
PRE = sys.argv[1] == "pre"
sys.path.insert(0, "C:/Users/Raul/AppData/Local/Temp/claude/D--mztrain/0fd148d5-7bed-475f-955f-d563a3883047/scratchpad/lab/src" if PRE else "D:/mztrain/src")
import torch, torch.nn as nn, numpy as np
import mztrain.elastic_shape as shape
import mztrain.shape_ops as so
from mztrain.layers import ZFactorizedLinear
print("module:", shape.__file__)
warnings.simplefilter("ignore")

def tryit(label, f):
    try:
        r = f(); print(f"  {label}: OK {r if r is not None else ''}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {e}")

print("== (viii) tipos de dims")
ln = nn.LayerNorm(4)
tryit("LN new_dim np.int64", lambda: so.widen_layernorm(ln, np.int64(6)).normalized_shape)
tryit("LN new_dim tensor0d", lambda: so.widen_layernorm(ln, torch.tensor(6)).normalized_shape)
lay = ZFactorizedLinear(8, 8, rank=4, bias=False, init_method="random")
tryit("WFL new_in tensor0d", lambda: so.widen_factorized_linear(lay, torch.tensor(12), 8).in_features)
tryit("WFL new_in np.int32", lambda: so.widen_factorized_linear(lay, np.int32(12), 8).in_features)
tryit("WFL map np array int", lambda: so.widen_factorized_linear(lay, 12, 8, in_map=np.arange(8)).in_features)
tryit("WFL map tensor int32", lambda: so.widen_factorized_linear(lay, 12, 8, in_map=torch.arange(8, dtype=torch.int32)).in_features)
tryit("WFL map range", lambda: so.widen_factorized_linear(lay, 12, 8, in_map=range(8)).in_features)
tryit("pad_state torch.Size", lambda: so.pad_state_tensor(torch.randn(4), torch.Size([6])).shape)
tryit("pad_state np.int64 shape", lambda: so.pad_state_tensor(torch.randn(4), (np.int64(6),)).shape)
m = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4))
tryit("widen_gpt new_d np.int64", lambda: shape.widen_gpt(copy.deepcopy(m), np.int64(12), noise_scale=0.0) and None)
tryit("widen_gpt noise np.float32", lambda: shape.widen_gpt(copy.deepcopy(m), 12, noise_scale=np.float32(1e-3)) and None)
tryit("widen_gpt noise np.float64", lambda: shape.widen_gpt(copy.deepcopy(m), 12, noise_scale=np.float64(1e-3)) and None)
tryit("widen_gpt noise tensor", lambda: shape.widen_gpt(copy.deepcopy(m), 12, noise_scale=torch.tensor(1e-3)) and None)
tryit("widen_gpt noise Fraction", lambda: shape.widen_gpt(copy.deepcopy(m), 12, noise_scale=__import__('fractions').Fraction(1,1000)) and None)

print("== (v) probe_targets dtypes")
for dt in (torch.int64, torch.int32, torch.int16, torch.uint8):
    mm = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)); op = torch.optim.AdamW(mm.parameters(), lr=1e-3)
    x = torch.randint(0, 23, (2, 6))
    tryit(f"targets {dt}", lambda: shape.apply_event(mm, op, shape.GrowthEvent(1, new_d=12), probe_x=x, probe_targets=x.to(dt), max_loss_increase=10.0)[1]["accepted"])
mm = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)); op = torch.optim.AdamW(mm.parameters(), lr=1e-3)
t = x.clone(); t[0, :3] = -100
tryit("targets con -100", lambda: shape.apply_event(mm, op, shape.GrowthEvent(1, new_d=12), probe_x=x, probe_targets=t, max_loss_increase=10.0)[1]["accepted"])
if torch.cuda.is_available():
    mm = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)); op = torch.optim.AdamW(mm.parameters(), lr=1e-3)
    tryit("targets en cuda, modelo cpu", lambda: shape.apply_event(mm, op, shape.GrowthEvent(1, new_d=12), probe_x=x, probe_targets=x.cuda(), max_loss_increase=10.0)[1]["accepted"])
    mm = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)).cuda(); op = torch.optim.AdamW(mm.parameters(), lr=1e-3)
    tryit("modelo cuda, targets cpu", lambda: shape.apply_event(mm, op, shape.GrowthEvent(1, new_d=12), probe_x=x, probe_targets=x, max_loss_increase=10.0)[1]["accepted"])

print("== (vi) NoiseError: ns diminuto en fp32/bf16")
for dt in (torch.float32, torch.bfloat16, torch.float16):
    for ns in (1e-10, 1e-20, 1e-22, 1e-24, 1e-30):
        torch.manual_seed(0)
        l = ZFactorizedLinear(16, 16, rank=4, bias=False, init_method="random")
        l = l.to(dt)
        if dt != torch.float32: l.S.data = l.S.data.float()
        def f():
            n = l.U.detach()
            new = so.widen_factorized_linear(l, 16, 32, noise_scale=ns, generator=torch.Generator().manual_seed(1))
            nz = new.U.detach()[16:]
            return f"nonzero={int((nz!=0).sum())}/{nz.numel()}"
        tryit(f"{dt} ns={ns:g}", f)

print("== (vi) colapso PARCIAL fp16 (gauge 1e-3)")
for ns in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
    torch.manual_seed(1)
    layer = ZFactorizedLinear(16, 16, rank=4, bias=False, init_method="random")
    with torch.no_grad():
        layer.U.normal_(); layer.V.normal_(); layer.S.fill_(1.0)
        layer.U.mul_(1e-3); layer.S.div_(1e-3)
    layer = layer.to(torch.float16); layer.S.data = layer.S.data.float()
    def g():
        new = so.widen_factorized_linear(layer, 16, 32, noise_scale=ns, generator=torch.Generator().manual_seed(3))
        W = (layer.U.double() * layer._gated_s().double()) @ layer.V.double()
        dU = new.U.detach()[16:].double()
        dW = (dU * new._gated_s().double()) @ new.V.double()[:, :16]
        return f"realized/ns={(dW.norm()/W.norm()).item()/ns:.3f} zeros={(new.U.detach()[16:]==0).float().mean().item():.2f}"
    tryit(f"fp16 ns={ns:g}", g)
