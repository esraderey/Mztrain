import sys, warnings, logging, copy
sys.path.insert(0, "D:/mztrain/src")
logging.disable(logging.CRITICAL); warnings.simplefilter("ignore")
import torch, torch.nn as nn
from mztrain.shape_ops import dense_to_factorized
from mztrain import elastic_shape as shape
out = {}
torch.manual_seed(0)
for dt in (torch.bfloat16, torch.float16):
    lin = nn.Linear(48, 144, bias=True).to(dt)
    fac = dense_to_factorized(lin)
    x = torch.randn(9, 48).to(dt)
    with torch.no_grad():
        y, yr = fac(x), lin(x)
    out[f"{dt}_dtypes"] = (str(fac.U.dtype), str(fac.S.dtype), str(y.dtype))
    out[f"{dt}_fwd_rel"] = float((y.float() - yr.float()).norm() / yr.float().norm())
    Wf = (fac.U.float() * fac.S.float()) @ fac.V.float()
    out[f"{dt}_recerr_reported_vs_fp32calc"] = (fac._reconstruction_error, float((Wf - lin.weight.float()).norm() / lin.weight.float().norm()))
if torch.cuda.is_available():
    lin = nn.Linear(48, 144, bias=True).cuda().to(torch.bfloat16)
    fac = dense_to_factorized(lin)
    x = torch.randn(9, 48, device="cuda").to(torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y = fac(x); yr = lin(x)
    out["cuda_autocast_bf16"] = (str(y.dtype), float((y.float() - yr.float()).norm() / yr.float().norm()))
# full bf16 GPT pipeline: factorize, widen, deepen, AdamW step, no autocast
torch.manual_seed(0)
m = shape.GPT(23, 6, 8, 1, 2, shape.dense_lin).to(torch.bfloat16)
o = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randint(0, 23, (2, 6))
try:
    o, r = shape.apply_event(m, o, shape.GrowthEvent(1, factorize=True, new_d=16, add_layers=1, noise_scale=1e-2))
    out["bf16_pipeline_accepted"] = r["accepted"]
    out["bf16_S_dtypes"] = sorted({str(b.qkv.S.dtype) for b in m.blocks})
    o.zero_grad(); loss = m(x, x)[1]; loss.backward(); o.step()
    out["bf16_train_step_loss"] = float(loss)
except Exception as e:
    out["bf16_pipeline"] = f"ERR {type(e).__name__}: {e}"
# D4 checks: positional constructors keep meaning; default eps
g = shape.GPT(23, 6, 8, 1, 2, shape.dense_lin)
out["default_eps"] = (g.ln_eps, g.lnf.eps, g.blocks[0].ln1.eps)
# widen twice -> ln_eps compounding, deepen after
torch.manual_seed(0)
g = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)).double()
shape.widen_gpt(g, 16, noise_scale=0.0); g._mzshape_pending_recs = False
shape.widen_gpt(g, 32, noise_scale=0.0); g._mzshape_pending_recs = False
shape.deepen_gpt(g, 1)
eps = {g.ln_eps, g.lnf.eps} | {b.ln1.eps for b in g.blocks} | {b.ln2.eps for b in g.blocks}
out["widen_twice_eps_set"] = eps
rb = shape.GPT(23, 6, 32, 2, 2, shape.fact_lin(4), ln_eps=g.ln_eps).double(); rb.load_state_dict(g.state_dict())
xi = torch.randint(0, 23, (2, 6))
with torch.no_grad(): out["rebuild_maxdiff"] = float((rb(xi)[0] - g(xi)[0]).abs().max())
# apply_event rejected (guard) restores ln_eps
torch.manual_seed(0)
g = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4))
o = torch.optim.AdamW(g.parameters(), lr=1e-3)
o2, r = shape.apply_event(g, o, shape.GrowthEvent(1, new_d=16, noise_scale=0.5), probe_x=xi, max_kl=0.0)
out["rejected_event"] = (r["accepted"], g.d, g.ln_eps, g.lnf.eps)
for k, v in out.items(): print(k, v)
