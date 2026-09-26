import sys, math, warnings, copy
sys.path.insert(0, "D:/mztrain/src")
import torch, torch.nn as nn
import mztrain.elastic_shape as shape
import mztrain.shape_ops as so
from mztrain.layers import ZFactorizedLinear

print("== (vii) aviso con filtros por defecto: cuantas veces se ve y a donde apunta")
warnings.resetwarnings(); warnings.simplefilter("default")
import io, contextlib
shown = []
old_show = warnings.showwarning
warnings.showwarning = lambda msg, cat, fn, ln, file=None, line=None: shown.append((str(msg)[:30], fn.split("\\")[-1].split("/")[-1], ln))
torch.manual_seed(0)
for episode in range(2):
    m = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 23, (2, 6))
    recs = shape.widen_gpt(m, 12)
    for _ in range(3):
        m(x, x)[1].backward()   # entrenamiento con opt viejo
    opt = shape.migrate_optimizer(m, opt, recs)
print("  avisos mostrados en 2 episodios x 3 forwards:", shown)
warnings.showwarning = old_show

print("== (vii) apply_event (fact+widen+deepen, probe) bajo simplefilter('error')")
with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    torch.manual_seed(0)
    m = shape.GPT(23, 6, 8, 1, 2, shape.dense_lin); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 23, (2, 6))
    o, r = shape.apply_event(m, opt, shape.GrowthEvent(1, factorize=True, new_d=12, add_layers=1), probe_x=x, probe_targets=x, max_kl=10.0)
    print("  accepted", r["accepted"])
    # autocast bf16 caller context
    m = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o, r = shape.apply_event(m, opt, shape.GrowthEvent(1, new_d=12, add_layers=1), probe_x=x, max_kl=10.0)
    print("  accepted bajo autocast", r["accepted"])

print("== (iii) exactitud SDPA noise=0, bias True/False, fp64")
for bias in (False, True):
    torch.manual_seed(3)
    lin = (lambda i,o: ZFactorizedLinear(i,o,rank=6,bias=True,init_method="random")) if bias else shape.fact_lin(6)
    m = shape.GPT(31, 8, 8, 2, 2, lin).double()
    if bias:
        for p in m.parameters():
            pass
        for b in m.blocks:
            for n in ("qkv","proj","fc1","fc2"):
                with torch.no_grad(): getattr(b,n).bias.normal_()
    x = torch.randint(0, 31, (3, 8))
    # compara activacion post-atencion del bloque 0 en dims viejas (antes de LN drift): usa hook sobre proj input
    def attn_in(model):
        cap = {}
        h = model.blocks[0].proj.register_forward_hook(lambda mod, inp, out: cap.__setitem__("a", inp[0].detach()))
        with torch.no_grad(): model(x)
        h.remove(); return cap["a"]
    a0 = attn_in(m)
    # para aislar la atencion: mismo input a ln1; la deriva de LN (mu) contamina. Medimos con capa aislada:
    blk = copy.deepcopy(m.blocks[0])
    shape.widen_gpt(m, 16, noise_scale=0.0)
    blk2 = m.blocks[0]
    h = torch.randn(3, 8, 8, dtype=torch.float64)
    # entrada centrada con var cualquiera para que ln1 sea exacto
    h = h - h.mean(-1, keepdim=True)
    def attn(b, hh, D):
        B,T,_ = hh.shape
        qkv = b.qkv(b.ln1(hh)).view(B,T,3,b.heads,D//b.heads).permute(2,0,3,1,4)
        return torch.nn.functional.scaled_dot_product_attention(qkv[0],qkv[1],qkv[2],is_causal=True).transpose(1,2).reshape(B,T,D)
    hp = torch.cat([h, torch.zeros(3,8,8,dtype=h.dtype)], -1)
    with torch.no_grad():
        y0 = attn(blk, h, 8); y1 = attn(blk2, hp, 16)
    imap = shape.attn_in_map(8, 16, 2)
    print(f"  bias={bias}: max|diff| dims viejas = {(y1[..., imap]-y0).abs().max().item():.2e}; dims nuevas max = {y1[..., [i for i in range(16) if i not in imap.tolist()]].abs().max().item():.2e}")

print("== (iv) eps tras widen encadenado + deepen, y reconstruccion desde config")
torch.manual_seed(0)
m = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randint(0, 23, (2, 6))
o, r = shape.apply_event(m, opt, shape.GrowthEvent(1, new_d=16, add_layers=1))
o, r = shape.apply_event(m, o, shape.GrowthEvent(2, new_d=32))
print("  eps:", m.lnf.eps, [ (b.ln1.eps, b.ln2.eps) for b in m.blocks])
fresh = shape.GPT(23, 6, 32, 2, 2, shape.fact_lin(4))
fresh.load_state_dict(m.state_dict())
with torch.no_grad():
    a, _ = m(x); b, _ = fresh(x)
print("  eps fresh:", fresh.lnf.eps, " rel diff logits modelo vs reconstruido:", ((a-b).norm()/a.norm()).item())
