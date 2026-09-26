"""Sonda del director (Z1). Cada bloque imprime OK/FALLA con la medida; no aborta al primer fallo."""
import sys, math, copy, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch, torch.nn as nn
import mztrain.elastic_shape as es
from mztrain.elastic_shape import GPT, dense_lin, fact_lin, GrowthEvent, apply_event, LrWarmup, widen_gpt, deepen_gpt, migrate_optimizer, factorize_gpt
from mztrain.shape_ops import dense_to_factorized, widen_factorized_linear
from mztrain.layers import ZFactorizedLinear

torch.manual_seed(0)
V, T, D, H = 64, 16, 24, 2
res = []
def rep(name, ok, detail=""):
    res.append((name, ok)); print(("OK   " if ok else "FALLA"), name, detail)

def mk(dense=True, bias=False):
    lin = dense_lin if dense else fact_lin(12)
    m = GPT(V, T, D, 2, H, lin)
    return m

def snapshot(model, opt):
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ids = {n: id(m) for n, m in model.named_modules()}
    pids = {n: id(p) for n, p in model.named_parameters()}
    ost = {id(p): {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in st.items()} for p, st in opt.state.items() if st}
    groups = [{k: copy.deepcopy(v) for k, v in g.items() if k != "params"} for g in opt.param_groups]
    return sd, ids, pids, ost, groups, torch.get_rng_state().clone()

def same_snapshot(a, b):
    sd1, ids1, pids1, ost1, g1, rng1 = a; sd2, ids2, pids2, ost2, g2, rng2 = b
    ok = sd1.keys() == sd2.keys() and all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    ok &= ids1 == ids2 and pids1 == pids2 and g1 == g2 and torch.equal(rng1, rng2)
    ok &= ost1.keys() == ost2.keys() and all(
        st1.keys() == ost2[k].keys() and all((torch.equal(st1[q], ost2[k][q]) if torch.is_tensor(st1[q]) else st1[q] == ost2[k][q]) for q in st1)
        for k, st1 in ost1.items())
    return ok

x = torch.randint(0, V, (2, T)); y = torch.randint(0, V, (2, T))
def train_steps(model, opt, n=3):
    for _ in range(n):
        opt.zero_grad(); _, loss = model(x, y); loss.backward(); opt.step()

# ---------- D1: rollback bit a bit con excepcion inyectada en cada etapa ----------
for stage in ("widen_gpt", "deepen_gpt", "migrate_optimizer", "_rescale_q_state"):
    m = mk(dense=False); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01, amsgrad=True); train_steps(m, opt)
    before = snapshot(m, opt)
    orig = getattr(es, stage)
    def boom(*a, **k): raise RuntimeError("inyectado")
    setattr(es, stage, boom)
    try:
        try:
            apply_event(m, opt, GrowthEvent(step=1, new_d=48, add_layers=1), probe_x=x)
            raised = False
        except RuntimeError as e:
            raised = "inyectado" in str(e)
    finally:
        setattr(es, stage, orig)
    after = snapshot(m, opt)
    rep(f"D1 rollback bitwise tras fallo en {stage}", raised and same_snapshot(before, after) and m.d == D and len(m.blocks) == 2 and not getattr(m, "_mzshape_pending_recs", False))

# D1b: rechazo por calidad (max_kl=0 con ruido) -> rollback + opt original
m = mk(dense=False); opt = torch.optim.AdamW(m.parameters(), lr=3e-4); train_steps(m, opt)
before = snapshot(m, opt)
o2, rpt = apply_event(m, opt, GrowthEvent(step=1, new_d=48, noise_scale=1e-2), probe_x=x, max_kl=0.0)
rep("D1b rechazo por calidad -> rollback bitwise y opt original", (o2 is opt) and rpt["accepted"] is False and same_snapshot(before, snapshot(m, opt)), str(rpt.get("rejection_reason")))

# ---------- D2: deepen -> dtype de buffers y de S ----------
m = mk(dense=False).to(torch.bfloat16)
for blk in m.blocks:
    for n in ("qkv", "proj", "fc1", "fc2"):
        getattr(blk, n).S.data = getattr(blk, n).S.data.float()  # S fp32 con U/V bf16 (politica del proyecto)
deepen_gpt(m, 1)
nb = m.blocks[-1]
rep("D2 deepen conserva S fp32 en bloque nuevo", nb.qkv.S.dtype == torch.float32, str(nb.qkv.S.dtype))
rep("D2 deepen conserva wake_gate fp32 en bloque nuevo", nb.qkv.wake_gate.dtype == m.blocks[0].qkv.wake_gate.dtype, f"{nb.qkv.wake_gate.dtype} vs plantilla {m.blocks[0].qkv.wake_gate.dtype}")

# ---------- D3: dense_to_factorized bf16 -> S dtype y error ----------
lin = nn.Linear(24, 72, bias=False).to(torch.bfloat16)
f = dense_to_factorized(lin)
rep("D3 dense_to_factorized bf16: S queda fp32 (regla 'S siempre FP32')", f.S.dtype == torch.float32, f"S.dtype={f.S.dtype}, rel_err={f._reconstruction_error:.3e}")
xx = torch.randn(4, 24).to(torch.bfloat16)
err = float((f(xx).float() - lin(xx).float()).norm() / lin(xx).float().norm())
rep("D3 dense_to_factorized bf16: forward reconstruye a <1e-2", err < 1e-2, f"rel forward err={err:.3e}")

# ---------- D4: LrWarmup base persistente vs LR externo ----------
m = mk(dense=False); opt = torch.optim.AdamW(m.parameters(), lr=3e-4); train_steps(m, opt)
w = LrWarmup(opt, steps=2); w.step(); w.step()
for g in opt.param_groups: g["lr"] = 1e-5  # un scheduler externo (cosine) baja el LR
o2, rpt = apply_event(m, opt, GrowthEvent(step=1, new_d=48), probe_x=x)
w2 = LrWarmup(o2, steps=2); w2.step(); w2.step()
rep("D4 LrWarmup tras LR externo: el objetivo del warmup es el LR vigente (1e-5), no la base vieja", abs(o2.param_groups[0]["lr"] - 1e-5) < 1e-12, f"lr final={o2.param_groups[0]['lr']}")

# ---------- D5: weight decay aplicado a S tras factorize ----------
m = mk(dense=True); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.1); train_steps(m, opt)
o2, rpt = apply_event(m, opt, GrowthEvent(step=1, factorize=True))
s_groups = [g["weight_decay"] for g in o2.param_groups for p in g["params"] if any(p is blk.qkv.S for blk in m.blocks)]
rep("D5 (observacion) S hereda weight_decay del grupo del weight denso", True, f"wd sobre S = {s_groups}")

# ---------- D6: escala del gradiente de U_q tras widen (noise=0) ----------
torch.manual_seed(1)
m = mk(dense=False).double(); xx = torch.randint(0, V, (2, T))
m.zero_grad(); _, loss = m(xx, y); loss.backward()
g_old = m.blocks[0].qkv.U.grad[:D].clone()  # filas q viejas (layout viejo: primeras D)
gV_old = m.blocks[0].qkv.V.grad.clone(); gS_old = m.blocks[0].qkv.S.grad.clone()
recs = widen_gpt(m, 48, noise_scale=0.0)
m.zero_grad(); _, loss2 = m(xx, y); loss2.backward()
qrows = es.qkv_out_map(D, 48, H)[:D]
g_new = m.blocks[0].qkv.U.grad[qrows]
c = math.sqrt((48 // H) / (D // H))
ratio = float((g_new.norm() / g_old.norm()))
rep("D6 grad U_q escala 1/c tras widen noise=0", abs(ratio - 1 / c) < 1e-6, f"ratio={ratio:.6f}, 1/c={1/c:.6f}")
rep("D6 grad V (compartido) invariante", torch.allclose(m.blocks[0].qkv.V.grad[:, :D], gV_old, atol=1e-9), f"max diff={float((m.blocks[0].qkv.V.grad[:, :D]-gV_old).abs().max()):.2e}")
rep("D6 grad S invariante", torch.allclose(m.blocks[0].qkv.S.grad, gS_old, atol=1e-9))
# gamma de LN: factor de escala del gradiente
# (informativo) medir ratio del grad de ln1.weight viejas
# ---------- D7: presupuesto de ruido tras scale_output_rows ----------
torch.manual_seed(2)
m = mk(dense=False).double()
qkv0 = m.blocks[0].qkv; W0 = qkv0.reconstruct_weight().clone()
gen = torch.Generator().manual_seed(3)
widen_gpt(m, 48, noise_scale=1e-2, generator=gen)
qkv1 = m.blocks[0].qkv; W1 = qkv1.reconstruct_weight()
omap = es.qkv_out_map(D, 48, H); fresh = torch.ones(3 * 48, dtype=torch.bool); fresh[omap] = False
dW = W1[fresh][:, :D]  # filas nuevas, columnas de entrada viejas
rep("D7 (medida) ||dW_filas_nuevas||/||W|| en qkv con noise 1e-2", True, f"ratio={float(dW.norm()/W0.norm()):.4e} (q-block escalado por c={c:.3f}; esperado 1e-2*sqrt((c^2+2)/3)={1e-2*math.sqrt((c*c+2)/3):.4e} si el ruido se repartio igual)")

print("\nRESUMEN:", sum(ok for _, ok in res), "/", len(res), "OK")
