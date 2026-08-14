"""Tests de M2 controller — criterios de SPEC-elasticshape-v1.md (M2)."""
import pytest
import torch
import torch.nn as nn

from mztrain.elastic_shape import (
    GrowthEvent,
    LrWarmup,
    ShapeSchedule,
    apply_event,
    deepen_gpt,
    factorize_gpt,
    migrate_optimizer,
    qkv_out_map,
    rel_drift,
    widen_gpt,
)
from mztrain.elastic_shape import GPT, dense_lin, fact_lin
from mztrain.shape_ops import widen_layernorm

torch.manual_seed(0)
VOCAB, SEQ = 50, 16


def _tokens(b=2):
    g = torch.Generator().manual_seed(42)
    return torch.randint(0, VOCAB, (b, SEQ), generator=g)


def _fact_gpt(d=48, layers=2, heads=2, r=12, seed=1):
    torch.manual_seed(seed)
    return GPT(VOCAB, SEQ, d, layers, heads, fact_lin(r))


# ---------- compensacion de varianza de LN (SPEC §5) ----------

def test_ln_variance_compensation_reduces_drift():
    torch.manual_seed(2)
    ln = nn.LayerNorm(48)
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.2); ln.bias.normal_(0.0, 0.1)
    x = torch.randn(64, 48)
    ref = ln(x)
    x_pad = torch.cat([x, torch.zeros(64, 24)], dim=1)
    no_comp = widen_layernorm(ln, 72)
    comp = widen_layernorm(ln, 72, variance_compensation=True)
    err_no = float((no_comp(x_pad)[:, :48] - ref).norm())
    err_si = float((comp(x_pad)[:, :48] - ref).norm())
    assert err_si < 0.5 * err_no  # cancela el factor sqrt(d'/d), queda el shift


# ---------- widen end-to-end ----------

def test_widen_gpt_drift_small_noise0_multiseed():
    # banda medida (G4-A/B): 3.85-5.27% en 7 seeds de este config; umbral 7%
    # = banda + margen (SPEC §5). Un solo seed era fragil a reseed (G4-B).
    x = _tokens()
    for seed in (1, 3, 9):
        model = _fact_gpt(seed=seed)
        with torch.no_grad():
            ref, _ = model(x)
        recs = widen_gpt(model, 72, noise_scale=0.0)
        assert model.d == 72 and len(recs) > 0
        drift = rel_drift(model, x, ref)
        assert drift <= 0.07, f"seed {seed}: deriva {drift:.4f} > 7%"


def test_widen_gpt_drift_small_with_noise():
    model = _fact_gpt(seed=3)
    x = _tokens()
    with torch.no_grad():
        ref, _ = model(x)
    g = torch.Generator().manual_seed(7)
    widen_gpt(model, 72, noise_scale=1e-3, generator=g)
    assert rel_drift(model, x, ref) <= 0.08


def test_widen_gpt_requires_factorized_and_leaves_model_intact():
    """G4-B 1c: el preflight debe correr ANTES de mutar nada — tras el
    TypeError el modelo debe seguir funcionando identico."""
    torch.manual_seed(4)
    model = GPT(VOCAB, SEQ, 32, 2, 2, dense_lin)
    x = _tokens()
    with torch.no_grad():
        ref, _ = model(x)
    with pytest.raises(TypeError):
        widen_gpt(model, 48)
    assert model.d == 32
    assert model.tok.embedding_dim == 32     # embeddings NO mutados
    with torch.no_grad():
        out, _ = model(x)                    # forward sigue vivo
    assert torch.equal(out, ref)


def test_widen_pending_ledger_guard():
    """G4-B 1a/1b: dos widen sin migrar en medio deben FALLAR ruidosamente."""
    model = _fact_gpt(d=32, r=8, seed=10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    recs = widen_gpt(model, 48, noise_scale=0.0)
    with pytest.raises(RuntimeError, match="pendiente"):
        widen_gpt(model, 64, noise_scale=0.0)
    migrate_optimizer(model, opt, recs)      # limpia el flag
    recs2 = widen_gpt(model, 64, noise_scale=0.0)
    assert len(recs2) > 0


def test_migrate_inherits_lr_and_weight_decay():
    """G4-B item 3: wd del viejo optimizer heredado, no pisado por default."""
    model = _fact_gpt(d=32, r=8, seed=11)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.1)
    recs = widen_gpt(model, 48, noise_scale=0.0)
    new_opt = migrate_optimizer(model, opt, recs)
    assert abs(new_opt.param_groups[0]["lr"] - 2e-3) < 1e-12
    assert abs(new_opt.param_groups[0]["weight_decay"] - 0.1) < 1e-12


def test_factorize_redundante_no_resetea_estado():
    """G4-A item 6: factorize sobre modelo ya factorizado NO debe vaciar el
    optimizer."""
    model = _fact_gpt(d=32, r=8, seed=12)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = _tokens()
    for _ in range(3):
        opt.zero_grad(); _, loss = model(x, x); loss.backward(); opt.step()
    n_state = len(opt.state)
    assert n_state > 0
    ev = GrowthEvent(step=3, factorize=True)  # redundante
    opt2, report = apply_event(model, opt, ev)
    assert opt2 is opt                        # sin reset
    assert len(opt2.state) == n_state
    assert any("INTACTO" in s for s in report["optimizer"])


def test_apply_event_qstate_rescale():
    """G4-A item 5 (fix real): tras widen via apply_event, m del bloque q
    queda dividido por c y v por c^2; k/v intactos."""
    import math as m_
    model = _fact_gpt(d=32, r=8, seed=13)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = _tokens()
    for _ in range(4):
        opt.zero_grad(); _, loss = model(x, x); loss.backward(); opt.step()
    blk = model.blocks[0]
    m_old = opt.state[blk.qkv.U]["exp_avg"].clone()
    v_old = opt.state[blk.qkv.U]["exp_avg_sq"].clone()
    ev = GrowthEvent(step=4, new_d=48, noise_scale=0.0)
    opt2, _ = apply_event(model, opt, ev)
    c = m_.sqrt((48 // 2) / (32 // 2))
    omap = qkv_out_map(32, 48, 2)
    st = opt2.state[blk.qkv.U]
    q_new, kv_new = omap[:32], omap[32:]
    assert torch.allclose(st["exp_avg"][q_new, :], m_old[:32, :] / c)
    assert torch.allclose(st["exp_avg_sq"][q_new, :], v_old[:32, :] / (c * c))
    assert torch.equal(st["exp_avg"][kv_new, :], m_old[32:, :])


def test_chained_widens_drift_composes():
    model = _fact_gpt(d=32, r=8, seed=14)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = _tokens()
    with torch.no_grad():
        ref, _ = model(x)
    recs = widen_gpt(model, 48, noise_scale=0.0)
    opt = migrate_optimizer(model, opt, recs)
    recs = widen_gpt(model, 64, noise_scale=0.0)
    opt = migrate_optimizer(model, opt, recs)
    assert model.d == 64
    assert rel_drift(model, x, ref) <= 0.12   # dos etapas 1.5x y 1.33x


def test_event_widen_plus_deepen_sin_factorize():
    model = _fact_gpt(d=32, r=8, seed=15)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = _tokens()
    for _ in range(3):
        opt.zero_grad(); _, loss = model(x, x); loss.backward(); opt.step()
    ev = GrowthEvent(step=3, new_d=48, add_layers=1, noise_scale=0.0)
    opt2, report = apply_event(model, opt, ev, probe_x=x)
    assert len(model.blocks) == 3 and model.d == 48
    assert report["logits_drift_rel"] <= 0.08
    assert len(opt2.state) > 0                # estado migrado, no vacio


def test_heads3_widen_drift():
    """G4-A gap: heads distinto de 2 (interleaving por cabeza)."""
    torch.manual_seed(16)
    model = GPT(VOCAB, SEQ, 48, 2, 3, fact_lin(12))
    x = _tokens()
    with torch.no_grad():
        ref, _ = model(x)
    widen_gpt(model, 72, noise_scale=0.0)   # hd 16 -> 24, heads=3
    assert rel_drift(model, x, ref) <= 0.07


def test_widen_gpt_validates_dims():
    model = _fact_gpt()
    with pytest.raises(ValueError):
        widen_gpt(model, 47)   # no multiplo de heads
    with pytest.raises(ValueError):
        widen_gpt(model, 24)   # encoger


# ---------- profundidad exacta / conversion exacta ----------

def test_deepen_exact_identity():
    model = _fact_gpt(seed=5)
    x = _tokens()
    with torch.no_grad():
        ref, _ = model(x)
    deepen_gpt(model, 2)
    assert len(model.blocks) == 4
    with torch.no_grad():
        out, _ = model(x)
    assert torch.equal(out, ref)


def test_factorize_gpt_exact():
    torch.manual_seed(6)
    model = GPT(VOCAB, SEQ, 32, 2, 2, dense_lin)
    x = _tokens()
    with torch.no_grad():
        ref, _ = model(x)
    factorize_gpt(model)
    with torch.no_grad():
        out, _ = model(x)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


# ---------- migracion del optimizer ----------

def test_migrate_optimizer_momentum_preserved_and_step_cloned():
    model = _fact_gpt(d=32, r=8, seed=7)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = _tokens()
    for _ in range(5):
        opt.zero_grad()
        _, loss = model(x, x)
        loss.backward(); opt.step()
    blk = model.blocks[0]
    old_U, old_S = blk.qkv.U, blk.qkv.S
    old_tok_w = model.tok.weight
    m_old = opt.state[old_U]["exp_avg"].clone()
    s_m_old = opt.state[old_S]["exp_avg"].clone()
    tok_m_old = opt.state[old_tok_w]["exp_avg"].clone()
    step_old = opt.state[old_U]["step"].clone()

    recs = widen_gpt(model, 48, noise_scale=0.0)
    new_opt = migrate_optimizer(model, opt, recs, lr=1e-3)

    omap = qkv_out_map(32, 48, model.heads)
    st = new_opt.state[blk.qkv.U]
    assert torch.equal(st["exp_avg"][omap, :], m_old)          # momento intacto
    assert torch.equal(new_opt.state[blk.qkv.S]["exp_avg"], s_m_old)
    assert float(st["step"]) == float(step_old)
    st["step"] += 1                                            # sin alias
    assert float(opt.state[old_U]["step"]) == float(step_old)
    # embedding reemplazado: momento migrado por columnas, cols nuevas en cero
    tok_st = new_opt.state[model.tok.weight]
    assert tok_st["exp_avg"].shape == (VOCAB, 48)
    assert torch.equal(tok_st["exp_avg"][:, :32], tok_m_old)
    assert torch.equal(tok_st["exp_avg"][:, 32:], torch.zeros(VOCAB, 16))


def test_full_morph_trains_and_new_dims_alive():
    torch.manual_seed(8)
    model = GPT(VOCAB, SEQ, 32, 2, 2, dense_lin)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = _tokens(4)
    for _ in range(5):
        opt.zero_grad(); _, loss = model(x, x); loss.backward(); opt.step()
    ev = GrowthEvent(step=5, factorize=True, new_d=48, add_layers=1)
    g = torch.Generator().manual_seed(9)
    opt, report = apply_event(model, opt, ev, lr=3e-3, probe_x=x, generator=g)
    assert report["logits_drift_rel"] <= 0.10
    assert len(model.blocks) == 3 and model.d == 48
    warm = LrWarmup(opt, steps=10)
    losses = []
    for _ in range(60):
        opt.zero_grad(); _, loss = model(x, x); loss.backward(); opt.step()
        warm.step(); losses.append(float(loss))
    assert all(l == l for l in losses)                       # sin NaN
    assert sum(losses[-10:]) < sum(losses[:10])              # aprende
    assert float(model.tok.weight.grad[:, 32:].norm()) > 0.0  # dims nuevas vivas


# ---------- schedule y warmup ----------

def test_schedule_pop_due_consumes():
    sched = ShapeSchedule([GrowthEvent(step=10, new_d=64),
                           GrowthEvent(step=20, add_layers=1)])
    assert sched.pending == 2
    assert sched.pop_due(5) == []
    due = sched.pop_due(10)
    assert len(due) == 1 and due[0].new_d == 64
    assert sched.pop_due(10) == []
    assert sched.pending == 1


def test_lr_warmup_stacking_preserves_base():
    """G4-B item 6: un warmup sobre otro sin terminar NO debe capturar el lr
    escalado como base — la clave persistente en el param_group lo garantiza."""
    p = nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=1e-3)
    w1 = LrWarmup(opt, steps=50, floor=0.1)
    for _ in range(5):
        w1.step()                              # lr actual ~1.9e-4, no done
    w2 = LrWarmup(opt, steps=10, floor=0.1)    # warmup encima de warmup
    for _ in range(10):
        w2.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-12  # objetivo original


def test_lr_warmup_ramp():
    p = nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=1e-3)
    warm = LrWarmup(opt, steps=10, floor=0.1)
    assert abs(opt.param_groups[0]["lr"] - 1e-4) < 1e-12
    for _ in range(5):
        warm.step()
    assert abs(opt.param_groups[0]["lr"] - 0.55e-3) < 1e-9
    for _ in range(5):
        warm.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-12
    assert warm.done
    warm.step()  # no-op
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-12
