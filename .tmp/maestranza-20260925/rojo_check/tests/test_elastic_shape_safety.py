"""Regresiones de cirugia atomica, calibracion funcional y estado AdamW."""

import copy
import math

import pytest
import torch
import torch.nn as nn

import mztrain.elastic_shape as shape
from mztrain.layers import ZFactorizedLinear
from mztrain.shape_ops import pad_adam_entry, widen_factorized_linear


def model_and_opt(dense=False, **kwargs):
    torch.manual_seed(123)
    model = shape.GPT(23, 6, 8, 1, 2, shape.dense_lin if dense else shape.fact_lin(4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, **kwargs)
    x = torch.randint(0, 23, (2, 6))
    opt.zero_grad()
    model(x, x)[1].backward()
    opt.step()
    return model, opt, x


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_functional_noise_budget_independent_of_factor_scale(scale):
    torch.manual_seed(4)
    layer = ZFactorizedLinear(6, 5, 3, bias=False, init_method="random").double()
    with torch.no_grad():
        layer.U.normal_()
        layer.V.normal_()
        layer.U[:, 0] *= scale
        layer.S[0] /= scale
        layer.wake_gate[1] = 0.2
    old = (layer.U * layer._gated_s()) @ layer.V
    new = widen_factorized_linear(layer, 9, 8, noise_scale=0.003, generator=torch.Generator().manual_seed(8))
    weight = (new.U * new._gated_s()) @ new.V
    torch.testing.assert_close(weight[:5, :6], old, atol=1e-12, rtol=1e-12)
    assert weight[:, 6:].count_nonzero() == 0
    assert math.isclose((weight[5:].norm() / old.norm()).item(), 0.003, rel_tol=1e-10)


@pytest.mark.parametrize("noise", [-1.0, float("nan"), float("inf")])
def test_invalid_noise_rejected_before_mutation(noise):
    model, opt, _ = model_and_opt()
    params = list(model.parameters())
    with pytest.raises(ValueError):
        shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12, noise_scale=noise))
    assert all(a is b for a, b in zip(params, model.parameters()))


def test_rank_one_noise_is_finite():
    layer = ZFactorizedLinear(1, 1, 1, bias=False, init_method="random")
    wide = widen_factorized_linear(layer, 2, 2, noise_scale=0.01)
    assert all(torch.isfinite(p).all() for p in wide.parameters())


@pytest.mark.parametrize("failed_layer", ["proj", "fc1", "fc2"])
def test_all_layers_preflight_preserves_model(failed_layer):
    model, _, x = model_and_opt()
    old = getattr(model.blocks[0], failed_layer)
    setattr(model.blocks[0], failed_layer, nn.Linear(old.in_features, old.out_features, bias=False))
    params = list(model.parameters())
    logits = model(x)[0].detach()
    with pytest.raises(TypeError):
        shape.widen_gpt(model, 12)
    assert model.d == 8
    assert all(a is b for a, b in zip(params, model.parameters()))
    assert torch.equal(model(x)[0].detach(), logits)


def test_exception_rolls_back_model_optimizer_and_rng(monkeypatch):
    model, opt, x = model_and_opt(dense=True)
    original = copy.deepcopy(model.state_dict())
    old_opt = copy.deepcopy(opt.state_dict())
    params = list(model.parameters())
    blocks = model.blocks
    generator = torch.Generator().manual_seed(19)
    rng = torch.get_rng_state().clone()
    grng = generator.get_state().clone()

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure after widen")

    monkeypatch.setattr(shape, "migrate_optimizer", fail)
    with pytest.raises(RuntimeError, match="injected"):
        shape.apply_event(
            model, opt, shape.GrowthEvent(1, factorize=True, new_d=12, add_layers=1), probe_x=x, generator=generator
        )
    assert model.d == 8 and len(model.blocks) == 1 and model.blocks is blocks
    assert all(a is b for a, b in zip(params, model.parameters()))
    assert_state_equal(model.state_dict(), original)
    assert_state_equal(opt.state_dict(), old_opt)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(generator.get_state(), grng)


def test_quality_rejection_restores_live_training_state():
    model, opt, x = model_and_opt(amsgrad=True)
    model.eval()
    model.blocks[0].ln1.train()
    modes = [(m, m.training) for m in model.modules()]
    state = copy.deepcopy(opt.state_dict())
    params = list(model.parameters())
    logits = model(x)[0].detach()
    rng = torch.get_rng_state().clone()
    returned, report = shape.apply_event(
        model, opt, shape.GrowthEvent(1, new_d=16, add_layers=1), probe_x=x, probe_targets=x, max_kl=0.0
    )
    assert returned is opt and report["accepted"] is False and report["rolled_back"] is True
    assert report["kl_div"] > 0
    assert "loss_delta" in report
    assert model.d == 8 and len(model.blocks) == 1
    assert all(a is b for a, b in zip(params, model.parameters()))
    assert all(m.training == mode for m, mode in modes)
    assert torch.equal(model(x)[0].detach(), logits)
    assert_state_equal(opt.state_dict(), state)
    assert torch.equal(torch.get_rng_state(), rng)
    opt.zero_grad()
    model(x, x)[1].backward()
    opt.step()


def test_guard_accepts_identity_and_reports_loss_and_kl():
    model, opt, x = model_and_opt()
    returned, report = shape.apply_event(
        model, opt, shape.GrowthEvent(1, add_layers=1), probe_x=x, probe_targets=x, max_kl=1e-10, max_loss_increase=1e-10
    )
    assert returned is not opt and report["accepted"] and report["guarded"]
    assert report["loss_delta"] == 0
    assert report["kl_div"] <= 1e-10
    assert len(model.blocks) == 2


@pytest.mark.parametrize("kwargs", [{"max_kl": 0.01}, {"max_loss_increase": 0.01}, {"max_kl": -1.0}])
def test_guard_requires_valid_probe_and_thresholds(kwargs):
    model, opt, _ = model_and_opt()
    with pytest.raises(ValueError):
        shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12), **kwargs)
    assert model.d == 8


@pytest.mark.parametrize("dense", [False, True])
def test_multiple_groups_and_options_survive_complete_event(dense):
    model, _, x = model_and_opt(dense=dense)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if "ln" in name or name.endswith("bias") else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "lr": 0.001, "weight_decay": 0.02, "label": "decay"},
            {"params": no_decay, "lr": 0.002, "weight_decay": 0.0, "label": "norm"},
        ],
        betas=(0.8, 0.91),
        eps=1e-5,
        amsgrad=True,
        foreach=False,
    )
    model(x, x)[1].backward()
    opt.step()
    new, report = shape.apply_event(model, opt, shape.GrowthEvent(1, factorize=dense, new_d=12, add_layers=1))
    assert report["accepted"]
    assert len(new.param_groups) == 2
    for before, after in zip(opt.param_groups, new.param_groups):
        for key in ["lr", "weight_decay", "label", "betas", "eps", "amsgrad", "foreach"]:
            assert before[key] == after[key]
    membership = {id(p): i for i, group in enumerate(new.param_groups) for p in group["params"]}
    assert len(membership) == len(list(model.parameters()))
    for name, p in model.named_parameters():
        assert membership[id(p)] == (1 if "ln" in name or name.endswith("bias") else 0)
    new.zero_grad()
    model(x, x)[1].backward()
    new.step()


def test_amsgrad_moments_are_migrated_and_rescaled():
    model, opt, _ = model_and_opt(amsgrad=True)
    old = model.blocks[0].qkv.U
    old_state = copy.deepcopy(opt.state[old])
    new, _ = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12, noise_scale=0))
    rows = shape.qkv_out_map(8, 12, 2)
    state = new.state[model.blocks[0].qkv.U]
    for key in ["exp_avg_sq", "max_exp_avg_sq"]:
        torch.testing.assert_close(state[key][rows[:8]], old_state[key][:8] / 1.5)
        assert torch.equal(state[key][rows[8:]], old_state[key][8:])
    state["step"] += 10
    assert torch.equal(opt.state[old]["step"], old_state["step"])


def test_deepen_only_clones_untouched_optimizer_state():
    model, opt, _ = model_and_opt()
    p = model.tok.weight
    old = copy.deepcopy(opt.state[p])
    new, _ = shape.apply_event(model, opt, shape.GrowthEvent(1, add_layers=1))
    new.state[p]["exp_avg"].add_(10)
    new.state[p]["step"].add_(10)
    assert_state_equal(opt.state[p], old)


def test_warmup_base_survives_optimizer_rebuild():
    model, opt, _ = model_and_opt()
    warm = shape.LrWarmup(opt, steps=100)
    warm.step()
    new, _ = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12))
    after = shape.LrWarmup(new, steps=2)
    after.step()
    after.step()
    assert new.param_groups[0]["lr"] == 0.001


def test_pad_amsgrad_and_metadata_without_aliases():
    entry = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.ones(2, 2),
        "exp_avg_sq": torch.ones(2, 2),
        "max_exp_avg_sq": torch.full((2, 2), 2.0),
        "metadata": {"history": [1]},
    }
    out = pad_adam_entry(entry, (3, 3))
    assert out["max_exp_avg_sq"].shape == (3, 3)
    assert out["max_exp_avg_sq"][2].count_nonzero() == 0
    out["metadata"]["history"].append(2)
    assert entry["metadata"]["history"] == [1]


def test_deepen_preserves_bfloat16_and_trainability():
    model, _, x = model_and_opt()
    model = model.to(torch.bfloat16).eval()
    model.blocks[0].ln1.weight.requires_grad_(False)
    before = model(x)[0].detach()
    shape.deepen_gpt(model, 1)
    assert all(p.dtype == torch.bfloat16 for p in model.blocks[-1].parameters())
    assert model.blocks[-1].ln1.weight.requires_grad is False
    assert model.blocks[-1].training is False
    assert torch.equal(model(x)[0].detach(), before)


def test_loss_guard_rejects_actual_loss_increase():
    model, opt, x = model_and_opt()
    trial = copy.deepcopy(model)
    trial_opt = torch.optim.AdamW(trial.parameters())
    event = shape.GrowthEvent(1, new_d=16, noise_scale=0)
    shape.apply_event(trial, trial_opt, event)
    before = model(x)[0].detach().log_softmax(-1)
    after = trial(x)[0].detach().log_softmax(-1)
    targets = (before - after).argmax(-1)
    returned, report = shape.apply_event(model, opt, event, probe_x=x, probe_targets=targets, max_loss_increase=0)
    assert returned is opt and not report["accepted"]
    assert report["loss_delta"] > 0 and "loss" in report["rejection_reason"]
    assert model.d == 8


def test_nonfinite_candidate_is_rejected_without_probe(monkeypatch):
    model, opt, _ = model_and_opt()
    original = copy.deepcopy(model.state_dict())
    widen = shape.widen_factorized_linear

    def poison(*args, **kwargs):
        new = widen(*args, **kwargs)
        with torch.no_grad():
            new.U.fill_(float("nan"))
        return new

    monkeypatch.setattr(shape, "widen_factorized_linear", poison)
    returned, report = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12, noise_scale=0))
    assert returned is opt and not report["accepted"]
    assert "no finitos" in report["rejection_reason"]
    assert_state_equal(model.state_dict(), original)


def test_kl_and_loss_ignore_shared_logit_offset():
    before = torch.tensor([[[1000.0, 1001.0]]])
    after = torch.tensor([[[1001.0, 1000.0]]])
    target = torch.tensor([[0]])
    metrics = shape._probe_metrics(before, after, target)
    centered = shape._probe_metrics(before - 1000, after - 1000, target)
    assert metrics["logits_drift_rel"] < 0.001
    assert metrics["loss_delta"] == -1.0
    assert metrics["kl_div"] > 0.4
    assert metrics["kl_div"] == centered["kl_div"]


def test_mixed_s_dtype_and_frozen_parameters_survive_widen():
    layer = ZFactorizedLinear(4, 4, 2, bias=False).to(torch.bfloat16)
    layer.S.data = torch.tensor([1.001, 0.0000234])
    layer.wake_gate = torch.tensor([0.9999, 0.1234])
    layer.S.requires_grad_(False)
    new = widen_factorized_linear(layer, 6, 6, noise_scale=0)
    assert new.U.dtype == torch.bfloat16
    assert new.S.dtype == torch.float32 and torch.equal(new.S, layer.S)
    assert new.wake_gate.dtype == torch.float32 and torch.equal(new.wake_gate, layer.wake_gate)
    assert not new.S.requires_grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_guard_handles_bfloat16_factors_with_fp32_s(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA no disponible")
    model, _, x = model_and_opt()
    model.to(device=device, dtype=torch.bfloat16)
    for layer in model.modules():
        if isinstance(layer, ZFactorizedLinear):
            layer.S.data = layer.S.data.float()
            layer.wake_gate = layer.wake_gate.float()
    x = x.to(device)
    opt = torch.optim.AdamW(model.parameters())
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        model(x, x)[1].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    new, report = shape.apply_event(
        model, opt, shape.GrowthEvent(1, new_d=12, add_layers=1), probe_x=x, probe_targets=x, max_kl=1.0
    )
    assert report["accepted"] and math.isfinite(report["loss_delta"])
    assert all(layer.S.dtype == torch.float32 for layer in model.modules() if isinstance(layer, ZFactorizedLinear))
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss = model(x, x)[1]
    assert torch.isfinite(loss)
    loss.backward()
    new.step()


def test_zero_function_receives_zero_functional_noise():
    layer = ZFactorizedLinear(4, 4, 2, bias=False)
    with torch.no_grad():
        layer.S.zero_()
    new = widen_factorized_linear(layer, 6, 6, noise_scale=0.1)
    assert torch.equal(new(torch.ones(2, 6)), torch.zeros(2, 6))


def test_bias_moments_and_new_block_bias_are_preserved():
    model = shape.GPT(23, 6, 8, 1, 2, lambda i, o: ZFactorizedLinear(i, o, 4, bias=True))
    opt = torch.optim.AdamW(model.parameters(), amsgrad=True)
    x = torch.randint(0, 23, (2, 6))
    model(x, x)[1].backward()
    opt.step()
    old = copy.deepcopy(opt.state[model.blocks[0].qkv.bias])
    new, _ = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12, add_layers=1))
    rows = shape.qkv_out_map(8, 12, 2)
    state = new.state[model.blocks[0].qkv.bias]
    torch.testing.assert_close(state["exp_avg"][rows[:8]], old["exp_avg"][:8] / math.sqrt(1.5))
    assert model.blocks[-1].qkv.bias is not None
    new.zero_grad()
    model(x, x)[1].backward()
    new.step()


def test_excluded_parameters_do_not_reenter_optimizer():
    model, _, _ = model_and_opt()
    excluded = model.blocks[0].qkv.S
    opt = torch.optim.AdamW([p for p in model.parameters() if p is not excluded])
    new, _ = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12, add_layers=1))
    owned = {id(p) for group in new.param_groups for p in group["params"]}
    assert id(model.blocks[0].qkv.S) not in owned
    assert id(model.blocks[1].qkv.S) not in owned


def test_stale_ledger_rejected_without_clearing_pending_state():
    model, opt, _ = model_and_opt()
    recs = shape.widen_gpt(model, 12)
    with pytest.raises(ValueError, match="ledger"):
        shape.migrate_optimizer(model, opt, recs[:-1])
    assert model._mzshape_pending_recs
    shape.migrate_optimizer(model, opt, recs)
    assert not model._mzshape_pending_recs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
@pytest.mark.parametrize("options", [{"capturable": True, "foreach": False}, {"fused": True}])
def test_cuda_adamw_options_and_rollback_rng(options):
    model, _, x = model_and_opt()
    model, x = model.cuda(), x.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True, **options)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x, x)[1].backward()
    opt.step()
    event = shape.GrowthEvent(1, new_d=12)
    rng = torch.cuda.get_rng_state().clone()
    gen = torch.Generator(device="cuda").manual_seed(44)
    grng = gen.get_state().clone()
    returned, report = shape.apply_event(model, opt, event, probe_x=x, max_kl=0, generator=gen)
    assert returned is opt and not report["accepted"]
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert torch.equal(gen.get_state(), grng)
    new, report = shape.apply_event(model, opt, event, probe_x=x, max_kl=1.0, generator=gen)
    assert report["accepted"]
    for key, value in options.items():
        assert new.param_groups[0][key] == value
    new.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, x)[1]
    loss.backward()
    new.step()
    assert torch.isfinite(loss)


# --- anclas de peritaje (PER-ELASTIC, 2026-09-25) ---------------------------


def test_ancla_per_log_001_warmup_base_stale_after_external_lr_change():
    """PER-LOG-001: un warmup TERMINADO debe soltar su base persistente; si un
    scheduler externo cambia el LR despues (step decay), el proximo warmup
    debe rampear a partir del LR vigente, no del pico del warmup anterior.
    El caso G4-B (warmup SIN terminar) esta cubierto por
    test_warmup_base_survives_optimizer_rebuild y debe seguir verde."""

    def run(decay_between_growths):
        model, opt, x = model_and_opt()
        w1 = shape.LrWarmup(opt, steps=5)
        for _ in range(5):
            opt.zero_grad()
            model(x, x)[1].backward()
            opt.step()
            w1.step()
        assert w1.done and math.isclose(opt.param_groups[0]["lr"], 1e-3)
        if decay_between_growths:
            for g in opt.param_groups:
                g["lr"] = 1e-4
        pre_growth_lr = opt.param_groups[0]["lr"]
        new, report = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12, noise_scale=0.0))
        assert report["accepted"]
        w2 = shape.LrWarmup(new, steps=5)
        for _ in range(5):
            new.zero_grad()
            model(x, x)[1].backward()
            new.step()
            w2.step()
        assert w2.done
        return pre_growth_lr, new.param_groups[0]["lr"]

    # control negativo: sin cambio externo, el warmup vuelve al LR previo.
    pre, final = run(decay_between_growths=False)
    assert math.isclose(final, pre), f"control roto: {final} != {pre}"

    # defecto: con decaimiento externo tras el warmup, el siguiente warmup
    # debe terminar en el LR vigente (1e-4), no en la base caduca (1e-3).
    pre, final = run(decay_between_growths=True)
    assert math.isclose(final, pre, rel_tol=1e-9), (
        f"PER-LOG-001: warmup termino en lr={final:g}, no en el LR vigente {pre:g} tras el cambio externo"
    )


def test_ancla_per_log_002_lr_weight_decay_overrides_apply_without_surgery():
    """PER-LOG-002: si apply_event(..., lr=L, weight_decay=W) acepta el
    evento, TODOS los grupos deben quedar con lr==L y weight_decay==W, aun
    cuando el evento no dispara ninguna cirugia (factorize redundante,
    new_d == d actual)."""
    LR, WD = 5e-4, 0.05

    def setup():
        torch.manual_seed(1)
        model = shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4))  # ya factorizado
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        x = torch.randint(0, 23, (2, 6))
        opt.zero_grad()
        model(x, x)[1].backward()
        opt.step()
        return model, opt

    # control negativo: un evento que SI migra (deepen) aplica el override.
    model, opt = setup()
    new, report = shape.apply_event(model, opt, shape.GrowthEvent(1, add_layers=1), lr=LR, weight_decay=WD)
    assert report["accepted"]
    assert all((g["lr"], g["weight_decay"]) == (LR, WD) for g in new.param_groups)

    for event in (shape.GrowthEvent(1, factorize=True), shape.GrowthEvent(1, new_d=8)):
        model, opt = setup()
        new, report = shape.apply_event(model, opt, event, lr=LR, weight_decay=WD)
        assert report["accepted"], event
        groups = [(g["lr"], g["weight_decay"]) for g in new.param_groups]
        assert all(g == (LR, WD) for g in groups), (event, groups)
        assert new is not opt
        assert opt.param_groups[0]["lr"] == 1e-3  # optimizer original intacto


def test_ancla_per_log_005_rejects_reuse_of_consumed_ledger():
    """PER-LOG-005: re-migrar un ledger YA consumido (o nunca emitido) debe
    fallar ruidosamente, no transplantar momentos caducos en silencio.
    recs == [] (deepen) debe seguir permitido sin ledger pendiente."""
    model, opt, x = model_and_opt()
    recs = shape.widen_gpt(model, 12)
    opt2 = shape.migrate_optimizer(model, opt, recs)
    assert not model._mzshape_pending_recs
    opt2.zero_grad()
    model(x, x)[1].backward()
    opt2.step()
    live_step = float(opt2.state[model.tok.weight]["step"])
    assert live_step > 0

    with pytest.raises(ValueError, match="ledger ya consumido"):
        shape.migrate_optimizer(model, opt, recs)
    # el optimizer vivo no debe haberse visto afectado por el intento.
    assert float(opt2.state[model.tok.weight]["step"]) == live_step

    # control: recs vacio (deepen) sigue permitido sin ledger pendiente.
    shape.deepen_gpt(model, 1)
    opt3 = shape.migrate_optimizer(model, opt2, [])
    assert opt3 is not None and not model._mzshape_pending_recs


def test_ancla_per_log_003_noise_error_becomes_quality_rejection():
    """PER-LOG-003: un candidato no finito por ruido no representable en el
    dtype (NoiseError de shape_ops) debe verse como cualquier otro rechazo de
    calidad: accepted=False, rolled_back=True, optimizer ORIGINAL, modelo
    restaurado — nunca una excepcion propagada."""
    model, opt, x = model_and_opt()
    original_state = copy.deepcopy(opt.state_dict())
    params = list(model.parameters())

    returned, report = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12, noise_scale=1e300))

    assert returned is opt
    assert report["accepted"] is False and report["rolled_back"] is True
    assert "ruido no representable" in report["rejection_reason"]
    assert model.d == 8
    assert all(a is b for a, b in zip(params, model.parameters()))
    assert_state_equal(opt.state_dict(), original_state)


@pytest.mark.parametrize(
    "bad_targets",
    [
        torch.full((2, 6), 999, dtype=torch.long),
        torch.zeros(2, 6, dtype=torch.float32),
    ],
)
def test_ancla_per_log_004_probe_targets_validated_before_surgery(monkeypatch, bad_targets):
    """PER-LOG-004: probe_targets invalido (fuera de [0, vocab) o dtype no
    entero) debe rechazarse ANTES de ejecutar cualquier cirugia, no
    descubrirse tarde dentro de _probe_metrics."""
    model, opt, x = model_and_opt()
    calls = {"widen": 0}
    original_widen = shape.widen_gpt

    def counting_widen(*args, **kwargs):
        calls["widen"] += 1
        return original_widen(*args, **kwargs)

    monkeypatch.setattr(shape, "widen_gpt", counting_widen)
    with pytest.raises(ValueError, match="probe_targets"):
        shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=12), probe_x=x, probe_targets=bad_targets, max_kl=10.0)
    assert calls["widen"] == 0
    assert model.d == 8


def test_ancla_per_api_006_floor_above_one_rejected():
    """PER-API-006: LrWarmup con floor > 1 invierte el proposito documentado
    (rampa DESCENDENTE desde un pico); debe rechazarse."""
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    with pytest.raises(ValueError, match="floor"):
        shape.LrWarmup(opt, steps=10, floor=2.0)

    # control negativo: floor=0.1 (valido) sigue arrancando bajo.
    opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    shape.LrWarmup(opt2, steps=10, floor=0.1)
    assert math.isclose(opt2.param_groups[0]["lr"], 1e-4)


@pytest.mark.parametrize("bad_floor", [float("nan"), -0.5, float("inf")])
def test_ancla_per_log_007_floor_nonfinite_or_negative_rejected(bad_floor):
    """PER-LOG-007: floor no finito o negativo debe rechazarse; nunca debe
    dejar g["lr"] no finito o negativo (ascenso de gradiente en silencio)."""
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.0)
    with pytest.raises(ValueError, match="floor"):
        shape.LrWarmup(opt, steps=10, floor=bad_floor)


def test_ancla_per_api_010_noise_scale_none_raises_clear_valueerror():
    """PER-API-010: GrowthEvent(noise_scale=None) (y widen_gpt(noise_scale=
    None) directamente) deben producir el ValueError claro de validacion,
    no un TypeError generico de math.isfinite(None)."""
    model = shape.GPT(23, 6, 8, 1, 2, shape.dense_lin)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ev = shape.GrowthEvent(step=0, factorize=True, new_d=16, noise_scale=None)
    with pytest.raises(ValueError, match="noise_scale"):
        shape.apply_event(model, opt, ev)

    with pytest.raises(ValueError, match="noise_scale"):
        shape.widen_gpt(shape.GPT(23, 6, 8, 1, 2, shape.fact_lin(4)), 16, noise_scale=None)

    # control negativo: noise_scale=0.0 es valido.
    model2 = shape.GPT(23, 6, 8, 1, 2, shape.dense_lin)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    _, report = shape.apply_event(model2, opt2, shape.GrowthEvent(step=0, factorize=True, new_d=16, noise_scale=0.0))
    assert report["accepted"] is True


def test_ancla_per_mat_001_qkv_noise_budget_survives_sdpa_correction():
    """PER-MAT-001: el presupuesto funcional de ruido ||dW||=noise_scale*||W||
    debe cumplirse en la capa qkv RESULTANTE de widen_gpt (tras la correccion
    de escala SDPA), no solo en la primitiva aislada."""
    torch.manual_seed(0)
    ns = 1e-2
    d, new_d, heads = 8, 32, 2  # hd 4 -> 16, c = 2

    def eff(layer):
        return (layer.U.detach().double() * layer._gated_s().detach().double()) @ layer.V.detach().double()

    model = shape.GPT(64, 16, d, 1, heads, shape.fact_lin(4)).double()
    old_qkv = model.blocks[0].qkv
    W_old = eff(old_qkv)
    omap = shape.qkv_out_map(d, new_d, heads)
    fresh = torch.ones(3 * new_d, dtype=torch.bool)
    fresh[omap] = False

    shape.widen_gpt(model, new_d, noise_scale=ns, generator=torch.Generator().manual_seed(7))
    W_new = eff(model.blocks[0].qkv)
    realized = (W_new[fresh].norm() / W_old.norm()).item() / ns
    assert abs(realized - 1.0) < 1e-6, f"PER-MAT-001: ratio realizado/ns en qkv = {realized:.6f}, esperado 1.0"
