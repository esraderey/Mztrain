"""Smoke test CPU reproducible: prediccion del siguiente token en un ciclo."""

import copy
import json

import torch
import torch.nn.functional as F

from mztrain.elastic_shape import GPT, GrowthEvent, LrWarmup, apply_event, fact_lin


torch.set_num_threads(1)
torch.manual_seed(20260907)
torch.use_deterministic_algorithms(True)
vocab, seq = 16, 8
data_rng = torch.Generator().manual_seed(73)


def batch():
    starts = torch.randint(vocab, (16, 1), generator=data_rng)
    tokens = (starts + torch.arange(seq + 1)) % vocab
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def train_step(model, opt, x, y):
    model.train()
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    opt.zero_grad(set_to_none=True)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits, _ = model(x)
    loss = F.cross_entropy(logits.double().reshape(-1, vocab), y.reshape(-1))
    return {"loss_nats": float(loss), "accuracy": float((logits.argmax(-1) == y).float().mean())}


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


model = GPT(vocab=vocab, seq=seq, d=16, layers=1, heads=2, lin=fact_lin(8))
opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01, amsgrad=True)
probe_x, probe_y = batch()
initial = evaluate(model, probe_x, probe_y)
for _ in range(60):
    train_step(model, opt, *batch())
baseline = evaluate(model, probe_x, probe_y)
continuation = [batch() for _ in range(10)]
results = []
for name, event in (
    ("profundidad_1_a_2", GrowthEvent(step=60, add_layers=1)),
    ("ancho_16_a_24", GrowthEvent(step=60, new_d=24, noise_scale=0.001)),
):
    candidate, candidate_opt = copy.deepcopy((model, opt))
    old_opt = candidate_opt
    state_before = copy.deepcopy(candidate.state_dict())
    opt_before = copy.deepcopy(candidate_opt.state_dict())
    refs = list(candidate.parameters())
    count_before = sum(p.numel() for p in candidate.parameters())
    generator = torch.Generator().manual_seed(71)
    rng_before = torch.get_rng_state().clone()
    generator_before = generator.get_state().clone()
    candidate_opt, report = apply_event(
        candidate,
        candidate_opt,
        event,
        probe_x=probe_x,
        probe_targets=probe_y,
        generator=generator,
        max_loss_increase=0.02,
        max_kl=0.01,
    )
    rollback_exact = None
    if not report["accepted"]:
        rollback_exact = (
            candidate_opt is old_opt
            and equal_state(candidate.state_dict(), state_before)
            and equal_state(candidate_opt.state_dict(), opt_before)
            and len(refs) == len(list(candidate.parameters()))
            and all(a is b for a, b in zip(refs, candidate.parameters()))
            and torch.equal(torch.get_rng_state(), rng_before)
            and torch.equal(generator.get_state(), generator_before)
        )
        assert rollback_exact
    active_after = evaluate(candidate, probe_x, probe_y)
    warmup = LrWarmup(candidate_opt, steps=10) if report["accepted"] else None
    for x, y in continuation:
        train_step(candidate, candidate_opt, x, y)
        if warmup is not None:
            warmup.step()
    results.append(
        {
            "case": name,
            "report": report,
            "rollback_exact": rollback_exact,
            "active_width": candidate.d,
            "active_layers": len(candidate.blocks),
            "parameters_before": count_before,
            "parameters_after": sum(p.numel() for p in candidate.parameters()),
            "active_immediately_after": active_after,
            "after_10_training_steps": evaluate(candidate, probe_x, probe_y),
        }
    )

print(json.dumps({"device": "cpu", "seed": 20260907, "initial": initial, "after_60_steps": baseline, "results": results}, indent=2))
