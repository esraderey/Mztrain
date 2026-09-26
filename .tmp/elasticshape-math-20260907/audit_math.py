"""Independent mathematical checks of ElasticShape; no training or file writes.

Run from D:/mztrain:
  .venv/Scripts/python.exe .tmp/elasticshape-math-20260907/audit_math.py
These assertions verify equations and reproduce counterexamples, not a claim
that all behaviors checked here are desirable.
"""

import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn.functional as F

from mztrain.elastic_shape import GPT, attn_in_map, deepen_gpt, fact_lin, qkv_out_map
from mztrain.layers import ZFactorizedLinear
from mztrain.shape_ops import (
    dense_to_factorized,
    scale_output_rows,
    widen_factorized_linear,
    widen_layernorm,
)


def layernorm_checks():
    d, width = 4, 8
    rho = d / width
    x = torch.tensor([[1., 2., 3., 4.], [101., 102., 103., 104.]], dtype=torch.float64)
    ln = torch.nn.LayerNorm(d).double()
    wide = widen_layernorm(ln, width, variance_compensation=True)
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    formula = (x - rho * mu) / torch.sqrt(var + (1 - rho) * mu.square() + ln.eps / rho)
    old = ln(x).detach()
    new = wide(F.pad(x, (0, width - d))).detach()[:, :d]
    torch.testing.assert_close(new, formula, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(old[0], old[1], atol=0, rtol=0)
    constant = torch.ones(1, d, dtype=torch.float64)
    torch.testing.assert_close(ln(constant), torch.zeros_like(constant), atol=0, rtol=0)
    assert wide(F.pad(constant, (0, width - d)))[:, :d].norm().item() > 1
    result = {
        "formula_max_error": (new - formula).abs().max().item(),
        "relative_errors": [(new[k] - old[k]).norm().item() / old[k].norm().item() for k in range(2)],
        "constant_new_prefix": wide(F.pad(constant, (0, width - d))).detach()[:, :d].tolist(),
    }
    x = torch.randn(5, d, dtype=torch.float64)
    x = x - x.mean(-1, keepdim=True)
    ln = torch.nn.LayerNorm(d, eps=0).double()
    wide = widen_layernorm(ln, width, variance_compensation=True)
    readout = torch.randn_like(x)
    (ln(x) * readout).sum().backward()
    (wide(F.pad(x, (0, width - d)))[:, :d] * readout).sum().backward()
    torch.testing.assert_close(wide.weight.grad[:d], ln.weight.grad / math.sqrt(rho))
    result["gamma_gradient_scale"] = (wide.weight.grad[:d] / ln.weight.grad).tolist()
    return result


def noise_gauge_checks():
    layer = ZFactorizedLinear(4, 4, rank=2, bias=False, init_method="random").double()
    with torch.no_grad():
        layer.U.normal_()
        layer.V.normal_()
        layer.S.fill_(1)
    scaled = copy.deepcopy(layer)
    with torch.no_grad():
        scaled.U[:, 0] *= 1e6
        scaled.S[0] /= 1e6
    x = torch.randn(16, 4, dtype=torch.float64)
    old = layer(x).detach()
    other = scaled(x).detach()
    torch.testing.assert_close(other, old, atol=1e-12, rtol=1e-12)
    result = {"original_map_rel_difference": (other - old).norm().item() / old.norm().item()}
    for name, base in [("balanced", layer), ("rescaled_same_map", scaled)]:
        new = widen_factorized_linear(base, 4, 8, noise_scale=1e-3,
                                      generator=torch.Generator().manual_seed(1))
        output = new(x).detach()
        torch.testing.assert_close(output[:, :4], old, atol=1e-12, rtol=1e-12)
        result[name] = output[:, 4:].norm().item() / old.norm().item()
    assert result["rescaled_same_map"] > 100
    assert result["balanced"] < .01
    return result


def attention_checks():
    torch.manual_seed(17)
    d, width, heads, rank = 8, 12, 2, 5
    old = ZFactorizedLinear(d, 3 * d, rank=rank, bias=False, init_method="random").double()
    with torch.no_grad():
        old.U.normal_(std=.4)
        old.V.normal_(std=.4)
    rows = qkv_out_map(d, width, heads)
    cols = attn_in_map(d, width, heads)
    new = widen_factorized_linear(old, width, 3 * width, out_map=rows)
    c = math.sqrt(width / d)
    scale_output_rows(new, torch.arange(width), c)
    x = torch.randn(2, 4, d, dtype=torch.float64)
    weights = torch.randn_like(x)

    def attn(layer, inputs, size):
        qkv = layer(inputs).view(2, 4, 3, heads, size // heads).permute(2, 0, 3, 1, 4)
        return F.scaled_dot_product_attention(*qkv, is_causal=True).transpose(1, 2).reshape(2, 4, size)

    a = attn(old, x, d)
    b = attn(new, F.pad(x, (0, width - d)), width)[:, :, cols]
    (a * weights).sum().backward()
    (b * weights).sum().backward()
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(new.U.grad[rows[:d]], old.U.grad[:d] / c, atol=1e-12, rtol=1e-12)
    return {
        "relative_output_error": (a.detach() - b.detach()).norm().item() / a.detach().norm().item(),
        "q_gradient_rescale_error": (new.U.grad[rows[:d]] - old.U.grad[:d] / c).abs().max().item(),
    }


def adam_checks():
    result = {}
    b1, b2 = .9, .999
    for age in [1000, 100000]:
        ratios = [(1 - b1**n) / (1 - b1**(age + n)) * math.sqrt((1 - b2**(age + n)) / (1 - b2**n))
                  for n in range(1, 201)]
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        opt = torch.optim.AdamW([p], lr=.001, weight_decay=0, eps=1e-12)
        opt.state[p] = {"step": torch.tensor(float(age)), "exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
        observed = []
        for n in range(20):
            before = p.detach().clone()
            p.grad = torch.ones_like(p)
            opt.step()
            observed.append(((before - p.detach()) / .001).item())
            assert math.isclose(observed[-1], ratios[n], rel_tol=1e-9)
        result[str(age)] = {"first_lr_units": ratios[0], "peak_lr_units": max(ratios),
                            "peak_step": ratios.index(max(ratios)) + 1,
                            "warmup200_peak_base_lr_units": max(value * (.1 + .9 * (n - 1) / 200)
                                                                 for n, value in enumerate(ratios, 1))}
    c = math.sqrt(2)
    p1 = torch.nn.Parameter(torch.tensor([.7], dtype=torch.float64))
    p2 = torch.nn.Parameter(c * p1.detach())
    opts = []
    before = [p1.detach().clone(), p2.detach().clone()]
    for p, scale in [(p1, 1), (p2, c)]:
        opt = torch.optim.AdamW([p], lr=.01, weight_decay=0, eps=1e-14)
        opt.state[p] = {"step": torch.tensor(100.), "exp_avg": torch.tensor([.3 / scale], dtype=torch.float64),
                        "exp_avg_sq": torch.tensor([.2 / scale**2], dtype=torch.float64)}
        ((p / scale).square() / 2).backward()
        opts.append(opt)
    for opt in opts:
        opt.step()
    delta1, delta2 = (before[0] - p1.detach()).item(), (before[1] - p2.detach()).item()
    assert math.isclose(delta2 / delta1, 1, rel_tol=1e-10)
    result["coordinate_change"] = {"parameter_update_ratio": delta2 / delta1,
                                   "effective_function_update_ratio": delta2 / c / delta1}
    return result


def decay_and_depth_checks():
    torch.manual_seed(18)
    linear = torch.nn.Linear(4, 6, bias=False)
    fact = dense_to_factorized(linear)
    expected = [1 - .1 * .2, (1 - .1 * .2)**3]
    measured = []
    for layer in [linear, fact]:
        weight = layer.weight if isinstance(layer, torch.nn.Linear) else layer.reconstruct_weight()
        before = weight.detach().norm().item()
        opt = torch.optim.AdamW(layer.parameters(), lr=.1, weight_decay=.2)
        for p in layer.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
        weight = layer.weight if isinstance(layer, torch.nn.Linear) else layer.reconstruct_weight()
        measured.append(weight.detach().norm().item() / before)
    for a, b in zip(measured, expected):
        assert math.isclose(a, b, rel_tol=1e-6)
    model = GPT(32, 8, 8, 1, 2, fact_lin(4))
    x, y = torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8))
    before = model(x)[0].detach()
    deepen_gpt(model, 1)
    torch.testing.assert_close(model(x)[0], before, atol=0, rtol=0)
    model(x, y)[1].backward()
    gradients = {name: p.grad.norm().item() for name, p in model.blocks[-1].named_parameters()}
    assert gradients["proj.U"] > 0 and gradients["fc2.U"] > 0
    assert all(value == 0 for name, value in gradients.items() if name not in ["proj.U", "fc2.U"])
    return {"decay_dense_factorized": measured, "identity_block_first_gradients": gradients}


def logit_metric_checks():
    old = torch.tensor([[1000., 1001.]], dtype=torch.float64)
    new = torch.tensor([[1001., 1000.]], dtype=torch.float64)
    target = torch.tensor([0])
    drift = ((new - old).norm() / old.norm()).item()
    loss_change = (F.cross_entropy(new, target) - F.cross_entropy(old, target)).abs().item()
    assert drift < .001
    assert math.isclose(loss_change, 1, abs_tol=1e-12)
    return {"relative_logit_drift": drift, "absolute_loss_change_nats": loss_change}


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(20260907)
    results = {"torch": torch.__version__, "layernorm": layernorm_checks(),
               "noise_gauge": noise_gauge_checks(), "attention": attention_checks(),
               "adam": adam_checks(), "decay_and_depth": decay_and_depth_checks(),
               "logit_metric": logit_metric_checks()}
    print(json.dumps(results, indent=2))
