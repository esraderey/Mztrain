"""Calidad final operativa; protocolo previo en PROTOCOL.md."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset

from mztrain.elastic_shape import GPT, GrowthEvent, LrWarmup, apply_event, dense_lin, fact_lin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE_PATH = HERE.parent / "elasticshape-evidence-20260907" / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("evidence_base", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
SEEDS = (17, 29, 43)
CAP, EVAL_EVERY, MIN_LR = 40000, 1000, 3e-5
START = time.perf_counter()
RESULTS = {"runs": []}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(item):
    print(json.dumps(item), flush=True)


def save():
    temp = HERE / "results.partial.json"
    temp.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    temp.replace(HERE / "results.json")


def check_deadline():
    if time.perf_counter() - START > 45 * 60:
        raise TimeoutError("Limite global de45min; preservar lo medido, sin declarar convergencia")


def build(vocab, kind):
    return GPT(vocab, 128, 64 if kind == "R" else 32, 2, 4, fact_lin(32) if kind == "R" else dense_lin).cuda()


def plateau_scheduler(opt):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2, threshold=0.005, threshold_mode="abs", min_lr=MIN_LR
    )


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def checkpoint(path, obj):
    temp = path.with_suffix(".partial.pt")
    torch.save(obj, temp)
    temp.replace(path)


def improvement_slope(curve):
    # Usar puntos entrenados a cadencia regular, no duplicar la cirugia.
    points = [p for p in curve if p["stage"] == "trained" and p["width"] == 64][-5:]
    if len(points) < 5:
        return None
    xs = [p["step"] for p in points]
    ys = [p["val_bpc"] for p in points]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return -500 * sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


def observe(run, model, val, opt, start, step, stage):
    bpc = base.evaluate(model, val)
    point = {
        "step": step,
        "stage": stage,
        "width": model.d,
        "wall_s": base.elapsed(start),
        "val_bpc": bpc,
        "lr": float(opt.param_groups[0]["lr"]),
    }
    run["curve"].append(point)
    if model.d == 64 and (run["best_val_bpc"] is None or bpc < run["best_val_bpc"]):
        run["best_val_bpc"], run["best_step"] = bpc, step
        checkpoint(
            HERE / run["best_checkpoint"],
            {"model": cpu_state(model), "step": step, "val_bpc": bpc, "layout": base.layout(model), "vocab": model.vocab},
        )
    emit({"kind": run["kind"], "seed": run["seed"], **point, "best_step": run.get("best_step")})
    save()
    return bpc


def run_arm(kind, seed, train, val, calibration, vocab, indices):
    gc.collect()
    torch.cuda.empty_cache()
    check_deadline()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    base.sync()
    start = time.perf_counter()
    model = build(vocab, kind)
    opt = base.make_opt(model)
    scheduler = plateau_scheduler(opt)
    warmup = None
    run = {
        "kind": kind,
        "seed": seed,
        "curve": [],
        "completed": False,
        "best_val_bpc": None,
        "best_checkpoint": f"best_{kind}_{seed}.pt",
        "params_start": sum(p.numel() for p in model.parameters()),
        "lr_changes": [],
    }
    RESULTS["runs"].append(run)
    observe(run, model, val, opt, start, 0, "initial")
    anchor, last_significant, min_lr_since = float("inf"), 0, None
    reason = "budget_cap"
    for begin in range(0, CAP, EVAL_EVERY):
        check_deadline()
        model.train()
        for step in range(begin, begin + EVAL_EVERY):
            last_train_loss = base.step_train(model, opt, train, indices[step], warmup)
        opt.zero_grad(set_to_none=True)
        done = begin + EVAL_EVERY
        bpc = observe(run, model, val, opt, start, done, "trained")
        if kind == "M" and done == 2000:
            tokens = calibration[torch.arange(8, device="cuda")[:, None] * 128 + torch.arange(129, device="cuda")]
            surgery_start = time.perf_counter()
            opt, report = apply_event(
                model,
                opt,
                GrowthEvent(step=done, factorize=True, new_d=64, noise_scale=0.001),
                probe_x=tokens[:, :-1].contiguous(),
                probe_targets=tokens[:, 1:].contiguous(),
                generator=torch.Generator(device="cuda").manual_seed(4242 + seed),
                max_loss_increase=0.05,
                max_kl=0.05,
            )
            run["surgery_s"] = base.elapsed(surgery_start)
            run["surgery_report"] = report
            emit({"event": "growth", "seed": seed, **report})
            if not report["accepted"]:
                reason = "growth_rejected"
                break
            warmup = LrWarmup(opt, steps=200, floor=0.1)
            scheduler = plateau_scheduler(opt)
            bpc = observe(run, model, val, opt, start, done, "after_surgery")
        if model.d != 64:
            continue
        if bpc < anchor - 0.005:
            anchor, last_significant = bpc, done
        if done >= 6000:
            lr_before = opt.param_groups[0]["lr"]
            scheduler.step(bpc)
            lr_after = opt.param_groups[0]["lr"]
            if lr_after != lr_before:
                change = {"step": done, "before": lr_before, "after": lr_after}
                run["lr_changes"].append(change)
                emit({"event": "lr_reduced", "kind": kind, "seed": seed, **change})
            if lr_after <= MIN_LR * (1 + 1e-8) and min_lr_since is None:
                min_lr_since = done
        slope = improvement_slope(run["curve"])
        settled = (
            done >= 12000 and min_lr_since is not None and done - min_lr_since >= 3000 and done - last_significant >= 6000
        )
        if settled and slope is not None:
            if abs(slope) < 0.005:
                reason = "validation_plateau"
                break
            if bpc >= run["best_val_bpc"] + 0.02:
                reason = "validation_overfit"
                break
    run.update(
        completed=True,
        stop_reason=reason,
        final_step=done,
        final_val_bpc=bpc,
        final_lr=float(opt.param_groups[0]["lr"]),
        last_significant_step=last_significant,
        min_lr_since=min_lr_since,
        final_improvement_bpc_per_500=improvement_slope(run["curve"]),
        params_end=sum(p.numel() for p in model.parameters()),
        layout_end=base.layout(model),
        wall_s=base.elapsed(start),
        last_train_loss=last_train_loss,
    )
    checkpoint(
        HERE / f"final_{kind}_{seed}.pt",
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": done,
            "seed": seed,
            "kind": kind,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "indices": indices.cpu(),
            "run": run,
        },
    )
    if (HERE / run["best_checkpoint"]).exists():
        run["best_checkpoint_sha256"] = sha(HERE / run["best_checkpoint"])
    save()
    emit(
        {
            "event": "run_done",
            **{
                k: run[k]
                for k in ("kind", "seed", "stop_reason", "final_step", "best_val_bpc", "best_step", "wall_s")
                if k in run
            },
        }
    )
    del model, opt, scheduler, warmup
    gc.collect()
    torch.cuda.empty_cache()


def load_test():
    train_text = "".join(Dataset.from_file(str(base.ARROW / "wikitext-train.arrow"))["text"])
    stoi = {char: i + 1 for i, char in enumerate(sorted(set(train_text)))}
    path = base.ARROW / "wikitext-test.arrow"
    text = "".join(Dataset.from_file(str(path))["text"])
    test = torch.tensor([stoi.get(c, 0) for c in text], dtype=torch.long, device="cuda")
    RESULTS["test_data"] = {"sha256": sha(path), "tokens": test.numel(), "unknown_chars": int((test == 0).sum())}
    return test


@torch.no_grad()
def test_metrics(model, data):
    model.eval()
    windows = (data.numel() - 1) // 128
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    correct = torch.zeros((), device="cuda", dtype=torch.long)
    offsets = torch.arange(129, device="cuda")
    for start in range(0, windows, 32):
        positions = torch.arange(start, min(start + 32, windows), device="cuda") * 128
        tokens = data[positions[:, None] + offsets]
        x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        logits, _ = model(x)
        losses = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1), reduction="none")
        total += losses.double().sum()
        correct += (logits.argmax(-1) == y).sum()
    n = windows * 128
    bpc = float(total / (n * math.log(2)))
    assert math.isfinite(bpc)
    return {"bpc": bpc, "character_perplexity": 2**bpc, "next_character_accuracy": float(correct / n), "evaluated_tokens": n}


def final_evaluation(vocab):
    assert len(RESULTS["runs"]) == 6 and all(run["completed"] for run in RESULTS["runs"])
    # Se bloquean y verifican todos los checkpoints ANTES de abrir el test.
    selected = [run for run in RESULTS["runs"] if "best_checkpoint_sha256" in run]
    for run in selected:
        assert sha(HERE / run["best_checkpoint"]) == run["best_checkpoint_sha256"]
    RESULTS["checkpoint_selection_locked_before_test"] = True
    save()
    test = load_test()
    for run in selected:
        blob = torch.load(HERE / run["best_checkpoint"], map_location="cpu", weights_only=True)
        model = build(vocab, "R")
        model.load_state_dict(blob["model"])
        assert base.layout(model) == blob["layout"]
        run["test"] = test_metrics(model, test)
        emit(
            {"event": "test_once", "kind": run["kind"], "seed": run["seed"], "selected_step": run["best_step"], **run["test"]}
        )
        save()
        del model, blob
        torch.cuda.empty_cache()
    pairs = []
    for seed in SEEDS:
        r = next(run for run in RESULTS["runs"] if run["kind"] == "R" and run["seed"] == seed)
        m = next(run for run in RESULTS["runs"] if run["kind"] == "M" and run["seed"] == seed)
        valid = (
            r["stop_reason"] in ("validation_plateau", "validation_overfit")
            and m["stop_reason"] in ("validation_plateau", "validation_overfit")
            and m["surgery_report"]["accepted"]
            and r["layout_end"] == m["layout_end"]
        )
        delta = m["test"]["bpc"] - r["test"]["bpc"] if "test" in m and "test" in r else None
        pairs.append({"seed": seed, "final_quality_eligible": valid, "delta_test_bpc_M_minus_R": delta})
    deltas = [p["delta_test_bpc_M_minus_R"] for p in pairs]
    mean = sum(deltas) / 3 if all(d is not None for d in deltas) else None
    verdict = "mixed_or_inconclusive"
    if all(p["final_quality_eligible"] for p in pairs):
        if all(d < 0 for d in deltas) and mean <= -0.02:
            verdict = "M_consistently_better_in_this_sample"
        elif all(d > 0 for d in deltas) and mean >= 0.02:
            verdict = "M_consistently_worse_in_this_sample"
        elif all(abs(d) <= 0.02 for d in deltas):
            verdict = "similar_within_descriptive_tolerance"
    RESULTS["summary"] = {"pairs": pairs, "mean_delta_test_bpc": mean, "verdict": verdict}
    save()
    emit({"event": "summary", **RESULTS["summary"]})


def main():
    if (HERE / "results.json").exists():
        raise FileExistsError("Conservar el experimento existente")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    RESULTS["environment"] = {
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "precision": "fp32 TF32 off",
    }
    RESULTS["protocol_sha256"] = sha(HERE / "PROTOCOL.md")
    RESULTS["harness_sha256"] = sha(__file__)
    RESULTS["base_harness_sha256"] = sha(BASE_PATH)
    RESULTS["source_sha256"] = {
        name: sha(ROOT / name)
        for name in ("src/mztrain/elastic_shape.py", "src/mztrain/shape_ops.py", "src/mztrain/layers.py")
    }
    train, val, calibration, vocab = base.load_data()
    RESULTS["data"] = base.RESULTS["data"]
    emit({"event": "data_ready", **RESULTS["data"]})
    # Calentamiento compartido; no se conservan pesos ni momentos.
    for kind in ("R", "M"):
        torch.manual_seed(900)
        model = build(vocab, kind)
        opt = base.make_opt(model)
        start = time.perf_counter()
        for _ in range(20):
            base.step_train(model, opt, train, torch.arange(16, device="cuda") * 128)
        emit(
            {
                "warmup_only": True,
                "kind": kind,
                "twenty_steps_s": base.elapsed(start),
                "params": sum(p.numel() for p in model.parameters()),
            }
        )
        del model, opt
    for i, seed in enumerate(SEEDS):
        generator = torch.Generator(device="cuda").manual_seed(1000 + seed)
        indices = torch.randint(train.numel() - 128, (CAP, 16), generator=generator, device="cuda")
        for kind in (("R", "M") if i % 2 == 0 else ("M", "R")):
            run_arm(kind, seed, train, val, calibration, vocab, indices)
    final_evaluation(vocab)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if RESULTS["runs"]:
            RESULTS["error"] = repr(error)
            save()
        raise
