"""Benchmark preespecificado; ver PROTOCOL.md antes de interpretar resultados."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset

from mztrain.elastic_shape import GPT, GrowthEvent, LrWarmup, apply_event, dense_lin, fact_lin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ARROW = Path(
    r"C:\Users\Raul\.cache\huggingface\datasets\wikitext\wikitext-2-raw-v1\0.0.0\b08601e04326c79dfdd32d625aee71d232d685c3"
)
SEEDS = (17, 29, 43)
BATCH, SEQ, LR = 16, 128, 3e-4
STEPS_R, STEP_GROW, STEPS_M, EVAL_EVERY = 4000, 2000, 6000, 250
START = time.perf_counter()
RESULTS = {"protocol_sha256": hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(), "runs": []}


def emit(message):
    print(json.dumps(message, ensure_ascii=True), flush=True)


def save():
    target = HERE / "results.json"
    temp = HERE / "results.partial.json"
    temp.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=True), encoding="utf-8")
    temp.replace(target)


def sync():
    torch.cuda.synchronize()


def elapsed(start):
    sync()
    return time.perf_counter() - start


def check_deadline():
    if time.perf_counter() - START > 45 * 60:
        raise TimeoutError("Presupuesto global de45min agotado; no se modifica el protocolo")


def load_data():
    texts = {}
    hashes = {}
    for split in ("train", "validation"):
        path = ARROW / f"wikitext-{split}.arrow"
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        texts[split] = "".join(Dataset.from_file(str(path))["text"])
    stoi = {char: i + 1 for i, char in enumerate(sorted(set(texts["train"])))}
    train = torch.tensor([stoi[c] for c in texts["train"]], dtype=torch.long, device="cuda")
    val = torch.tensor([stoi.get(c, 0) for c in texts["validation"]], dtype=torch.long, device="cuda")
    calibration = train[:8193].clone()
    train = train[8193:].clone()
    RESULTS["data"] = {
        "corpus": "wikitext-2-raw-v1 (local Arrow cache)",
        "arrow_sha256": hashes,
        "train_tokens": train.numel(),
        "validation_tokens": val.numel(),
        "calibration_tokens": calibration.numel(),
        "vocab": len(stoi) + 1,
        "validation_unknown_chars": int((val == 0).sum()),
    }
    return train, val, calibration, len(stoi) + 1


def make_opt(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, foreach=True)


def step_train(model, opt, data, indices, warmup=None):
    tokens = data[indices[:, None] + torch.arange(SEQ + 1, device="cuda")]
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    if warmup is not None:
        warmup.step()
    value = float(loss.detach())
    if not math.isfinite(value):
        raise FloatingPointError("loss no finita")
    return value


@torch.no_grad()
def evaluate(model, val):
    model.eval()
    windows = (val.numel() - 1) // SEQ
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    offsets = torch.arange(SEQ + 1, device="cuda")
    for start in range(0, windows, 32):
        positions = torch.arange(start, min(start + 32, windows), device="cuda") * SEQ
        tokens = val[positions[:, None] + offsets]
        x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        logits, _ = model(x)
        losses = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1), reduction="none")
        total += losses.double().sum()
    bpc = float(total / (windows * SEQ * math.log(2)))
    if not math.isfinite(bpc):
        raise FloatingPointError("BPC no finito")
    model.train()
    return bpc


def layout(model):
    return {name: list(p.shape) for name, p in model.named_parameters()}


def add_point(run, model, val, wall_start, step, stage):
    start = time.perf_counter()
    bpc = evaluate(model, val)
    run["evaluation_s"] += elapsed(start)
    point = {
        "step": step,
        "stage": stage,
        "width": model.d,
        "wall_s": elapsed(wall_start),
        "train_s": run["train_s"],
        "bpc": bpc,
    }
    run["curve"].append(point)
    emit({"kind": run["kind"], "seed": run["seed"], **point})
    save()


def warm_device(vocab, train):
    # Calentamiento compartido de ambas arquitecturas; pesos descartados.
    for d, factory in ((96, dense_lin), (192, fact_lin(96))):
        torch.manual_seed(900)
        m = GPT(vocab, SEQ, d, 4, 4, factory).cuda()
        o = make_opt(m)
        ix = torch.arange(BATCH, device="cuda") * SEQ
        sync()
        start = time.perf_counter()
        for _ in range(10):
            step_train(m, o, train, ix)
        seconds = elapsed(start)
        emit({"warmup_only": True, "width": d, "ten_steps_s": seconds})
        del m, o
        gc.collect()
        torch.cuda.empty_cache()


def run_arm(kind, seed, train, val, calibration, vocab, batch_indices):
    check_deadline()
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    sync()
    wall_start = time.perf_counter()
    model = GPT(vocab, SEQ, 192 if kind == "R" else 96, 4, 4, fact_lin(96) if kind == "R" else dense_lin).cuda()
    opt = make_opt(model)
    run = {
        "kind": kind,
        "seed": seed,
        "setup_s": elapsed(wall_start),
        "train_s": 0.0,
        "evaluation_s": 0.0,
        "surgery_s": 0.0,
        "params_start": sum(p.numel() for p in model.parameters()),
        "curve": [],
        "completed": False,
    }
    RESULTS["runs"].append(run)
    add_point(run, model, val, wall_start, 0, "initial")
    warmup = None
    limit = STEPS_R if kind == "R" else STEPS_M
    for begin in range(0, limit, EVAL_EVERY):
        check_deadline()
        sync()
        start = time.perf_counter()
        model.train()
        for step in range(begin, begin + EVAL_EVERY):
            last_loss = step_train(model, opt, train, batch_indices[step], warmup)
        opt.zero_grad(set_to_none=True)
        run["train_s"] += elapsed(start)
        done = begin + EVAL_EVERY
        add_point(run, model, val, wall_start, done, "trained")
        if kind == "M" and done == STEP_GROW:
            sync()
            start = time.perf_counter()
            token_indices = torch.arange(8, device="cuda")[:, None] * SEQ + torch.arange(SEQ + 1, device="cuda")
            tokens = calibration[token_indices]
            gx, gy = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            opt, report = apply_event(
                model,
                opt,
                GrowthEvent(step=done, factorize=True, new_d=192, noise_scale=0.001),
                probe_x=gx,
                probe_targets=gy,
                generator=torch.Generator(device="cuda").manual_seed(4242 + seed),
                max_loss_increase=0.05,
                max_kl=0.05,
            )
            if report["accepted"]:
                warmup = LrWarmup(opt, steps=200, floor=0.1)
            run["surgery_s"] += elapsed(start)
            run["surgery_report"] = report
            emit({"event": "growth", "seed": seed, "surgery_s": run["surgery_s"], **report})
            add_point(run, model, val, wall_start, done, "after_surgery")
            if not report["accepted"]:
                run["stopped_reason"] = "growth_rejected"
                break
    run.update(
        completed=True,
        params_end=sum(p.numel() for p in model.parameters()),
        layout_end=layout(model),
        final_bpc=run["curve"][-1]["bpc"],
        last_train_loss=last_loss,
        wall_s=elapsed(wall_start),
        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
    )
    save()
    emit({"event": "run_done", "kind": kind, "seed": seed, "wall_s": run["wall_s"], "final_bpc": run["final_bpc"]})
    del model, opt, warmup
    gc.collect()
    torch.cuda.empty_cache()


def crossing(run, target, require_final):
    curve = run["curve"]
    for i, point in enumerate(curve):
        if (not require_final or point["width"] == 192) and point["bpc"] <= target:
            # Antes de la cirugia M no cumple la arquitectura requerida.
            # Usar conservadoramente el ultimo punto medido anterior.
            lower = curve[i - 1]["wall_s"] if i else 0.0
            return {"point": point, "previous_observation_wall_s": lower}
    return None


def summarize():
    pairs, ratios = [], []
    valid = True
    for seed in SEEDS:
        ref = next(r for r in RESULTS["runs"] if r["seed"] == seed and r["kind"] == "R")
        morph = next(r for r in RESULTS["runs"] if r["seed"] == seed and r["kind"] == "M")
        q = ref["final_bpc"]
        r_hit, m_hit = crossing(ref, q, False), crossing(morph, q, True)
        sufficient = q <= 3.8 and ref["curve"][0]["bpc"] - q >= 1
        same_layout = ref["layout_end"] == morph["layout_end"]
        accepted = morph["surgery_report"]["accepted"]
        ratio = m_hit["point"]["wall_s"] / r_hit["point"]["wall_s"] if m_hit else None
        valid = valid and sufficient and same_layout and accepted and ratio is not None
        if ratio is not None:
            ratios.append(ratio)
        pairs.append(
            {
                "seed": seed,
                "target_bpc": q,
                "reference_sufficient": sufficient,
                "same_layout": same_layout,
                "growth_accepted": accepted,
                "R_hit": r_hit,
                "M_hit": m_hit,
                "ratio": ratio,
            }
        )
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if len(ratios) == len(SEEDS) else None
    favorable = bool(valid and all(r < 1 for r in ratios) and geo <= 0.90)
    RESULTS["summary"] = {
        "pairs": pairs,
        "geometric_mean_ratio": geo,
        "local_savings_supported": favorable,
        "scope": "Esta escala/configuracion, tres semillas y tiempo hasta calidad; no calidad asintotica",
    }
    save()
    emit({"event": "summary", **RESULTS["summary"]})


def main():
    if (HERE / "results.json").exists():
        raise FileExistsError("No sobrescribir una corrida existente")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    RESULTS["environment"] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "precision": "fp32 TF32 off",
    }
    RESULTS["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("src/mztrain/elastic_shape.py", "src/mztrain/shape_ops.py", "src/mztrain/layers.py")
    }
    RESULTS["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    train, val, calibration, vocab = load_data()
    emit({"event": "data_ready", **RESULTS["data"], **RESULTS["environment"]})
    warm_device(vocab, train)
    for index, seed in enumerate(SEEDS):
        generator = torch.Generator(device="cuda").manual_seed(1000 + seed)
        indices = torch.randint(train.numel() - SEQ, (STEPS_M, BATCH), generator=generator, device="cuda")
        for kind in (("R", "M") if index % 2 == 0 else ("M", "R")):
            run_arm(kind, seed, train, val, calibration, vocab, indices)
    summarize()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Conservar corridas parciales: nunca borrar una observacion desfavorable.
        if RESULTS["runs"]:
            RESULTS["error"] = repr(error)
            save()
        raise
