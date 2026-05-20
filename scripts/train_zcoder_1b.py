"""Smoke-train a 1B-equivalent compressed ZCoder model.

This script intentionally keeps the dense 1B model implicit. The transformer
linear layers are created as MZTrain low-rank factors, so the trainable model is
much smaller while reporting the dense equivalent parameter count.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from mztrain import ZActivationCheckpoint, ZCompressedAdam
from mztrain.checkpoint import HAS_MNEME as HAS_MNEME_CHECKPOINT
from mztrain.data import CausalCodeDataset, CodeDataset, CodeTokenizer, SyntheticCodeGenerator
from mztrain.models import ZCodeBERTConfig, ZCodeBERTForCausalLM


CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "htmlcov",
    "build",
    "dist",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", action="append", default=["src", "tests"])
    parser.add_argument("--max-files", type=int, default=240)
    parser.add_argument("--lines-per-sample", type=int, default=80)
    parser.add_argument("--line-stride", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--abort-loss", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--save-path", default="zcoder_1b_mztrain_mneme_smoke.pt")
    parser.add_argument("--metrics-path", default="zcoder_1b_mztrain_mneme_metrics.json")
    parser.add_argument("--mneme-storage-dir", default="mneme_storage_zcoder1b")
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--disable-activation-checkpointing", action="store_true")
    parser.add_argument("--synthetic-fallback", type=int, default=64)
    parser.add_argument("--mneme-log-level", default="WARNING")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.WARNING)
    logging.getLogger("mneme").setLevel(level)
    logging.getLogger("mneme.mneme_core").setLevel(level)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio --device cuda, pero torch.cuda.is_available() es False")
    return torch.device(requested)


def iter_code_files(roots: Iterable[str], max_files: int) -> list[Path]:
    files: list[Path] = []
    for root_text in roots:
        root = Path(root_text).resolve()
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in CODE_EXTENSIONS:
            files.append(root)
            continue
        for path in root.rglob("*"):
            if len(files) >= max_files:
                return files
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in CODE_EXTENSIONS:
                files.append(path)
    return files[:max_files]


def chunk_text(text: str, lines_per_sample: int, stride: int) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    if len(lines) <= lines_per_sample:
        return [text]

    chunks = []
    step = max(1, stride)
    for start in range(0, len(lines), step):
        window = lines[start:start + lines_per_sample]
        if not window:
            break
        chunk = "\n".join(window).strip()
        if len(chunk) >= 24:
            chunks.append(chunk)
        if start + lines_per_sample >= len(lines):
            break
    return chunks


def load_real_code_snippets(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    files = iter_code_files(args.corpus_root, args.max_files)
    snippets: list[str] = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        snippets.extend(chunk_text(text, args.lines_per_sample, args.line_stride))

    if not snippets:
        generator = SyntheticCodeGenerator(seed=args.seed)
        snippets = generator.generate(args.synthetic_fallback)

    return snippets, [str(p) for p in files]


def make_config(args: argparse.Namespace) -> ZCodeBERTConfig:
    config = ZCodeBERTConfig.coder_1b(rank=args.rank)
    config.vocab_size = args.vocab_size
    config.max_position_embeddings = max(512, args.seq_len)
    config.use_activation_checkpointing = not args.disable_activation_checkpointing
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    return config


def move_batch(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device):
    return tuple(t.to(device, non_blocking=True) for t in batch)


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


@torch.no_grad()
def evaluate_loss(
    model: ZCodeBERTForCausalLM,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    batches: int,
) -> float | None:
    if batches <= 0:
        return None

    model.eval()
    losses: list[float] = []
    for idx, batch in enumerate(loader):
        if idx >= batches:
            break
        input_ids, attention_mask, labels = move_batch(batch, device)
        with autocast_context(device, amp):
            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
        if loss is not None and torch.isfinite(loss):
            losses.append(float(loss.detach().cpu()))

    model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def gpu_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def save_metrics_with_mneme(metrics: dict, storage_dir: str) -> dict:
    try:
        from mneme import SecureStorageBackend, StorageConfig

        storage = SecureStorageBackend(
            StorageConfig(
                storage_path=storage_dir,
                enable_compression=True,
                max_file_size_mb=16,
            )
        )
        payload = json.dumps(metrics, indent=2, sort_keys=True).encode("utf-8")
        ok = storage.store("zcoder1b_metrics", payload, metadata={"kind": "metrics"})
        return {"available": True, "stored": bool(ok), "path": str(Path(storage_dir).resolve())}
    except Exception as exc:  # pragma: no cover - depends on optional MNEME runtime
        return {"available": False, "stored": False, "error": repr(exc)}


def save_checkpoint(path: str, model, config, metrics: dict, optimizer=None) -> None:
    state = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config.to_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, path)


def write_metrics(path: str, metrics: dict) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def add_checkpoint_stats(total: dict[str, int], step_stats: dict) -> None:
    for key in ("total_compressed", "total_decompressed", "bytes_saved"):
        total[key] = total.get(key, 0) + int(step_stats.get(key, 0))


def main() -> None:
    args = parse_args()
    configure_logging(args.mneme_log_level)
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    device = resolve_device(args.device)
    snippets, files = load_real_code_snippets(args)
    tokenizer = CodeTokenizer(vocab_size=args.vocab_size, max_length=args.seq_len)
    code_dataset = CodeDataset(snippets, tokenizer, max_length=args.seq_len)
    causal_dataset = CausalCodeDataset(code_dataset)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        causal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    config = make_config(args)
    model = ZCodeBERTForCausalLM(config).to(device)
    optimizer = ZCompressedAdam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        compress_states=True,
        compression_interval=1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp == "fp16")

    stats = model.get_model_stats()
    ZActivationCheckpoint.reset()

    started = time.perf_counter()
    initial_loss = evaluate_loss(model, loader, device, args.amp, args.eval_batches)

    train_losses: list[float] = []
    history: list[dict] = []
    checkpoint_cumulative = {
        "total_compressed": 0,
        "total_decompressed": 0,
        "bytes_saved": 0,
    }
    metrics = {
        "model": "ZCodeBERTForCausalLM",
        "scale": "1B_dense_equivalent_compressed",
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "amp": args.amp,
        "config": config.to_dict(),
        "model_stats": stats,
        "dataset": {
            "snippets": len(snippets),
            "files": len(files),
            "roots": args.corpus_root,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
        },
        "loss": {
            "initial_eval": initial_loss,
            "train_steps": train_losses,
            "final_eval": None,
            "history": history,
        },
        "status": "running",
    }
    write_metrics(args.metrics_path, metrics)

    model.train()
    data_iter = iter(loader)
    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids, attention_mask, labels = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, args.amp):
            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        add_checkpoint_stats(checkpoint_cumulative, ZActivationCheckpoint.get_stats())
        ZActivationCheckpoint.reset()

        loss_value = float(loss.detach().cpu())
        train_losses.append(loss_value)

        if not torch.isfinite(loss.detach()) or loss_value > args.abort_loss:
            metrics["status"] = "aborted"
            metrics["abort_reason"] = f"loss={loss_value} exceeded abort threshold"
            write_metrics(args.metrics_path, metrics)
            raise RuntimeError(metrics["abort_reason"])

        step_index = step + 1
        event = {"step": step_index, "train_loss": loss_value}
        should_eval = (
            args.eval_every > 0
            and args.eval_batches > 0
            and step_index % args.eval_every == 0
        )
        if should_eval:
            event["eval_loss"] = evaluate_loss(
                model, loader, device, args.amp, args.eval_batches
            )
        history.append(event)

        if args.log_every > 0 and (
            step_index % args.log_every == 0 or should_eval or step_index == args.max_steps
        ):
            print(json.dumps(event, sort_keys=True), flush=True)

        if should_eval:
            metrics["gpu_memory"] = gpu_memory(device)
            write_metrics(args.metrics_path, metrics)

    final_loss = evaluate_loss(model, loader, device, args.amp, args.eval_batches)
    elapsed = time.perf_counter() - started

    optimizer_stats = optimizer.get_memory_stats()
    metrics.update({
        "loss": {
            "initial_eval": initial_loss,
            "train_steps": train_losses,
            "final_eval": final_loss,
            "history": history,
        },
        "optimizer_stats": optimizer_stats,
        "activation_checkpoint": {
            "mneme_available": HAS_MNEME_CHECKPOINT,
            **checkpoint_cumulative,
        },
        "gpu_memory": gpu_memory(device),
        "elapsed_seconds": elapsed,
        "status": "completed",
    })

    metrics["mneme_storage"] = save_metrics_with_mneme(metrics, args.mneme_storage_dir)

    write_metrics(args.metrics_path, metrics)

    if not args.skip_save:
        save_checkpoint(
            args.save_path,
            model,
            config,
            metrics,
            optimizer=optimizer if args.save_optimizer else None,
        )
        metrics["checkpoint_path"] = str(Path(args.save_path).resolve())
        write_metrics(args.metrics_path, metrics)

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
