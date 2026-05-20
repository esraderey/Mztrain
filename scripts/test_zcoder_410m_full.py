"""Smoke-test "TODO PRENDIDO" sobre un ZCodeBERT ~410M params (densos equiv).

El modelo nunca se materializa denso: las proyecciones lineales del
transformer son factores low-rank MZTrain (init kaiming directo). Se reporta
el conteo DENSO equivalente (~410M) mientras se entrenan solo los factores.

"Todo prendido" = todos los subsistemas opt-in activos a la vez:
  - ElasticRank v2 (espectral+actividad + redundancia funcional) + loss-guard
  - VRAM Governor (gating de crecimiento bajo presion + OOM-retry)
  - Crecimiento de rango (RankSchedule.COSINE, growth_interval bajo)
  - Refactorizacion periodica (SVD fresca) + reset_after_refactorize
  - Compresion de estados del optimizer (ZCompressedAdam INT8)
  - Compresion de gradientes (TOP_K)
  - Activation checkpointing comprimido
  - AMP (activo en CUDA)

Es un SMOKE TEST: dataset sintetico minusculo, pocos steps. El objetivo no
es entrenar bien, sino demostrar que el pipeline completo arranca, corre y
reporta metricas a escala 410M con todo encendido simultaneamente.

Uso: .venv\\Scripts\\python.exe scripts\\test_zcoder_410m_full.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

for _name in ("mneme", "mneme.mneme_core", "mztrain"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mztrain import ZTrainConfig, RankSchedule, ZTrainEngine  # noqa: E402
from mztrain.config import GradientCompression  # noqa: E402
from mztrain.layers import ZFactorizedLinear  # noqa: E402
from mztrain.data import (  # noqa: E402
    CausalCodeDataset, CodeDataset, CodeTokenizer, SyntheticCodeGenerator,
)
from mztrain.models import ZCodeBERTConfig, ZCodeBERTForCausalLM  # noqa: E402

SEED = 1337
VOCAB = 50_000
SEQLEN = 32
N_SNIPPETS = 48
BATCH = 2
EPOCHS = 3


def build_410m_config() -> ZCodeBERTConfig:
    """~410M densos equivalentes. init kaiming => no materializa pesos densos.

    12*h^2 por bloque (4h^2 atencion + 8h^2 MLP) * L  +  embeddings.
    h=1024, L=28, vocab=50k  ->  ~406M equivalentes.
    """
    return ZCodeBERTConfig(
        vocab_size=VOCAB,
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=max(64, SEQLEN),
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        rank=48,
        factor_init_method="kaiming",   # CRITICO: evita SVD densa al arrancar
        use_activation_checkpointing=True,
    )


def build_config_todo_prendido(steps_per_epoch: int) -> ZTrainConfig:
    return ZTrainConfig(
        # --- rango factorizado + crecimiento ---
        initial_rank=48,
        max_rank=96,
        rank_schedule=RankSchedule.COSINE,
        rank_growth_interval=1,           # que el crecimiento dispare ya
        rank_growth_factor=2.0,
        rank_growth_warmup_steps=5,
        # mantener mlm_decoder denso y atado a embeddings (diseno del modelo):
        # su weight tiene vocab*h = 51.2M params; con el umbral por encima el
        # engine NO lo factoriza y se preserva el weight-tying.
        min_params_to_factorize=60_000_000,
        refactorize_interval=max(8, steps_per_epoch),  # dispara ~1/epoch
        # --- optimizer comprimido ---
        optimizer_type="compressed_adam",
        compress_optimizer_states=True,
        # --- compresion de gradientes ---
        gradient_compression=GradientCompression.TOP_K,
        gradient_top_k_ratio=0.1,
        # --- activation checkpointing comprimido ---
        checkpoint_activations=True,
        activation_compression=True,
        checkpoint_every_n_layers=2,
        # --- AMP (activo solo en CUDA) ---
        use_amp=True,
        precision_level="standard",
        # --- training ---
        learning_rate=3e-4,
        weight_decay=1e-4,
        warmup_steps=5,
        log_interval=1,
        # --- ElasticRank: v2 (redundancia) + loss-guard, TODO ON ---
        use_elastic_rank=True,
        elastic_rank_check_interval=2,
        elastic_rank_ema_beta=0.5,
        elastic_rank_min_per_layer=8,
        elastic_rank_sleep_patience_checks=2,
        elastic_rank_min_age_checks=1,
        elastic_rank_post_growth_grace_checks=0,
        elastic_rank_post_refactor_grace_checks=0,
        elastic_rank_compact_min_dirs=2,
        elastic_rank_prune_after_epochs=2,
        elastic_rank_use_redundancy_signal=True,      # v2
        elastic_rank_redundancy_threshold=0.9,
        elastic_rank_redundancy_patience_checks=2,
        elastic_rank_loss_guard_enabled=True,         # loss-guard
        elastic_rank_loss_guard_threshold=1e-3,
        # --- VRAM Governor: TODO ON ---
        use_vram_governor=True,
        vram_governor_interval=2,
        vram_governor_ema_beta=0.5,
    )


def make_loader() -> DataLoader:
    gen = SyntheticCodeGenerator(seed=SEED)
    snippets = gen.generate(N_SNIPPETS)
    tok = CodeTokenizer(vocab_size=VOCAB, max_length=SEQLEN)
    base = CodeDataset(snippets, tok, max_length=SEQLEN)
    causal = CausalCodeDataset(base)
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(causal, batch_size=BATCH, shuffle=True, generator=g)


def loss_fn(model, batch):
    input_ids, attn_mask, labels = batch
    loss, _ = model(input_ids, attention_mask=attn_mask, labels=labels)
    return loss


def trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def equivalent_params(model) -> int:
    eq = sum(
        m.full_parameters
        for m in model.modules() if isinstance(m, ZFactorizedLinear)
    )
    # embeddings (no factorizados) — sumar igual que get_model_stats
    bert = getattr(model, "bert", model)
    eq += sum(p.numel() for p in bert.embeddings.parameters())
    return eq


def main() -> None:
    # mneme reconfigura su logger al usarse; re-silenciar tras imports.
    for _name in ("mneme", "mneme.mneme_core"):
        logging.getLogger(_name).setLevel(logging.ERROR)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader()
    steps_per_epoch = max(1, len(loader))

    print(f"[410m-full] device={device}  seed={SEED}  "
          f"epochs={EPOCHS}  steps/epoch={steps_per_epoch}")

    model = ZCodeBERTForCausalLM(build_410m_config())
    eq = equivalent_params(model)
    real = trainable_params(model)
    print(f"[410m-full] modelo: ~{eq/1e6:.1f}M params densos equiv  |  "
          f"{real/1e6:.1f}M reales entrenables  |  "
          f"compresion {real/max(eq,1)*100:.1f}%")

    cfg = build_config_todo_prendido(steps_per_epoch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.time()
    engine = ZTrainEngine(model, cfg, device=device)
    init_real = trainable_params(engine.model)

    summary = engine.train(
        loader, loader, loss_fn,
        epochs=EPOCHS, early_stopping_patience=10_000,
    )
    wall = time.time() - t0
    final_real = trainable_params(engine.model)

    es = summary.get("elastic_rank_stats") or {}
    gov = summary.get("governor_stats") or {}
    opt = summary.get("optimizer_stats") or {}
    grad = summary.get("gradient_stats") or {}
    act = summary.get("activation_stats") or {}
    prec = summary.get("precision_stats") or {}

    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda" else None
    )

    print("\n" + "=" * 70)
    print("  RESULTADO  —  ZCodeBERT ~%.0fM equiv, TODO PRENDIDO" % (eq / 1e6))
    print("=" * 70)
    print(f"  wall time .................. {wall:.1f}s")
    print(f"  train_losses ............... {summary['train_losses']}")
    print(f"  val_losses ................. {summary['val_losses']}")
    print(f"  ranks (por epoch) .......... {summary['ranks']}")
    print(f"  params reales {init_real:,} -> {final_real:,}")
    if peak_mb is not None:
        print(f"  CUDA peak allocated ........ {peak_mb:,.1f} MB")
    print("-" * 70)
    print(f"  ElasticRank: slept={es.get('total_slept')} "
          f"pruned={es.get('total_pruned')} "
          f"revived={es.get('total_revived')} "
          f"rollbacks={es.get('total_rollbacks')} "
          f"max_coh={es.get('max_coherence_ema')}")
    print(f"  VRAM Governor: {gov}")
    print(f"  Optimizer (comprimido): "
          f"state_mb={opt.get('state_memory_mb')} "
          f"ratio={opt.get('compression_ratio')}")
    print(f"  Gradientes (TOP_K): {grad}")
    print(f"  Activation ckpt: {act}")
    print(f"  Precision: {prec}")
    print("=" * 70)

    out = {
        "model": "ZCodeBERTForCausalLM",
        "equivalent_params": eq,
        "real_trainable_params_init": init_real,
        "real_trainable_params_final": final_real,
        "device": str(device),
        "torch": torch.__version__,
        "epochs": EPOCHS,
        "wall_time_s": round(wall, 2),
        "cuda_peak_mb": peak_mb,
        "train_losses": summary["train_losses"],
        "val_losses": summary["val_losses"],
        "ranks": summary["ranks"],
        "elastic_rank_stats": es,
        "governor_stats": gov,
        "optimizer_stats": opt,
        "gradient_stats": grad,
        "activation_stats": act,
        "precision_stats": prec,
        "config": summary.get("config"),
    }
    res_dir = ROOT / "bench" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    p = res_dir / "test_zcoder_410m_full.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[410m-full] JSON -> {p}")


if __name__ == "__main__":
    main()
