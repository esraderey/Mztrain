"""
Regression tests for the 4 critical bugs identified during audit.

Each test must FAIL before the fix and PASS after the fix.

  Bug #1: ZRankScheduler.get_rank() mutates current_rank before returning, so
          engine.train() sees new_rank == current_rank and never calls
          _grow_model_rank(). Layers remain at initial_rank for entire training.

  Bug #2: ZGaLoreOptimizer._compress_state omits the *127 scale factor, so INT8
          values collapse to {-1, 0, 1}. Adam state (exp_avg, exp_avg_sq) is
          destroyed on every compression cycle.

  Bug #3: Same bug as #2 in ZAdaptiveOptimizer._compress_state (APOLLO path).

  Bug #7: ZFactorizedLinear._init_from_weight uses plain truncated SVD without
          EPSI scaling. For Kaiming-init weights at low rank, it loses ~97% of
          Frobenius energy → activations collapse → training cannot converge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from mztrain import (
    ZTrainEngine,
    ZTrainConfig,
    RankSchedule,
    ZFactorizedLinear,
    factorize_existing_model,
)
from mztrain.checkpoint import ZActivationCheckpoint, z_checkpoint
from mztrain.models.zcodebert import ZCodeBERTConfig, ZCodeBERTEncoder
from mztrain.projector import ZGaLoreOptimizer
from mztrain.adaptive_optimizer import ZAdaptiveOptimizer
from mztrain.scheduler import ZSpectralRankScheduler


class TestBug1RankActuallyGrows:
    """get_rank() mutating current_rank prevents engine from ever growing the model."""

    def test_layer_rank_grows_with_exponential_schedule(self):
        torch.manual_seed(0)

        # Build model with pre-factorized layers using init_method='svd' (EPSI),
        # so the model is numerically sane while we focus on rank-growth.
        class FactoredMLP(nn.Module):
            def __init__(self, rank: int):
                super().__init__()
                self.lin1 = ZFactorizedLinear(64, 128, rank=rank, init_method="svd")
                self.lin2 = ZFactorizedLinear(128, 64, rank=rank, init_method="svd")
                self.lin3 = ZFactorizedLinear(64, 10, rank=rank, init_method="svd")

            def forward(self, x):
                x = F.relu(self.lin1(x))
                x = F.relu(self.lin2(x))
                return self.lin3(x)

        model = FactoredMLP(rank=4)
        config = ZTrainConfig(
            initial_rank=4,
            max_rank=32,
            rank_schedule=RankSchedule.EXPONENTIAL,
            rank_growth_interval=1,        # attempt growth every epoch
            rank_growth_factor=2.0,
            learning_rate=1e-4,
            use_amp=False,
            log_interval=9999,             # silence logs during test
        )
        engine = ZTrainEngine(model, config)

        x = torch.randn(64, 64)
        y = torch.randint(0, 10, (64,))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=16,
        )

        def loss_fn(m, batch):
            xb, yb = batch
            return F.cross_entropy(m(xb), yb)

        initial_layer = next(
            m for m in engine.model.modules() if isinstance(m, ZFactorizedLinear)
        )
        assert initial_layer.rank == 4

        # Train 3 epochs. With growth_interval=1, growth should fire at epochs 1, 2.
        engine.train(
            loader,
            val_loader=None,
            loss_fn=loss_fn,
            epochs=3,
            early_stopping_patience=9999,
        )

        final_layer = next(
            m for m in engine.model.modules() if isinstance(m, ZFactorizedLinear)
        )
        assert final_layer.rank > 4, (
            f"Bug #1: layer rank stayed at {final_layer.rank}, expected > 4. "
            f"Scheduler reports current_rank={engine.rank_scheduler.current_rank}, "
            f"but actual layer was never grown by _grow_model_rank()."
        )


class TestSpectralSchedulerRankGrowth:
    """SPECTRAL schedule must let the engine apply rank growth."""

    def test_get_rank_does_not_mutate_current_rank_before_engine_growth(self):
        model = nn.Sequential(ZFactorizedLinear(32, 32, rank=4, init_method="svd"))
        scheduler = ZSpectralRankScheduler(
            model=model,
            initial_rank=4,
            max_rank=16,
            total_epochs=4,
            energy_threshold=0.90,
            growth_interval=1,
            growth_factor=2.0,
            factorized_cls=ZFactorizedLinear,
        )
        scheduler._compute_spectral_energy = lambda: 0.10

        new_rank = scheduler.get_rank(epoch=1, current_loss=1.0)

        assert new_rank == 8
        assert scheduler.current_rank == 4, (
            "SPECTRAL get_rank must be read-only; ZTrainEngine updates "
            "current_rank only after _grow_model_rank() succeeds."
        )


class TestCheckpointArgumentPreservation:
    """Compressed checkpointing must recompute with the original arg list."""

    def test_backward_preserves_tensor_args_without_grad(self):
        ZActivationCheckpoint.reset()
        x = torch.randn(8, requires_grad=True)
        mask = torch.linspace(0.1, 0.8, steps=8)

        out = z_checkpoint(lambda values, scale: values * scale, x, mask)
        out.sum().backward()

        assert torch.allclose(x.grad, mask, atol=1e-5)


class TestRootLinearFactorization:
    """A bare nn.Linear root model is a valid model and must be factorized."""

    def test_factorize_existing_model_handles_root_linear(self):
        model = nn.Linear(16, 8)

        z_model, stats = factorize_existing_model(model, rank=4, min_params=1)

        assert isinstance(z_model, ZFactorizedLinear)
        assert stats["replaced_layers"][0]["name"] == "<root>"

    def test_engine_handles_root_linear(self):
        model = nn.Linear(16, 8)
        config = ZTrainConfig(
            initial_rank=4,
            max_rank=4,
            min_params_to_factorize=1,
            use_amp=False,
            checkpoint_activations=False,
        )

        engine = ZTrainEngine(model, config)

        assert isinstance(engine.model, ZFactorizedLinear)
        assert isinstance(engine.export_full_model(), nn.Linear)


class TestZCodeBERTCompressedCheckpoint:
    """ZCodeBERT's checkpoint path should use MZTrain compressed checkpoints."""

    def test_encoder_checkpoint_updates_compressed_activation_stats(self):
        config = ZCodeBERTConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=16,
            rank=4,
            use_activation_checkpointing=True,
        )
        encoder = ZCodeBERTEncoder(config)
        encoder.train()
        ZActivationCheckpoint.reset()

        hidden = torch.randn(2, 8, config.hidden_size, requires_grad=True)
        out = encoder(hidden)
        out.sum().backward()

        stats = ZActivationCheckpoint.get_stats()
        assert stats["total_compressed"] > 0


class TestBug2GaLoreInt8:
    """ZGaLoreOptimizer._compress_state must round-trip with low error."""

    def test_galore_compress_decompress_low_error(self):
        torch.manual_seed(0)
        tensor = torch.randn(4096) * 0.01  # realistic Adam-state magnitude

        dummy = nn.Parameter(torch.zeros(1))
        opt = ZGaLoreOptimizer([dummy], lr=1e-3)

        compressed = opt._compress_state(tensor)
        recovered = opt._decompress_state(compressed, tensor.dtype)

        rel_err = ((tensor - recovered).norm() / tensor.norm().clamp(min=1e-12)).item()
        n_unique = recovered.unique().numel()

        assert rel_err < 0.05, (
            f"Bug #2: GaLore INT8 rel_err={rel_err:.4f} (expected <0.05). "
            f"~0.85 indicates ternary collapse from missing *127 scale."
        )
        assert n_unique > 50, (
            f"Bug #2: only {n_unique} unique values after round-trip "
            f"(ternary collapse produces ~3 per block)."
        )


class TestBug3AdaptiveInt8:
    """ZAdaptiveOptimizer._compress_state — same bug as #2."""

    def test_adaptive_compress_decompress_low_error(self):
        torch.manual_seed(0)
        tensor = torch.randn(4096) * 0.01

        dummy = nn.Parameter(torch.zeros(1))
        opt = ZAdaptiveOptimizer([dummy], lr=1e-3)

        compressed = opt._compress_state(tensor)
        recovered = opt._decompress_state(compressed, tensor.dtype)

        rel_err = ((tensor - recovered).norm() / tensor.norm().clamp(min=1e-12)).item()
        n_unique = recovered.unique().numel()

        assert rel_err < 0.05, (
            f"Bug #3: Adaptive INT8 rel_err={rel_err:.4f} (expected <0.05)."
        )
        assert n_unique > 50, (
            f"Bug #3: only {n_unique} unique values after round-trip."
        )


class TestFP8Path:
    """Verifica el path FP8 opt-in (mejora SOTA #3)."""

    def test_fp8_linear_round_trip_close_to_bf16(self):
        """fp8_linear debe dar resultados cerca de F.linear (BF16) en hardware FP8."""
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("FP8 path requires CUDA")
        from mztrain.precision import fp8_supported, fp8_linear
        if not fp8_supported():
            import pytest
            pytest.skip("FP8 requires SM89+")

        torch.manual_seed(0)
        device = torch.device("cuda")
        x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 128, device=device, dtype=torch.bfloat16)

        ref = F.linear(x, w)  # BF16 reference
        out = fp8_linear(x, w, out_dtype=torch.bfloat16)

        rel_err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        assert rel_err < 0.10, f"FP8 vs BF16 rel_err={rel_err:.4f} (expected <0.10)"

    def test_fp8_linear_supports_backward(self):
        """fp8_linear debe poder hacer backward (PyTorch nativo no, nuestro autograd si)."""
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("FP8 path requires CUDA")
        from mztrain.precision import fp8_supported, fp8_linear
        if not fp8_supported():
            import pytest
            pytest.skip("FP8 requires SM89+")

        device = torch.device("cuda")
        x = torch.randn(8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)

        out = fp8_linear(x, w)
        out.sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()
        assert w.grad is not None and w.grad.isfinite().all()


class TestBug7EpsiFromWeight:
    """_init_from_weight must preserve enough energy for activations to survive."""

    def test_low_rank_factorization_preserves_output_variance(self):
        torch.manual_seed(0)
        in_features, out_features, rank = 512, 256, 16

        lin = nn.Linear(in_features, out_features, bias=False)
        z = ZFactorizedLinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            existing_weight=lin.weight.data,
            bias=False,
        )

        x = torch.randn(64, in_features)
        with torch.no_grad():
            y_dense_var = lin(x).var().item()
            y_factored_var = z(x).var().item()

        ratio = y_factored_var / max(y_dense_var, 1e-12)
        assert 0.5 < ratio < 2.0, (
            f"Bug #7: factored/dense output variance ratio {ratio:.3f} "
            f"(expected ~1, allowed 0.5..2.0). Without EPSI in _init_from_weight, "
            f"low-rank truncation of Kaiming weights causes activation collapse."
        )
