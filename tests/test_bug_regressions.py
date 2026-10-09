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

import pytest
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


_PROJECTED_OPTIMIZERS = ("adaptive", "galore")


def _projected_growth_engine(opt_type: str, model: nn.Module, **overrides) -> ZTrainEngine:
    config = dict(
        initial_rank=4,
        max_rank=8,
        rank_schedule=RankSchedule.EXPONENTIAL,
        rank_growth_interval=1,
        use_elastic_rank=False,
        use_amp=False,
        compress_optimizer_states=False,
        refactorize_interval=0,
        optimizer_type=opt_type,
        galore_rank=2,
        learning_rate=1e-3,
        log_interval=9999,
    )
    config.update(overrides)
    return ZTrainEngine(model, ZTrainConfig(**config), device=torch.device("cpu"))


def _two_factorized_layers() -> nn.Module:
    return nn.Sequential(
        ZFactorizedLinear(16, 32, rank=4, init_method="svd", bias=True),
        nn.ReLU(),
        ZFactorizedLinear(32, 8, rank=4, init_method="svd", bias=True),
    )


def _manual_steps(engine: ZTrainEngine, x: torch.Tensor, n: int) -> None:
    for _ in range(n):
        engine.optimizer.zero_grad()
        engine.model(x).pow(2).mean().backward()
        engine.optimizer.step()


def _classification_loss(model, batch):
    xb, yb = batch
    return F.cross_entropy(model(xb), yb)


def _classification_loader():
    x = torch.randn(64, 128)
    y = torch.randint(0, 10, (64,))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=16,
    )


def _factorized_param_count(model: nn.Module) -> int:
    return sum(
        m.U.numel() + m.S.numel() + m.V.numel()
        for m in model.modules() if isinstance(m, ZFactorizedLinear)
    )


class TestProjectedOptimizersSurviveOptimizerRebuild:
    """optimizer_type="adaptive"/"galore" must keep stepping after the engine rebuilds the optimizer.

    After a topology change (rank growth, ElasticRank surgery, loss-guard
    rollback) the engine recreates the optimizer and writes back plain Adam
    states (step / exp_avg / exp_avg_sq / compressed). ZAdaptiveOptimizer and
    ZGaLoreOptimizer only created 'is_projected' in their lazy init, which a
    non-empty state skips, so the next step raised KeyError: 'is_projected'.
    """

    @pytest.mark.parametrize("opt_type", _PROJECTED_OPTIMIZERS)
    def test_train_with_rank_growth(self, opt_type):
        torch.manual_seed(0)
        engine = _projected_growth_engine(opt_type, _two_factorized_layers())
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(16, 16)), batch_size=4,
        )

        summary = engine.train(
            loader,
            val_loader=None,
            loss_fn=lambda m, b: m(b[0]).pow(2).mean(),
            epochs=3,
            early_stopping_patience=9999,
        )

        ranks = [
            m.rank for m in engine.model.modules() if isinstance(m, ZFactorizedLinear)
        ]
        assert ranks == [8, 8], f"rank growth did not run (ranks={ranks})"
        assert len(summary["train_losses"]) == 3
        assert torch.isfinite(torch.tensor(summary["train_losses"])).all()

    @pytest.mark.parametrize("opt_type", _PROJECTED_OPTIMIZERS)
    def test_step_after_growth_continues_migrated_adam_state(self, opt_type):
        """The migrated state must be completed, not thrown away with its momentum."""
        torch.manual_seed(0)
        engine = _projected_growth_engine(opt_type, _two_factorized_layers())
        x = torch.randn(24, 16)
        _manual_steps(engine, x, 6)
        engine._grow_model_rank(8)

        bias = next(
            m for m in engine.model.modules() if isinstance(m, ZFactorizedLinear)
        ).bias
        m_before = engine.optimizer.state[bias]["exp_avg"].clone()
        assert m_before.abs().sum() > 0, "precondition: growth migrated the bias momentum"

        _manual_steps(engine, x, 1)

        state = engine.optimizer.state[bias]
        beta1 = engine.optimizer.param_groups[0]["betas"][0]
        assert state["is_projected"] is False
        assert state["step"] == 7, f"Adam step counter restarted: {state['step']}"
        assert torch.allclose(
            state["exp_avg"], beta1 * m_before + (1 - beta1) * bias.grad
        ), "the step after growth did not build on the migrated exp_avg"

    @pytest.mark.parametrize("opt_type", _PROJECTED_OPTIMIZERS)
    def test_projected_param_stays_projected_after_growth(self, opt_type):
        """A parameter on the low-rank path must not end up with full-size Adam states."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Embedding(200, 128),  # 2D, min dim >= 128, >= 4096 elements: projected
            ZFactorizedLinear(128, 8, rank=4, init_method="svd", bias=True),
        )
        engine = _projected_growth_engine(opt_type, model, galore_rank=8)
        idx = torch.randint(0, 200, (32,))
        _manual_steps(engine, idx, 6)
        embedding = engine.model[0].weight
        assert engine.optimizer.state[embedding]["is_projected"] is True, "precondition"

        engine._grow_model_rank(8)
        _manual_steps(engine, idx, 1)

        state = engine.optimizer.state[embedding]
        assert state["is_projected"] is True
        assert tuple(state["exp_avg"].shape) == (8, 128), (
            f"projected exp_avg became {tuple(state['exp_avg'].shape)} after growth"
        )

    def _elastic_engine(self, model, opt_type, **overrides):
        return _projected_growth_engine(
            opt_type,
            model,
            initial_rank=16,
            max_rank=32,
            use_elastic_rank=True,
            elastic_rank_check_interval=1,
            elastic_rank_ema_beta=0.0,
            elastic_rank_sleep_patience_checks=2,
            elastic_rank_min_age_checks=0,
            elastic_rank_post_growth_grace_checks=0,
            elastic_rank_post_refactor_grace_checks=0,
            elastic_rank_min_per_layer=4,
            elastic_rank_compact_min_dirs=2,
            elastic_rank_wake_warmup_steps=4,
            rank_growth_warmup_steps=2,
            galore_rank=4,
            **overrides,
        )

    @pytest.mark.parametrize("opt_type", _PROJECTED_OPTIMIZERS)
    def test_training_continues_after_elastic_compaction(self, simple_model, opt_type):
        torch.manual_seed(0)
        engine = self._elastic_engine(simple_model, opt_type)
        engine.train_epoch(_classification_loader(), _classification_loss, epoch=0)
        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        rank0 = module.rank
        engine.elastic_rank.observe(
            engine.model, engine.optimizer, in_warmup=False, epoch=0
        )
        for i in range(3):
            module.sleep_mask[i] = True
        plan = engine.elastic_rank.build_plan(
            engine.model, requested_rank=rank0, grow_requested=False, epoch=1,
        )
        engine._apply_elastic_topology(plan, epoch=1)
        assert dict(engine.model.named_modules())[name].rank < rank0, "precondition"

        loss = engine.train_epoch(_classification_loader(), _classification_loss, epoch=1)

        assert torch.isfinite(torch.tensor(loss))

    @pytest.mark.parametrize("opt_type", _PROJECTED_OPTIMIZERS)
    def test_training_continues_after_loss_guard_rollback(self, simple_model, opt_type):
        torch.manual_seed(0)
        engine = self._elastic_engine(
            simple_model,
            opt_type,
            elastic_rank_loss_guard_enabled=True,
            elastic_rank_loss_guard_threshold=0.0,
            elastic_rank_loss_guard_cooldown_epochs=2,
        )
        engine.train_epoch(_classification_loader(), _classification_loss, epoch=0)
        engine._probe_batch = torch.zeros(1)
        # Fewer active factor params -> higher probe loss: compaction always rolls back.
        engine._loss_fn = lambda model, _batch: 1.0e6 / max(_factorized_param_count(model), 1)
        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        rank0 = module.rank
        engine.elastic_rank.observe(
            engine.model, engine.optimizer, in_warmup=False, epoch=0
        )
        for i in range(8):
            module.sleep_mask[i] = True
        plan = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=1,
        )
        engine._apply_elastic_topology_guarded(plan, epoch=1)
        assert engine.elastic_rank._total_rollbacks == 1, "precondition"
        assert dict(engine.model.named_modules())[name].rank == rank0, "precondition"

        loss = engine.train_epoch(_classification_loader(), _classification_loss, epoch=1)

        assert torch.isfinite(torch.tensor(loss))


class TestProjectedOptimizersMigrateCompressedStates:
    """A compressed low-rank moment must not be migrated as a full-size Adam moment.

    With compress_optimizer_states=True (the default) ZAdaptiveOptimizer and
    ZGaLoreOptimizer hold INT8 tuples whenever their step counter is a multiple
    of compression_interval. For a projected parameter the tuple decompresses
    to the subspace shape ((8, 128) for a (256, 128) factor), and
    decompress_opt_state returned it without the shape check it applies to
    plain tensors, so the engine and ElasticRank indexed it as a moment of the
    parameter: RuntimeError on rank growth, on ElasticRank compaction (after
    storing subspace-sized slices in the sleep bank) and on observe().
    """

    _RANK = 128  # smallest factor rank that takes the low-rank path

    def _compressed_engine(self, opt_type, model, **overrides):
        return _projected_growth_engine(
            opt_type,
            model,
            initial_rank=self._RANK,
            max_rank=192,
            compress_optimizer_states=True,
            galore_rank=8,
            **overrides,
        )

    def _steps_until_compressed(self, engine, x, projected_params) -> int:
        n = engine.optimizer.compression_interval
        _manual_steps(engine, x, n)
        for p in projected_params:
            state = engine.optimizer.state[p]
            assert state["is_projected"] is True, "precondition"
            assert isinstance(state["exp_avg"], tuple), "precondition: compressed state"
        return n

    @pytest.mark.parametrize("opt_type", _PROJECTED_OPTIMIZERS)
    def test_rank_growth(self, opt_type):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Embedding(200, 256),
            ZFactorizedLinear(256, 256, rank=self._RANK, init_method="svd", bias=True),
        )
        engine = self._compressed_engine(opt_type, model)
        idx = torch.randint(0, 200, (32,))
        layer = engine.model[1]
        n = self._steps_until_compressed(engine, idx, (layer.U, layer.V))

        engine._grow_model_rank(160)

        assert layer.rank == 160
        bias_state = engine.optimizer.state[layer.bias]
        assert bias_state["step"] == n and bias_state["exp_avg"].abs().sum() > 0, (
            "a moment that does have the parameter's shape must still be migrated"
        )
        _manual_steps(engine, idx, 1)
        for p, low_rank_shape in ((layer.U, (8, 160)), (layer.V, (160, 8))):
            state = engine.optimizer.state[p]
            assert state["is_projected"] is True
            assert tuple(state["exp_avg"].shape) == low_rank_shape

    @pytest.mark.parametrize("opt_type", _PROJECTED_OPTIMIZERS)
    def test_elastic_compaction(self, opt_type):
        torch.manual_seed(0)
        model = nn.Sequential(
            ZFactorizedLinear(256, 256, rank=self._RANK, init_method="svd", bias=True),
        )
        engine = self._compressed_engine(opt_type, model, use_elastic_rank=True)
        x = torch.randn(32, 256)
        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        self._steps_until_compressed(engine, x, (module.U, module.V))
        n_sleep = engine.config.elastic_rank_compact_min_dirs
        module.sleep_mask[:n_sleep] = True
        plan = engine.elastic_rank.build_plan(
            engine.model, requested_rank=self._RANK, grow_requested=False, epoch=1,
        )

        engine._apply_elastic_topology(plan, epoch=1)

        assert module.rank == self._RANK - n_sleep
        sleepers = engine.elastic_rank.bank[name]
        assert len(sleepers) == n_sleep
        for sd in sleepers:
            assert sd.m_u.shape == sd.vsq_u.shape == sd.u.shape
            assert sd.m_v.shape == sd.vsq_v.shape == sd.v.shape
        _manual_steps(engine, x, 1)

    @pytest.mark.parametrize("opt_type", _PROJECTED_OPTIMIZERS)
    def test_elastic_observe_with_square_factor(self, opt_type):
        """rank == in_features: V's subspace moment is (8, 128), with no row per direction."""
        torch.manual_seed(0)
        model = nn.Sequential(
            ZFactorizedLinear(self._RANK, 256, rank=self._RANK, init_method="svd", bias=True),
        )
        engine = self._compressed_engine(opt_type, model, use_elastic_rank=True)
        x = torch.randn(32, self._RANK)
        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        self._steps_until_compressed(engine, x, (module.U, module.V))

        engine.elastic_rank.observe(engine.model, engine.optimizer, in_warmup=False, epoch=0)

        update_ema = engine.elastic_rank._states[name].update_ema
        assert update_ema.shape == (self._RANK,)
        assert torch.isfinite(update_ema).all()
