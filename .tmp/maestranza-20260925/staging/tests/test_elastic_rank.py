"""
Tests para mztrain.elastic_rank - ElasticRank (rango bidireccional).

Cubre:
- buffers/forward gated en ZFactorizedLinear (inertes con gate=1),
- scoring hibrido (espectral AND actividad de Adam),
- observe -> soft sleep con sus protecciones (min_per_layer, gracia),
- mask_gradients / tick_wake,
- build_plan: compactar, podar, revivir-al-crecer,
- round trip sleep -> bank -> revive a traves del engine,
- que el camino legacy queda intacto cuando use_elastic_rank=False.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mztrain.config import ZTrainConfig
from mztrain.engine import ZTrainEngine
from mztrain.layers import ZFactorizedLinear
from mztrain.optimizer import ZCompressedAdam
from mztrain.elastic_rank import (
    ElasticRankController,
    SleepingDirection,
)


def _loader(n=64, d=128, k=10, bs=16):
    x = torch.randn(n, d)
    y = torch.randint(0, k, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True)


def loss_fn(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)


def _elastic_cfg(**kw):
    base = dict(
        initial_rank=16, max_rank=32, use_amp=False,
        use_elastic_rank=True,
        elastic_rank_check_interval=1,
        elastic_rank_ema_beta=0.0,            # ema = score (determinista)
        elastic_rank_sleep_patience_checks=2,
        elastic_rank_min_age_checks=0,
        elastic_rank_post_growth_grace_checks=0,
        elastic_rank_post_refactor_grace_checks=0,
        elastic_rank_min_per_layer=4,
        elastic_rank_compact_min_dirs=2,
        elastic_rank_wake_warmup_steps=4,
        rank_growth_warmup_steps=2,
    )
    base.update(kw)
    return ZTrainConfig(**base)


# ---------------------------------------------------------------------------
# Capa: buffers + forward gated
# ---------------------------------------------------------------------------

class TestLayerElastic:
    def test_buffers_default_inert(self):
        layer = ZFactorizedLinear(64, 48, rank=8)
        assert layer.wake_gate.shape == (8,)
        assert torch.all(layer.wake_gate == 1.0)
        assert layer.sleep_mask.shape == (8,)
        assert not layer.sleep_mask.any()

        x = torch.randn(4, 64)
        ref = (F.linear(x, layer.V) * layer.S.unsqueeze(0)) @ layer.U.t()
        ref = ref + layer.bias
        out = layer(x)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_gate_scales_contribution(self):
        layer = ZFactorizedLinear(32, 32, rank=6, bias=False)
        x = torch.randn(2, 32)
        full = layer(x)
        layer.wake_gate[0] = 0.0           # apagar direccion 0
        gated = layer(x)
        assert not torch.allclose(full, gated)
        # gate=0 en todas -> salida nula
        layer.wake_gate.zero_()
        assert torch.allclose(layer(x), torch.zeros_like(full), atol=1e-6)

    def test_gate_size_mismatch_guard(self):
        """Un mutador externo (p. ej. refactorize) puede reemplazar S sin
        tocar el buffer; el guard debe ignorar el gate, no romper."""
        layer = ZFactorizedLinear(32, 32, rank=6)
        layer.wake_gate = torch.ones(99)   # desincronizado a proposito
        x = torch.randn(2, 32)
        out = layer(x)                     # no debe lanzar
        assert out.shape == (2, 32)

    def test_grow_rank_resizes_buffers(self):
        layer = ZFactorizedLinear(64, 64, rank=4)
        layer.sleep_mask[1] = True
        layer.grow_rank(10)
        assert layer.wake_gate.shape == (10,)
        assert layer.sleep_mask.shape == (10,)
        assert torch.all(layer.wake_gate == 1.0)
        # prefijo conservado
        assert bool(layer.sleep_mask[1])
        assert not bool(layer.sleep_mask[5])

    def test_elastic_replace_factors(self):
        layer = ZFactorizedLinear(40, 24, rank=8)
        new_r = 5
        U = torch.randn(24, new_r)
        S = torch.randn(new_r)
        V = torch.randn(new_r, 40)
        wg = torch.zeros(new_r)
        sm = torch.zeros(new_r, dtype=torch.bool)
        layer.elastic_replace_factors(U, S, V, wg, sm)
        assert layer.rank == new_r
        assert layer.U.shape == (24, new_r)
        assert layer.V.shape == (new_r, 40)
        assert layer.wake_gate.shape == (new_r,)
        out = layer(torch.randn(3, 40))
        assert out.shape == (3, 24)


# ---------------------------------------------------------------------------
# Controller: scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_scores_shapes_and_range(self):
        cfg = _elastic_cfg()
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(64, 64, rank=10)
        opt = ZCompressedAdam(layer.parameters(), lr=1e-3)
        spectral, update = ec._scores(layer, opt)
        assert spectral.shape == (10,)
        assert update.shape == (10,)
        assert float(spectral.max()) <= 1.0 + 1e-6
        assert float(spectral.min()) >= 0.0
        assert torch.isfinite(spectral).all()
        assert torch.isfinite(update).all()

    def test_adam_step_zero_without_state(self):
        cfg = _elastic_cfg()
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(32, 32, rank=4)
        opt = ZCompressedAdam(layer.parameters(), lr=1e-3)
        step = ec._adam_step(opt, layer.U)
        assert step.shape == layer.U.shape
        assert torch.all(step == 0)


# ---------------------------------------------------------------------------
# Controller: observe / soft sleep
# ---------------------------------------------------------------------------

class TestObserve:
    def _layer_with_dead_dir(self):
        # rank 8: direccion 0 minuscula, resto fuerte.
        layer = ZFactorizedLinear(64, 64, rank=8)
        with torch.no_grad():
            layer.U[:, 0] *= 1e-6
            layer.V[0, :] *= 1e-6
            layer.S[0] = 1e-6
            layer.S[1:] = 5.0
        return layer

    def test_marks_dead_direction(self):
        cfg = _elastic_cfg(elastic_rank_min_per_layer=2)
        ec = ElasticRankController(cfg)
        layer = self._layer_with_dead_dir()
        opt = ZCompressedAdam(layer.parameters(), lr=1e-3)
        model = nn.Sequential(layer)
        for _ in range(3):  # patience=2 -> tras 2 chequeos duerme
            ec.observe(model, opt, in_warmup=False, epoch=0)
        assert bool(layer.sleep_mask[0])           # debil -> dormida
        assert not bool(layer.sleep_mask[1])       # fuerte -> despierta

    def test_respects_min_per_layer(self):
        # Todas las direcciones debiles, pero no debe dormir por debajo del min.
        cfg = _elastic_cfg(elastic_rank_min_per_layer=3)
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(32, 32, rank=6)
        with torch.no_grad():
            layer.U *= 1e-6
            layer.V *= 1e-6
            layer.S.fill_(1e-6)
        opt = ZCompressedAdam(layer.parameters(), lr=1e-3)
        model = nn.Sequential(layer)
        for _ in range(6):
            ec.observe(model, opt, in_warmup=False, epoch=0)
        awake = int((~layer.sleep_mask).sum())
        assert awake >= 3

    def test_warmup_suppresses_sleep(self):
        cfg = _elastic_cfg(elastic_rank_min_per_layer=2)
        ec = ElasticRankController(cfg)
        layer = self._layer_with_dead_dir()
        opt = ZCompressedAdam(layer.parameters(), lr=1e-3)
        model = nn.Sequential(layer)
        for _ in range(5):
            ec.observe(model, opt, in_warmup=True, epoch=0)
        assert not layer.sleep_mask.any()          # en warmup no se duerme

    def test_grace_suppresses_sleep(self):
        cfg = _elastic_cfg(
            elastic_rank_min_per_layer=2,
            elastic_rank_post_growth_grace_checks=10,
        )
        ec = ElasticRankController(cfg)
        ec.note_growth()
        layer = self._layer_with_dead_dir()
        opt = ZCompressedAdam(layer.parameters(), lr=1e-3)
        model = nn.Sequential(layer)
        for _ in range(5):
            ec.observe(model, opt, in_warmup=False, epoch=0)
        assert not layer.sleep_mask.any()


# ---------------------------------------------------------------------------
# Controller: mask_gradients / tick_wake
# ---------------------------------------------------------------------------

class TestMaskAndWake:
    def test_mask_gradients_freezes_only_slept(self):
        cfg = _elastic_cfg()
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(16, 16, rank=5)
        model = nn.Sequential(layer)
        layer.sleep_mask[2] = True
        layer.U.grad = torch.ones_like(layer.U)
        layer.S.grad = torch.ones_like(layer.S)
        layer.V.grad = torch.ones_like(layer.V)
        ec.mask_gradients(model)
        assert torch.all(layer.U.grad[:, 2] == 0)
        assert float(layer.S.grad[2]) == 0.0
        assert torch.all(layer.V.grad[2, :] == 0)
        # otras direcciones intactas
        assert torch.all(layer.U.grad[:, 0] == 1)

    def test_tick_wake_ramps_gate(self):
        cfg = _elastic_cfg(elastic_rank_wake_warmup_steps=4)
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(16, 16, rank=4)
        model = nn.Sequential(layer)
        # Estado de scoring con una direccion en wake warmup.
        st = ec._state_for("0", 4)
        st.wake_left[1] = 4
        layer.wake_gate[1] = 0.0
        gates = []
        for _ in range(5):
            ec.tick_wake(model)
            gates.append(float(layer.wake_gate[1]))
        assert gates[0] < gates[-1]
        assert gates[-1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Controller: build_plan
# ---------------------------------------------------------------------------

class TestBuildPlan:
    def test_compaction_respects_min_and_threshold(self):
        cfg = _elastic_cfg(
            elastic_rank_min_per_layer=4,
            elastic_rank_compact_min_dirs=2,
        )
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(64, 64, rank=10)
        model = nn.Sequential(layer)
        ec._state_for("0", 10)
        for i in range(7):                 # 7 marcadas, min=4 -> compacta 6
            layer.sleep_mask[i] = True
        plan = ec.build_plan(model, requested_rank=10,
                              grow_requested=False, epoch=1)
        lp = plan.layers["0"]
        assert lp.changed
        assert len(lp.sleep) == 6          # rank(10) - min(4)
        assert lp.new_rank == 4

    def test_no_compaction_below_compact_min(self):
        cfg = _elastic_cfg(elastic_rank_compact_min_dirs=5)
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(64, 64, rank=10)
        model = nn.Sequential(layer)
        ec._state_for("0", 10)
        layer.sleep_mask[0] = True
        layer.sleep_mask[1] = True         # solo 2 < compact_min(5)
        plan = ec.build_plan(model, requested_rank=10,
                              grow_requested=False, epoch=1)
        assert not plan.layers["0"].changed

    def test_prune_old_sleepers(self):
        cfg = _elastic_cfg(elastic_rank_prune_after_epochs=3)
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(32, 32, rank=8)
        model = nn.Sequential(layer)
        ec._state_for("0", 8)
        sd = SleepingDirection(
            layer_name="0",
            u=torch.zeros(32), s=torch.zeros(()), v=torch.zeros(32),
            m_u=torch.zeros(32), m_s=torch.zeros(()), m_v=torch.zeros(32),
            vsq_u=torch.zeros(32), vsq_s=torch.zeros(()), vsq_v=torch.zeros(32),
            step=0, spectral_at_sleep=0.0, update_at_sleep=0.0,
            epoch_slept=0,
        )
        ec.bank["0"] = [sd]
        plan = ec.build_plan(model, requested_rank=8,
                             grow_requested=False, epoch=10)
        assert plan.pruned == 1
        assert ec.bank["0"] == []

    def test_revive_on_grow_prefers_bank(self):
        cfg = _elastic_cfg()
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(64, 64, rank=6)
        model = nn.Sequential(layer)
        ec._state_for("0", 6)
        sd = SleepingDirection(
            layer_name="0",
            u=torch.ones(64), s=torch.ones(()), v=torch.ones(64),
            m_u=torch.zeros(64), m_s=torch.zeros(()), m_v=torch.zeros(64),
            vsq_u=torch.zeros(64), vsq_s=torch.zeros(()), vsq_v=torch.zeros(64),
            step=3, spectral_at_sleep=0.5, update_at_sleep=0.5,
            epoch_slept=2,
        )
        ec.bank["0"] = [sd]
        plan = ec.build_plan(model, requested_rank=8,
                             grow_requested=True, epoch=3)
        lp = plan.layers["0"]
        assert len(lp.revive) == 1         # revive antes que ruido
        assert lp.grow == 1                # 6 -> 8: 1 revive + 1 ruido
        assert lp.new_rank == 8


# ---------------------------------------------------------------------------
# Engine: integracion end-to-end + round trip sleep/revive
# ---------------------------------------------------------------------------

class TestV2Redundancy:
    """v2: señal de redundancia funcional (leverage del Gram rango-1)."""

    def test_metric_flags_duplicate_low_for_independent(self):
        cfg = _elastic_cfg()
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(24, 24, rank=6, bias=False)
        with torch.no_grad():
            # Direccion 1 = copia exacta de la 0 (redundante);
            # 2..5 aleatorias (genericamente casi ortogonales en 24x24).
            layer.U[:, 1] = layer.U[:, 0]
            layer.V[1, :] = layer.V[0, :]
            layer.S[1] = layer.S[0]
        coh = ec._redundancy(layer)
        assert coh.shape == (6,)
        assert torch.isfinite(coh).all()
        assert float(coh[0]) > 0.8 and float(coh[1]) > 0.8
        assert float(coh[2:].median()) < float(coh[0])

    def test_metric_low_rank_layer_mostly_redundant(self):
        cfg = _elastic_cfg()
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(16, 16, rank=8, bias=False)
        with torch.no_grad():
            bu = torch.randn(16)
            bv = torch.randn(16)
            for i in range(8):                 # todas colineales (W rango 1)
                layer.U[:, i] = bu * (0.5 + i)
                layer.V[i, :] = bv * (1.0 + 0.3 * i)
                layer.S[i] = 1.0 + i
        coh = ec._redundancy(layer)
        # >=7 de 8 casi enteramente en el span de las demas.
        assert int((coh > 0.9).sum()) >= 7

    def _collinear_layer(self):
        layer = ZFactorizedLinear(12, 12, rank=8, bias=False)
        with torch.no_grad():
            bu, bv = torch.randn(12), torch.randn(12)
            for i in range(8):
                layer.U[:, i] = bu * (0.7 + 0.1 * i)
                layer.V[i, :] = bv * (0.9 + 0.05 * i)
                layer.S[i] = 2.0 + 0.5 * i      # magnitud espectral SANA
        return layer

    def test_v2_sleeps_redundant_where_v1_cannot(self):
        # Capa colineal: contribucion espectral alta (v1 NO duerme) pero
        # funcionalmente redundante (v2 SI).
        cfg = _elastic_cfg(
            elastic_rank_ema_beta=0.0,
            elastic_rank_sleep_spectral_threshold=1e-4,
            elastic_rank_min_per_layer=2,
            elastic_rank_use_redundancy_signal=True,
            elastic_rank_redundancy_threshold=0.8,
            elastic_rank_redundancy_patience_checks=2,
        )
        ec = ElasticRankController(cfg)
        layer = self._collinear_layer()
        model = nn.Sequential(layer)
        opt = ZCompressedAdam(layer.parameters(), lr=1e-3)
        for _ in range(4):
            ec.observe(model, opt, in_warmup=False, epoch=0)
        assert bool(layer.sleep_mask.any())          # v2 durmio redundantes
        assert int((~layer.sleep_mask).sum()) >= 2   # respeta min_per_layer

    def test_v1_only_does_not_sleep_collinear(self):
        cfg = _elastic_cfg(
            elastic_rank_ema_beta=0.0,
            elastic_rank_sleep_spectral_threshold=1e-4,
            elastic_rank_min_per_layer=2,
            elastic_rank_use_redundancy_signal=False,   # solo v1
        )
        ec = ElasticRankController(cfg)
        layer = self._collinear_layer()
        model = nn.Sequential(layer)
        opt = ZCompressedAdam(layer.parameters(), lr=1e-3)
        for _ in range(6):
            ec.observe(model, opt, in_warmup=False, epoch=0)
        assert not layer.sleep_mask.any()            # v1 ciego a esto

    def test_config_validation_v2(self):
        with pytest.raises(ValueError):
            ZTrainConfig(use_elastic_rank=True,
                         elastic_rank_redundancy_threshold=1.5).validate()
        with pytest.raises(ValueError):
            ZTrainConfig(use_elastic_rank=True,
                         elastic_rank_redundancy_patience_checks=0).validate()
        ZTrainConfig(use_elastic_rank=True).validate()  # defaults validos

    def test_remap_carries_redundancy_state(self):
        cfg = _elastic_cfg()
        ec = ElasticRankController(cfg)
        from mztrain.elastic_rank import LayerPlan
        st = ec._state_for("L", 5)
        st.redund_ema[:] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        st.redund_counter[:] = torch.tensor([0, 1, 2, 3, 4])
        # conservar indices 1 y 3 -> nuevo rank 2
        plan = LayerPlan(name="L", keep=[1, 3], sleep=[0, 2, 4],
                         revive=[], grow=0, new_rank=2)
        ec.remap_state("L", plan)
        new = ec._states["L"]
        assert new.redund_ema.numel() == 2
        assert float(new.redund_ema[0]) == pytest.approx(0.2)
        assert float(new.redund_ema[1]) == pytest.approx(0.4)
        assert int(new.redund_counter[0]) == 1
        assert int(new.redund_counter[1]) == 3


class TestEngineIntegration:
    def test_disabled_by_default(self, simple_model):
        engine = ZTrainEngine(simple_model, ZTrainConfig(initial_rank=8),
                              device=torch.device("cpu"))
        assert engine.elastic_rank is None

    def test_enabled_attaches_controller(self, simple_model):
        engine = ZTrainEngine(simple_model, _elastic_cfg(),
                              device=torch.device("cpu"))
        assert isinstance(engine.elastic_rank, ElasticRankController)

    def test_training_runs_and_reports_stats(self, simple_model):
        cfg = _elastic_cfg(log_interval=100)
        engine = ZTrainEngine(simple_model, cfg, device=torch.device("cpu"))
        summary = engine.train(_loader(), None, loss_fn, epochs=3)
        assert len(summary["train_losses"]) == 3
        assert summary["elastic_rank_stats"] is not None
        assert "sleeping_now" in summary["elastic_rank_stats"]

    def test_sleep_revive_round_trip(self, simple_model):
        cfg = _elastic_cfg()
        engine = ZTrainEngine(simple_model, cfg, device=torch.device("cpu"))
        # Poblar estado del optimizer.
        engine.train_epoch(_loader(), loss_fn, epoch=0)

        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        rank0 = module.rank
        engine.elastic_rank.observe(
            engine.model, engine.optimizer, in_warmup=False, epoch=0
        )
        # Forzar 3 direcciones a soft sleep.
        for i in range(3):
            module.sleep_mask[i] = True

        # --- Compactar (dormir de verdad, reduce rango) ---
        plan = engine.elastic_rank.build_plan(
            engine.model, requested_rank=rank0,
            grow_requested=False, epoch=1,
        )
        assert plan.has_changes()
        engine._apply_elastic_topology(plan, epoch=1)

        module = dict(engine.model.named_modules())[name]
        rank1 = module.rank
        assert rank1 < rank0
        assert len(engine.elastic_rank.bank.get(name, [])) >= 1
        out = engine.model(torch.randn(4, 128))
        assert torch.isfinite(out).all()
        # optimizer reconstruido y con estado para los nuevos factores
        assert module.U in engine.optimizer.state

        # --- Revivir al crecer ---
        plan2 = engine.elastic_rank.build_plan(
            engine.model, requested_rank=rank0,
            grow_requested=True, epoch=2,
        )
        assert plan2.has_changes()
        engine._apply_elastic_topology(plan2, epoch=2)

        module = dict(engine.model.named_modules())[name]
        assert module.rank > rank1
        # alguna direccion revivida arranca con gate 0 (mini-warmup)
        assert float(module.wake_gate.min()) == 0.0
        out = engine.model(torch.randn(4, 128))
        assert torch.isfinite(out).all()

    def test_sleeping_direction_value_round_trip(self):
        """Los factores dormidos se restauran (dentro de la tolerancia de
        la precision del sleep bank) al revivir."""
        cfg = _elastic_cfg(elastic_rank_sleep_store_dtype="float32")
        ec = ElasticRankController(cfg)
        u = torch.randn(32)
        sd = SleepingDirection(
            layer_name="L",
            u=u.clone(), s=torch.tensor(2.5), v=torch.randn(32),
            m_u=torch.zeros(32), m_s=torch.zeros(()), m_v=torch.zeros(32),
            vsq_u=torch.zeros(32), vsq_s=torch.zeros(()), vsq_v=torch.zeros(32),
            step=1, spectral_at_sleep=0.3, update_at_sleep=0.2,
            epoch_slept=0,
        )
        ec.add_sleeper("L", sd)
        assert ec.bank["L"][0].u.dtype == torch.float32
        assert torch.allclose(ec.bank["L"][0].u, u, atol=1e-6)
        ec.take_revivals("L", [sd])
        assert ec.bank["L"] == []


class TestPeerReviewFixes:
    """Regresiones de los 5 hallazgos de la revision externa."""

    # --- Fix #1: wake_warmup_steps=0 no debe dejar revivals apagados ---
    def test_warmup_zero_revives_with_full_gate(self, simple_model):
        cfg = _elastic_cfg(elastic_rank_wake_warmup_steps=0)
        engine = ZTrainEngine(simple_model, cfg, device=torch.device("cpu"))
        engine.train_epoch(_loader(), loss_fn, epoch=0)
        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        rank0 = module.rank
        engine.elastic_rank.observe(
            engine.model, engine.optimizer, in_warmup=False, epoch=0)
        for i in range(3):
            module.sleep_mask[i] = True
        p1 = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=1)
        engine._apply_elastic_topology(p1, epoch=1)
        module = dict(engine.model.named_modules())[name]
        rank1 = module.rank
        p2 = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=True, epoch=2)
        assert p2.has_changes()
        engine._apply_elastic_topology(p2, epoch=2)
        module = dict(engine.model.named_modules())[name]
        assert module.rank > rank1
        # Con warmup=0 NINGUNA direccion debe quedar con gate 0 (apagada
        # para siempre): el revival arranca a gate pleno.
        assert float(module.wake_gate.min()) == 1.0
        engine.elastic_rank.tick_wake(engine.model)
        assert float(module.wake_gate.min()) == 1.0

    # --- Fix #2: refactorize resetea el estado por-capa ---
    def test_reset_after_refactorize(self):
        cfg = _elastic_cfg()
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(32, 32, rank=8)
        model = nn.Sequential(layer)
        st = ec._state_for("0", 8)
        st.spectral_ema[:] = 0.01
        layer.sleep_mask[0] = True
        layer.sleep_mask[3] = True
        layer.wake_gate[2] = 0.4
        ec.reset_after_refactorize(model)
        assert not layer.sleep_mask.any()
        assert torch.all(layer.wake_gate == 1.0)
        assert "0" not in ec._states          # se reinicia limpio

    def test_engine_refactorize_resets_elastic(self, simple_model):
        cfg = _elastic_cfg(refactorize_interval=4)
        engine = ZTrainEngine(simple_model, cfg, device=torch.device("cpu"))
        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        module.sleep_mask[0] = True
        # 4 steps -> refactorize_interval dispara refactorize_model + reset
        engine.train_epoch(_loader(n=64, bs=16), loss_fn, epoch=0)
        module = dict(engine.model.named_modules())[name]
        # tras la refactorizacion la mascara antigua no debe seguir
        # congelando una direccion de la base nueva
        assert not bool(module.sleep_mask[0])

    # --- Fix #3: soft_sleep=False NO congela intra-epoch ---
    def test_soft_sleep_false_does_not_freeze(self):
        cfg = _elastic_cfg(elastic_rank_soft_sleep=False)
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(16, 16, rank=5)
        model = nn.Sequential(layer)
        layer.sleep_mask[2] = True
        layer.U.grad = torch.ones_like(layer.U)
        layer.S.grad = torch.ones_like(layer.S)
        layer.V.grad = torch.ones_like(layer.V)
        ec.mask_gradients(model)
        # soft_sleep=False -> mask_gradients es no-op: la direccion sigue
        # entrenando hasta la compactacion en el boundary.
        assert torch.all(layer.U.grad[:, 2] == 1)
        assert float(layer.S.grad[2]) == 1.0

    def test_soft_sleep_true_still_freezes(self):
        cfg = _elastic_cfg(elastic_rank_soft_sleep=True)
        ec = ElasticRankController(cfg)
        layer = ZFactorizedLinear(16, 16, rank=5)
        model = nn.Sequential(layer)
        layer.sleep_mask[2] = True
        layer.U.grad = torch.ones_like(layer.U)
        ec.mask_gradients(model)
        assert torch.all(layer.U.grad[:, 2] == 0)

    # --- Fix #4: flags avanzadas fallan fuerte ---
    def test_advanced_flags_rejected(self):
        with pytest.raises(ValueError):
            ZTrainConfig(use_elastic_rank=True,
                         elastic_rank_allow_rank_neutral_swap=True).validate()
        with pytest.raises(ValueError):
            ZTrainConfig(use_elastic_rank=True,
                         elastic_rank_allow_independent_growth=True).validate()

    # --- Fix #5: validacion de config completa ---
    @pytest.mark.parametrize("kw", [
        {"elastic_rank_sleep_spectral_threshold": 0.0},
        {"elastic_rank_sleep_update_threshold": 0.0},
        {"elastic_rank_sleep_patience_checks": 0},
        {"elastic_rank_min_age_checks": -1},
        {"elastic_rank_post_growth_grace_checks": -1},
        {"elastic_rank_post_refactor_grace_checks": -1},
        {"elastic_rank_compact_min_dirs": 0},
        {"elastic_rank_revive_cooldown_checks": -1},
    ])
    def test_missing_validation_now_enforced(self, kw):
        with pytest.raises(ValueError):
            ZTrainConfig(use_elastic_rank=True, **kw).validate()

    def test_valid_defaults_still_pass(self):
        ZTrainConfig(use_elastic_rank=True).validate()
        ZTrainConfig(use_elastic_rank=True,
                     elastic_rank_wake_warmup_steps=0).validate()


class TestLossSensitivityProbe:
    """Probe diagnostico: ablacion -> rel_delta, sin mutar topologia."""

    def _setup(self, tmp_path, s_values):
        torch.manual_seed(0)
        layer = ZFactorizedLinear(8, 8, rank=len(s_values), bias=False)
        with torch.no_grad():
            layer.S.copy_(torch.tensor(s_values))
        model = nn.Sequential(layer)
        x = torch.randn(4, 8)
        with torch.no_grad():
            target = model(x).clone()

        def loss_eval():
            return float(((model(x) - target) ** 2).mean())

        cfg = _elastic_cfg(
            elastic_rank_probe_loss_sensitivity=True,
            elastic_rank_probe_max_dirs_per_layer=8,
            elastic_rank_probe_output=str(tmp_path / "ls.json"),
        )
        ec = ElasticRankController(cfg)
        opt = ZCompressedAdam(model.parameters(), lr=1e-3)
        return ec, model, opt, loss_eval, layer

    def test_detects_heterogeneity(self, tmp_path):
        # Espectro muy heterogeneo -> rel_delta debe serlo tambien.
        ec, model, opt, loss_eval, layer = self._setup(
            tmp_path, [10.0, 1.0, 0.1, 0.01])
        rec = ec.probe_loss_sensitivity(model, opt, loss_eval, epoch=0)
        assert rec["loss_base"] == pytest.approx(0.0, abs=1e-5)
        assert rec["global"]["p90_p10_ratio"] > 2.0
        assert rec["verdict"]["heterogeneous"] is True
        lay = rec["layers"][0]
        # rel_delta crece con S -> Spearman fuerte positivo vs spectral.
        assert lay["spearman_vs_spectral"] is not None
        assert lay["spearman_vs_spectral"] >= 0.5

    def test_does_not_mutate_topology(self, tmp_path):
        ec, model, opt, loss_eval, layer = self._setup(
            tmp_path, [3.0, 2.0, 1.0, 0.5])
        U0 = layer.U.detach().clone()
        S0 = layer.S.detach().clone()
        V0 = layer.V.detach().clone()
        wg0 = layer.wake_gate.detach().clone()
        sm0 = layer.sleep_mask.detach().clone()
        model.train()
        ec.probe_loss_sensitivity(model, opt, loss_eval, epoch=0)
        assert layer.rank == 4
        assert torch.equal(layer.U.detach(), U0)
        assert torch.equal(layer.S.detach(), S0)
        assert torch.equal(layer.V.detach(), V0)
        assert torch.equal(layer.wake_gate, wg0)      # gate restaurado exacto
        assert torch.equal(layer.sleep_mask, sm0)
        assert model.training                          # modo restaurado
        assert ec.bank == {}                           # nada dormido

    def test_writes_json(self, tmp_path):
        ec, model, opt, loss_eval, _ = self._setup(tmp_path, [2.0, 1.0, 0.3])
        ec.probe_loss_sensitivity(model, opt, loss_eval, epoch=0)
        ec.probe_loss_sensitivity(model, opt, loss_eval, epoch=2)
        import json
        data = json.loads((tmp_path / "ls.json").read_text())
        assert isinstance(data, list) and len(data) == 2
        assert data[0]["epoch"] == 0 and data[1]["epoch"] == 2

    def test_disabled_by_default(self, simple_model, tmp_path):
        engine = ZTrainEngine(simple_model, _elastic_cfg(),
                              device=torch.device("cpu"))
        engine.train(_loader(), None, loss_fn, epochs=2)
        assert engine.elastic_rank._probe_history == []

def _active_fp(model):
    return sum(m.U.numel() + m.S.numel() + m.V.numel()
               for m in model.modules() if isinstance(m, ZFactorizedLinear))


class TestLossGuard:
    """Rollback de compactacion por perdida (nucleo de decision v3).

    La loss del batch sonda se ESTUBA para que la relacion antes/despues
    sea determinista (lo que se prueba es la logica del guard, no las
    dinamicas de entrenamiento, que con threshold=0 son knife-edge)."""

    def _engine_ready(self, simple_model, probe_loss, **guard_kw):
        cfg = _elastic_cfg(elastic_rank_loss_guard_enabled=True, **guard_kw)
        engine = ZTrainEngine(simple_model, cfg, device=torch.device("cpu"))
        engine.train_epoch(_loader(), loss_fn, epoch=0)   # estado optimizer
        engine._probe_batch = torch.zeros(1)              # placeholder no-None
        engine._loss_fn = probe_loss                      # estub determinista
        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        engine.elastic_rank.observe(
            engine.model, engine.optimizer, in_warmup=False, epoch=0)
        for i in range(8):
            module.sleep_mask[i] = True
        return engine, name, module

    @staticmethod
    def _loss_up_on_compaction(model, _batch):
        # menos params activos -> loss mayor: compactar SIEMPRE sube la loss.
        return 1.0e6 / max(_active_fp(model), 1)

    @staticmethod
    def _loss_constant(model, _batch):
        return 1.0

    def test_reverts_on_loss_spike(self, simple_model):
        engine, name, module = self._engine_ready(
            simple_model, self._loss_up_on_compaction,
            elastic_rank_loss_guard_threshold=0.0,
            elastic_rank_loss_guard_cooldown_epochs=2)
        rank0 = module.rank
        plan = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=1)
        assert any(len(lp.sleep) > 0 for _n, lp in plan.changed_layers())
        engine._apply_elastic_topology_guarded(plan, epoch=1)
        module = dict(engine.model.named_modules())[name]
        assert module.rank == rank0                       # revertido
        assert engine.elastic_rank._total_rollbacks == 1
        assert engine.elastic_rank._rollback_cd.get(name, 0) > 0
        out = engine.model(torch.randn(4, 128))
        assert torch.isfinite(out).all()

    def test_confirms_when_within_threshold(self, simple_model):
        # loss constante -> rel_delta = 0, no supera threshold 0 -> confirma.
        engine, name, module = self._engine_ready(
            simple_model, self._loss_constant,
            elastic_rank_loss_guard_threshold=0.0)
        rank0 = module.rank
        plan = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=1)
        engine._apply_elastic_topology_guarded(plan, epoch=1)
        module = dict(engine.model.named_modules())[name]
        assert module.rank < rank0                        # confirmado
        assert engine.elastic_rank._total_rollbacks == 0

    def test_cooldown_blocks_then_releases(self, simple_model):
        engine, name, module = self._engine_ready(
            simple_model, self._loss_up_on_compaction,
            elastic_rank_loss_guard_threshold=0.0,
            elastic_rank_loss_guard_cooldown_epochs=2)
        rank0 = module.rank
        p1 = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=1)
        engine._apply_elastic_topology_guarded(p1, epoch=1)
        assert engine.elastic_rank._total_rollbacks == 1
        # cooldown_epochs=2 -> los 2 siguientes build_plan no tocan la capa
        p2 = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=2)
        assert not p2.layers[name].changed
        p3 = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=3)
        assert not p3.layers[name].changed
        # ya liberado: vuelve a poder proponer compactacion
        p4 = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=4)
        assert p4.layers[name].changed

    def test_disabled_applies_unconditionally(self, simple_model):
        cfg = _elastic_cfg(elastic_rank_loss_guard_enabled=False)
        engine = ZTrainEngine(simple_model, cfg, device=torch.device("cpu"))
        engine.train_epoch(_loader(), loss_fn, epoch=0)
        name, module = next(engine.elastic_rank._iter_layers(engine.model))
        engine.elastic_rank.observe(
            engine.model, engine.optimizer, in_warmup=False, epoch=0)
        for i in range(8):
            module.sleep_mask[i] = True
        rank0 = module.rank
        plan = engine.elastic_rank.build_plan(
            engine.model, rank0, grow_requested=False, epoch=1)
        engine._apply_elastic_topology_guarded(plan, epoch=1)
        module = dict(engine.model.named_modules())[name]
        assert module.rank < rank0                        # sin guard, aplica
        assert engine.elastic_rank._total_rollbacks == 0

    def test_config_validation(self):
        for kw in (
            {"elastic_rank_loss_guard_mode": "trained_batches"},
            {"elastic_rank_loss_guard_threshold": -0.1},
            {"elastic_rank_loss_guard_cooldown_epochs": -1},
            {"elastic_rank_loss_guard_batches": 0},
        ):
            with pytest.raises(ValueError):
                ZTrainConfig(use_elastic_rank=True,
                             elastic_rank_loss_guard_enabled=True,
                             **kw).validate()
        ZTrainConfig(use_elastic_rank=True,
                     elastic_rank_loss_guard_enabled=True).validate()


class TestProbeBooleanFix:
    """El veredicto no debe marcar heterogeneo cuando el efecto absoluto
    es ~0 (antes el CV puro daba falsos positivos)."""

    def test_tiny_effect_not_heterogeneous(self, tmp_path):
        torch.manual_seed(0)
        layer = ZFactorizedLinear(8, 8, rank=4, bias=False)
        with torch.no_grad():
            layer.S.copy_(torch.tensor([1e-4, 1e-4, 1e-4, 1e-4]))
        model = nn.Sequential(layer)
        x = torch.randn(4, 8)
        target = torch.randn(4, 8)            # distinto -> loss_base sizable

        def loss_eval():
            return float(((model(x) - target) ** 2).mean())

        cfg = _elastic_cfg(
            elastic_rank_probe_loss_sensitivity=True,
            elastic_rank_probe_output=str(tmp_path / "t.json"))
        ec = ElasticRankController(cfg)
        opt = ZCompressedAdam(model.parameters(), lr=1e-3)
        rec = ec.probe_loss_sensitivity(model, opt, loss_eval, epoch=0)
        assert rec["loss_base"] > 0.1                      # base no es ~0
        assert rec["verdict"]["p90_abs_rel_delta"] < 1e-3
        assert rec["verdict"]["heterogeneous"] is False


class TestProbeEngineIntegration:
    """El probe corre end-to-end dentro de engine.train()."""

    def test_engine_runs_probe(self, simple_model, tmp_path):
        cfg = _elastic_cfg(
            elastic_rank_probe_loss_sensitivity=True,
            elastic_rank_probe_interval_epochs=1,
            elastic_rank_probe_max_dirs_per_layer=4,
            elastic_rank_probe_output=str(tmp_path / "p.json"),
        )
        engine = ZTrainEngine(simple_model, cfg, device=torch.device("cpu"))
        engine.train(_loader(), None, loss_fn, epochs=2)
        assert len(engine.elastic_rank._probe_history) >= 1
        assert (tmp_path / "p.json").exists()
        rec = engine.elastic_rank._probe_history[0]
        assert "verdict" in rec and "global_spearman" in rec
