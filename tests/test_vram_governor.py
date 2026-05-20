"""
Tests para mztrain.vram_governor - ZVRAMGovernor (MVP).

Sin GPU en el entorno: la logica (EMA, modos, histeresis, gating de growth,
OOM-retry) se prueba con un LECTOR DE MEMORIA INYECTADO. La ruta CUDA real
queda guardada (inerte sin GPU) y no se testea aqui a proposito.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mztrain.config import ZTrainConfig, RankSchedule
from mztrain.engine import ZTrainEngine
from mztrain.vram_governor import ZVRAMGovernor


def _cfg(**kw):
    base = dict(
        use_vram_governor=True,
        vram_governor_interval=1,
        vram_governor_ema_beta=0.0,          # ema = lectura (determinista)
        vram_preventive_threshold=0.90,
        vram_emergency_threshold=0.98,
        vram_hysteresis_checks=3,
        vram_oom_retry=True,
    )
    base.update(kw)
    return ZTrainConfig(**base)


def _reader(pressure_frac):
    """Lector que produce reserved/total = pressure_frac (peak menor)."""
    total = 1_000_000
    return lambda: {
        "allocated": int(0.5 * pressure_frac * total),
        "reserved": int(pressure_frac * total),
        "peak": int(0.5 * pressure_frac * total),
        "free": int((1.0 - pressure_frac) * total),
        "total": total,
    }


class TestSensorAndInert:
    def test_sensor_path_matches_cuda_availability(self):
        # Con CUDA: observe() devuelve snapshot real. Sin CUDA: None (inerte).
        g = ZVRAMGovernor(_cfg())
        snap = g.observe(device=None)
        if torch.cuda.is_available():
            assert snap is not None and snap.total > 0
        else:
            assert snap is None
            assert g.block_growth is False
            assert g.approve_rank_growth(64, 32) == 64   # pass-through

    def test_snapshot_fields_and_pressure(self):
        g = ZVRAMGovernor(_cfg())
        snap = g.observe(mem_reader=_reader(0.5))
        assert snap is not None
        assert snap.total == 1_000_000
        assert snap.pressure == pytest.approx(0.5)      # max(0.5, 0.25)
        assert snap.mode == "normal"
        assert g.block_growth is False


class TestModesAndHysteresis:
    def test_normal_low_pressure(self):
        g = ZVRAMGovernor(_cfg())
        for _ in range(5):
            g.observe(mem_reader=_reader(0.30))
        assert g.mode == "normal"
        assert g.block_growth is False

    def test_preventive_requires_hysteresis(self):
        g = ZVRAMGovernor(_cfg(vram_hysteresis_checks=3))
        rd = _reader(0.93)                                # > preventive 0.90
        g.observe(mem_reader=rd)
        assert g.mode == "normal"                         # 1 chequeo: no aun
        g.observe(mem_reader=rd)
        assert g.mode == "normal"                         # 2: no aun
        g.observe(mem_reader=rd)
        assert g.mode == "preventive"                     # 3 consecutivos
        assert g.block_growth is True

    def test_single_spike_does_not_flip(self):
        g = ZVRAMGovernor(_cfg(vram_hysteresis_checks=3))
        g.observe(mem_reader=_reader(0.30))
        g.observe(mem_reader=_reader(0.95))               # spike aislado
        g.observe(mem_reader=_reader(0.30))
        g.observe(mem_reader=_reader(0.30))
        assert g.mode == "normal"
        assert g.block_growth is False

    def test_emergency_band(self):
        g = ZVRAMGovernor(_cfg(vram_hysteresis_checks=2))
        rd = _reader(0.99)                                # > emergency 0.98
        g.observe(mem_reader=rd)
        g.observe(mem_reader=rd)
        assert g.mode == "emergency"
        assert g.block_growth is True


class TestGrowthGating:
    def test_blocks_growth_under_pressure(self):
        g = ZVRAMGovernor(_cfg(vram_hysteresis_checks=1))
        g.observe(mem_reader=_reader(0.93))
        assert g.block_growth is True
        assert g.approve_rank_growth(128, 64) == 64       # bloqueado
        assert g._counters["blocked_grows"] == 1

    def test_allows_growth_when_calm(self):
        g = ZVRAMGovernor(_cfg(vram_hysteresis_checks=1))
        g.observe(mem_reader=_reader(0.40))
        assert g.approve_rank_growth(128, 64) == 128      # permitido
        assert g._counters["allowed_grows"] == 1

    def test_no_growth_request_is_passthrough(self):
        g = ZVRAMGovernor(_cfg(vram_hysteresis_checks=1))
        g.observe(mem_reader=_reader(0.93))               # block_growth True
        # requested <= current: no es crecimiento, no se cuenta ni bloquea
        assert g.approve_rank_growth(32, 64) == 32
        assert g._counters["blocked_grows"] == 0
        assert g._counters["allowed_grows"] == 0


class TestOOMGuard:
    def _raiser(self, fail_times):
        state = {"n": 0}

        def fn():
            if state["n"] < fail_times:
                state["n"] += 1
                raise torch.cuda.OutOfMemoryError("simulado")
            return "ok"
        return fn

    def test_recovers_after_single_oom(self):
        g = ZVRAMGovernor(_cfg())
        assert g.oom_guarded(self._raiser(1)) == "ok"
        assert g._counters["oom_events"] == 1
        assert g._counters["oom_recoveries"] == 1

    def test_reraises_if_retry_also_ooms(self):
        g = ZVRAMGovernor(_cfg())
        with pytest.raises(torch.cuda.OutOfMemoryError):
            g.oom_guarded(self._raiser(2))
        assert g._counters["oom_events"] == 1
        assert g._counters["oom_recoveries"] == 0

    def test_retry_disabled_reraises_immediately(self):
        g = ZVRAMGovernor(_cfg(vram_oom_retry=False))
        with pytest.raises(torch.cuda.OutOfMemoryError):
            g.oom_guarded(self._raiser(1))
        assert g._counters["oom_events"] == 1


class TestConfigValidation:
    @pytest.mark.parametrize("kw", [
        {"vram_governor_interval": 0},
        {"vram_governor_ema_beta": 1.0},
        {"vram_preventive_threshold": 0.0},
        {"vram_emergency_threshold": 1.5},
        {"vram_emergency_threshold": 0.80,   # < preventive 0.90
         "vram_preventive_threshold": 0.90},
        {"vram_hysteresis_checks": 0},
    ])
    def test_invalid_rejected(self, kw):
        with pytest.raises(ValueError):
            _cfg(**kw).validate()

    def test_defaults_valid(self):
        _cfg().validate()
        ZTrainConfig(use_vram_governor=True).validate()
        ZTrainConfig().validate()            # OFF por defecto, sin checks


def _loader():
    x = torch.randn(64, 128)
    y = torch.randint(0, 10, (64,))
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def _loss(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)


class TestEngineIntegration:
    def test_disabled_by_default(self, simple_model):
        engine = ZTrainEngine(simple_model, ZTrainConfig(initial_rank=8),
                              device=torch.device("cpu"))
        assert engine.vram_governor is None
        s = engine.get_training_summary()
        assert s["governor_stats"] is None

    @staticmethod
    def _fixed_mem(frac):
        # Lector inyectado (independiente del GPU real / peak global del
        # proceso) para que la integracion sea determinista.
        total = 1_000_000

        def rd(device=None, mem_reader=None):
            return {
                "allocated": int(0.5 * frac * total),
                "reserved": int(frac * total),
                "peak": int(0.5 * frac * total),
                "free": int((1.0 - frac) * total),
                "total": total,
            }
        return rd

    def test_low_pressure_allows_growth(self, simple_model):
        cfg = _cfg(initial_rank=8, max_rank=32, use_amp=False,
                   vram_hysteresis_checks=1,
                   rank_schedule=RankSchedule.EXPONENTIAL,
                   rank_growth_interval=1, rank_growth_factor=2.0)
        engine = ZTrainEngine(simple_model, cfg,
                              device=torch.device("cpu"))
        assert isinstance(engine.vram_governor, ZVRAMGovernor)
        engine.vram_governor._read_memory = self._fixed_mem(0.30)
        summary = engine.train(_loader(), None, _loss, epochs=3)
        assert summary["ranks"][-1] > 8                  # crecio
        assert summary["governor_stats"] is not None
        assert summary["governor_stats"]["actions"]["blocked_grows"] == 0

    def test_high_pressure_blocks_growth(self, simple_model):
        cfg = _cfg(initial_rank=8, max_rank=64, use_amp=False,
                   vram_hysteresis_checks=1,
                   rank_schedule=RankSchedule.EXPONENTIAL,
                   rank_growth_interval=1, rank_growth_factor=2.0)
        engine = ZTrainEngine(simple_model, cfg,
                              device=torch.device("cpu"))
        # Presion alta inyectada -> el governor decide bloquear growth.
        engine.vram_governor._read_memory = self._fixed_mem(0.95)
        summary = engine.train(_loader(), None, _loss, epochs=3)
        assert summary["ranks"][-1] == 8                 # crecimiento vetado
        assert summary["governor_stats"]["actions"]["blocked_grows"] >= 1
        assert summary["governor_stats"]["mode"] in (
            "preventive", "emergency")
