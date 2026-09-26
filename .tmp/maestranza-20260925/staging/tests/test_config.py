"""
Tests para mztrain.config - Configuracion y enumeraciones.
"""

import pytest

from mztrain.config import (
    ZTrainConfig,
    RankSchedule,
    GradientCompression,
)


class TestRankSchedule:
    """Tests para RankSchedule enum."""

    def test_values(self):
        assert RankSchedule.CONSTANT.value == "constant"
        assert RankSchedule.LINEAR.value == "linear"
        assert RankSchedule.EXPONENTIAL.value == "exponential"
        assert RankSchedule.COSINE.value == "cosine"
        assert RankSchedule.ADAPTIVE.value == "adaptive"
        assert RankSchedule.SPECTRAL.value == "spectral"

    def test_all_schedules(self):
        assert len(RankSchedule) == 6


class TestGradientCompression:
    """Tests para GradientCompression enum."""

    def test_values(self):
        assert GradientCompression.NONE.value == "none"
        assert GradientCompression.TOP_K.value == "top_k"
        assert GradientCompression.QUANTIZE_1BIT.value == "1bit"
        assert GradientCompression.QUANTIZE_INT8.value == "int8"
        assert GradientCompression.SVD.value == "svd"


class TestZTrainConfig:
    """Tests para ZTrainConfig."""

    def test_defaults(self):
        config = ZTrainConfig()
        assert config.initial_rank == 32
        assert config.max_rank == 256
        assert config.min_params_to_factorize == 4096
        assert config.learning_rate == 3e-4
        assert config.use_amp is True

    def test_custom_values(self):
        config = ZTrainConfig(
            initial_rank=16,
            max_rank=128,
            learning_rate=1e-3,
        )
        assert config.initial_rank == 16
        assert config.max_rank == 128
        assert config.learning_rate == 1e-3

    def test_validate_ok(self):
        config = ZTrainConfig()
        config.validate()  # No deberia lanzar excepcion

    def test_validate_invalid_rank(self):
        config = ZTrainConfig(initial_rank=0)
        with pytest.raises(ValueError, match="initial_rank"):
            config.validate()

    def test_validate_max_less_than_initial(self):
        config = ZTrainConfig(initial_rank=64, max_rank=32)
        with pytest.raises(ValueError, match="max_rank"):
            config.validate()

    def test_validate_invalid_lr(self):
        config = ZTrainConfig(learning_rate=-1e-3)
        with pytest.raises(ValueError, match="learning_rate"):
            config.validate()

    def test_validate_invalid_energy(self):
        config = ZTrainConfig(energy_retention=1.5)
        with pytest.raises(ValueError, match="energy_retention"):
            config.validate()

    def test_validate_invalid_grad_norm(self):
        config = ZTrainConfig(max_grad_norm=0)
        with pytest.raises(ValueError, match="max_grad_norm"):
            config.validate()

    def test_validate_invalid_top_k(self):
        config = ZTrainConfig(gradient_top_k_ratio=0)
        with pytest.raises(ValueError, match="gradient_top_k_ratio"):
            config.validate()

    def test_schedule_types(self):
        config = ZTrainConfig(rank_schedule=RankSchedule.COSINE)
        assert config.rank_schedule == RankSchedule.COSINE

    def test_gradient_compression_types(self):
        config = ZTrainConfig(gradient_compression=GradientCompression.TOP_K)
        assert config.gradient_compression == GradientCompression.TOP_K
