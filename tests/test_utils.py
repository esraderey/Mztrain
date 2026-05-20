"""
Tests para mztrain.utils - Utilidades de conversion y estimacion.
"""

import pytest
import torch
import torch.nn as nn

from mztrain.utils import factorize_existing_model, estimate_memory_savings
from mztrain.layers import ZFactorizedLinear


class TestFactorizeExistingModel:
    """Tests para factorize_existing_model."""

    def test_basic_factorization(self, simple_model):
        z_model, stats = factorize_existing_model(simple_model, rank=16)
        assert stats["total_savings_pct"] > 0
        assert len(stats["replaced_layers"]) > 0

    def test_stats_contents(self, simple_model):
        _, stats = factorize_existing_model(simple_model, rank=16)
        assert "original_params" in stats
        assert "factorized_params" in stats
        assert "replaced_layers" in stats
        assert "skipped_layers" in stats
        assert "total_savings_pct" in stats

    def test_replaced_layer_info(self, simple_model):
        _, stats = factorize_existing_model(simple_model, rank=16)
        for layer in stats["replaced_layers"]:
            assert "name" in layer
            assert "original_params" in layer
            assert "factorized_params" in layer
            assert "compression" in layer
            assert "rank" in layer

    def test_high_min_params(self):
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        _, stats = factorize_existing_model(model, rank=4, min_params=10000)
        assert len(stats["replaced_layers"]) == 0
        assert len(stats["skipped_layers"]) == 2

    def test_exclude_patterns(self, simple_model):
        _, stats = factorize_existing_model(
            simple_model, rank=16,
            exclude_patterns=["0"],  # Excluir primera capa
        )
        replaced_names = [l["name"] for l in stats["replaced_layers"]]
        assert "0" not in replaced_names

    def test_model_still_works(self, simple_model):
        z_model, _ = factorize_existing_model(simple_model, rank=16)
        x = torch.randn(4, 128)
        y = z_model(x)
        assert y.shape == (4, 10)

    def test_deepcopy(self, simple_model):
        original_params = sum(p.numel() for p in simple_model.parameters())
        z_model, _ = factorize_existing_model(simple_model, rank=16)
        # Original no deberia haber cambiado
        assert sum(p.numel() for p in simple_model.parameters()) == original_params


class TestEstimateMemorySavings:
    """Tests para estimate_memory_savings."""

    def test_basic_estimate(self, simple_model):
        result = estimate_memory_savings(simple_model, rank=16)
        assert "traditional" in result
        assert "mztrain" in result
        assert "savings" in result
        assert "details" in result

    def test_traditional_values(self, simple_model):
        result = estimate_memory_savings(simple_model, rank=16)
        trad = result["traditional"]
        assert trad["params_mb"] > 0
        assert trad["gradients_mb"] > 0
        assert trad["optimizer_mb"] > 0
        assert trad["total_mb"] > 0

    def test_savings_positive(self, simple_model):
        result = estimate_memory_savings(simple_model, rank=16)
        assert result["savings"]["total_pct"] > 0
        assert result["savings"]["factor"] > 1.0

    def test_mztrain_less_than_traditional(self, simple_model):
        result = estimate_memory_savings(simple_model, rank=16)
        assert result["mztrain"]["total_mb"] < result["traditional"]["total_mb"]

    def test_high_rank_less_savings(self, simple_model):
        low_rank = estimate_memory_savings(simple_model, rank=8)
        high_rank = estimate_memory_savings(simple_model, rank=64)
        assert low_rank["savings"]["total_pct"] > high_rank["savings"]["total_pct"]

    def test_details(self, simple_model):
        result = estimate_memory_savings(simple_model, rank=16)
        details = result["details"]
        assert details["total_params"] > 0
        assert details["rank"] == 16

    def test_no_factorizable(self):
        model = nn.Sequential(nn.Linear(4, 2))  # Muy pequeno
        result = estimate_memory_savings(model, rank=1, min_params=100)
        assert result["details"]["factorizable_params"] == 0
