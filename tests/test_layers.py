"""
Tests para mztrain.layers - Capas factorizadas.
"""

import pytest
import torch
import torch.nn as nn

from mztrain.layers import (
    ZFactorizedLinear,
    ZFactorizedAttention,
    ZFactorizedTransformerBlock,
)


class TestZFactorizedLinear:
    """Tests para ZFactorizedLinear."""

    def test_creation_default(self):
        layer = ZFactorizedLinear(128, 64, rank=16)
        assert layer.in_features == 128
        assert layer.out_features == 64
        assert layer.rank == 16
        assert layer.bias is not None

    def test_creation_no_bias(self):
        layer = ZFactorizedLinear(128, 64, rank=16, bias=False)
        assert layer.bias is None

    def test_rank_clamping(self):
        layer = ZFactorizedLinear(32, 16, rank=64)
        assert layer.rank == 16  # min(32, 16) = 16

    def test_forward_shape(self):
        layer = ZFactorizedLinear(128, 64, rank=16)
        x = torch.randn(8, 128)
        y = layer(x)
        assert y.shape == (8, 64)

    def test_forward_3d(self):
        layer = ZFactorizedLinear(128, 64, rank=16)
        x = torch.randn(4, 10, 128)
        y = layer(x)
        assert y.shape == (4, 10, 64)

    def test_forward_count(self):
        layer = ZFactorizedLinear(128, 64, rank=16)
        x = torch.randn(4, 128)
        for _ in range(5):
            layer(x)
        assert layer._forward_count == 5

    def test_compression_ratio(self):
        layer = ZFactorizedLinear(768, 768, rank=64)
        assert layer.compression_ratio < 1.0
        assert layer.compression_ratio > 0.0

    def test_num_parameters(self):
        layer = ZFactorizedLinear(768, 768, rank=64)
        expected = 768 * 64 + 64 + 64 * 768 + 768  # U + S + V + bias
        assert layer.num_parameters == expected

    def test_full_parameters(self):
        layer = ZFactorizedLinear(768, 768, rank=64)
        expected = 768 * 768 + 768  # W + bias
        assert layer.full_parameters == expected

    def test_init_methods(self):
        for method in ["svd", "kaiming", "random"]:
            layer = ZFactorizedLinear(64, 64, rank=16, init_method=method)
            x = torch.randn(4, 64)
            y = layer(x)
            assert y.shape == (4, 64)
            assert not torch.isnan(y).any()

    def test_init_from_weight(self):
        original = nn.Linear(64, 32)
        W = original.weight.data.clone()
        layer = ZFactorizedLinear(
            64, 32, rank=16, existing_weight=W
        )
        assert layer._reconstruction_error >= 0.0
        x = torch.randn(4, 64)
        y = layer(x)
        assert y.shape == (4, 32)

    def test_reconstruct_weight(self):
        layer = ZFactorizedLinear(64, 32, rank=16)
        W = layer.reconstruct_weight()
        assert W.shape == (32, 64)

    def test_grow_rank(self):
        layer = ZFactorizedLinear(64, 64, rank=8)
        assert layer.rank == 8
        layer.grow_rank(16)
        assert layer.rank == 16
        assert layer.U.shape == (64, 16)
        assert layer.S.shape == (16,)
        assert layer.V.shape == (16, 64)

    def test_grow_rank_preserves_output(self):
        layer = ZFactorizedLinear(64, 64, rank=8)
        x = torch.randn(4, 64)
        y_before = layer(x).detach().clone()
        layer.grow_rank(16, preserve_weights=True)
        y_after = layer(x).detach()
        # Deberia ser similar (nuevos componentes son pequenos)
        assert torch.allclose(y_before, y_after, atol=0.1)

    def test_grow_rank_no_op(self):
        layer = ZFactorizedLinear(64, 64, rank=16)
        layer.grow_rank(8)  # Menor que actual
        assert layer.rank == 16  # Sin cambio

    def test_grow_rank_max_clamp(self):
        layer = ZFactorizedLinear(32, 16, rank=8)
        layer.grow_rank(64)  # Mayor que max posible
        assert layer.rank == 16  # min(32, 16)

    def test_get_stats(self):
        layer = ZFactorizedLinear(128, 64, rank=16)
        stats = layer.get_stats()
        assert "shape" in stats
        assert "rank" in stats
        assert "compression_ratio" in stats
        assert "memory_saved_pct" in stats

    def test_extra_repr(self):
        layer = ZFactorizedLinear(128, 64, rank=16)
        r = layer.extra_repr()
        assert "in=128" in r
        assert "out=64" in r
        assert "rank=16" in r

    def test_gradient_flow(self):
        layer = ZFactorizedLinear(64, 32, rank=16)
        x = torch.randn(4, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.U.grad is not None
        assert layer.S.grad is not None
        assert layer.V.grad is not None
        assert x.grad is not None


class TestZFactorizedAttention:
    """Tests para ZFactorizedAttention."""

    def test_creation(self):
        attn = ZFactorizedAttention(embed_dim=64, num_heads=4, rank=8)
        assert attn.embed_dim == 64
        assert attn.num_heads == 4
        assert attn.head_dim == 16

    def test_invalid_heads(self):
        with pytest.raises(AssertionError):
            ZFactorizedAttention(embed_dim=64, num_heads=3)

    def test_forward_shape(self):
        attn = ZFactorizedAttention(embed_dim=64, num_heads=4, rank=8)
        x = torch.randn(2, 10, 64)
        y = attn(x)
        assert y.shape == (2, 10, 64)

    def test_forward_with_mask(self):
        attn = ZFactorizedAttention(embed_dim=64, num_heads=4, rank=8)
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 4, 10, 10)
        y = attn(x, mask=mask)
        assert y.shape == (2, 10, 64)

    def test_grow_rank(self):
        attn = ZFactorizedAttention(embed_dim=64, num_heads=4, rank=8)
        attn.grow_rank(16)
        assert attn.q_proj.rank == 16
        assert attn.k_proj.rank == 16
        assert attn.v_proj.rank == 16
        assert attn.out_proj.rank == 16

    def test_gradient_flow(self):
        attn = ZFactorizedAttention(embed_dim=64, num_heads=4, rank=8)
        x = torch.randn(2, 10, 64, requires_grad=True)
        y = attn(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestZFactorizedTransformerBlock:
    """Tests para ZFactorizedTransformerBlock."""

    def test_creation(self):
        block = ZFactorizedTransformerBlock(
            embed_dim=64, num_heads=4, rank=8
        )
        assert block.norm1 is not None
        assert block.attention is not None
        assert block.norm2 is not None
        assert block.mlp is not None

    def test_forward_shape(self):
        block = ZFactorizedTransformerBlock(
            embed_dim=64, num_heads=4, rank=8
        )
        x = torch.randn(2, 10, 64)
        y = block(x)
        assert y.shape == (2, 10, 64)

    def test_residual_connection(self):
        block = ZFactorizedTransformerBlock(
            embed_dim=64, num_heads=4, rank=8, dropout=0.0
        )
        x = torch.randn(2, 10, 64)
        y = block(x)
        # Output should differ from input (not just passthrough)
        assert not torch.allclose(x, y)

    def test_grow_rank(self):
        block = ZFactorizedTransformerBlock(
            embed_dim=64, num_heads=4, rank=8
        )
        block.grow_rank(16)
        assert block.attention.q_proj.rank == 16

    def test_activations(self):
        for act in ["gelu", "relu"]:
            block = ZFactorizedTransformerBlock(
                embed_dim=64, num_heads=4, rank=8, activation=act
            )
            x = torch.randn(2, 10, 64)
            y = block(x)
            assert y.shape == (2, 10, 64)
