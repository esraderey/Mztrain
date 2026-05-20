"""
Tests para mztrain.checkpoint - Activation checkpointing.
"""

import pytest
import torch

from mztrain.checkpoint import ZActivationCheckpoint, z_checkpoint


class TestZActivationCheckpoint:
    """Tests para ZActivationCheckpoint."""

    def setup_method(self):
        """Reset antes de cada test."""
        ZActivationCheckpoint.reset()

    def test_compress_decompress(self):
        tensor = torch.randn(4, 64)
        key, shape, dtype = ZActivationCheckpoint.compress_activation(tensor)
        recovered = ZActivationCheckpoint.decompress_activation(key, shape, dtype)
        # Con fallback INT8, no es exacto pero cercano
        assert recovered.shape == tensor.shape
        assert torch.allclose(tensor, recovered, atol=0.1)

    def test_stats(self):
        tensor = torch.randn(4, 64)
        ZActivationCheckpoint.compress_activation(tensor)
        stats = ZActivationCheckpoint.get_stats()
        assert stats["total_compressed"] == 1

    def test_multiple_compress(self):
        for i in range(5):
            tensor = torch.randn(4, 64)
            ZActivationCheckpoint.compress_activation(tensor)
        stats = ZActivationCheckpoint.get_stats()
        assert stats["total_compressed"] == 5

    def test_reset(self):
        tensor = torch.randn(4, 64)
        ZActivationCheckpoint.compress_activation(tensor)
        ZActivationCheckpoint.reset()
        stats = ZActivationCheckpoint.get_stats()
        assert stats["total_compressed"] == 0

    def test_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="not found"):
            ZActivationCheckpoint.decompress_activation(
                "nonexistent_key", torch.Size([4, 64]), torch.float32
            )

    def test_dtype_preservation(self):
        for dtype in [torch.float32, torch.float64]:
            ZActivationCheckpoint.reset()
            tensor = torch.randn(4, 64, dtype=dtype)
            key, shape, dt = ZActivationCheckpoint.compress_activation(tensor)
            assert dt == dtype


class TestZCheckpoint:
    """Tests para z_checkpoint function."""

    def setup_method(self):
        ZActivationCheckpoint.reset()

    def test_basic_checkpoint(self):
        import torch.nn as nn
        layer = nn.Linear(64, 32)
        x = torch.randn(4, 64, requires_grad=True)
        out = z_checkpoint(layer, x)
        assert out.shape == (4, 32)

    def test_gradient_through_checkpoint(self):
        import torch.nn as nn
        layer = nn.Linear(64, 32)
        x = torch.randn(4, 64, requires_grad=True)
        out = z_checkpoint(layer, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
