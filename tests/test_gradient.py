"""
Tests para mztrain.gradient - Compresion de gradientes.
"""

import pytest
import torch

from mztrain.config import GradientCompression
from mztrain.gradient import ZGradientCompressor


class TestZGradientCompressor:
    """Tests para ZGradientCompressor."""

    def test_no_compression(self):
        compressor = ZGradientCompressor(method=GradientCompression.NONE)
        grad = torch.randn(64, 64)
        result = compressor.compress("test", grad)
        assert torch.equal(grad, result)

    def test_top_k(self):
        compressor = ZGradientCompressor(
            method=GradientCompression.TOP_K,
            top_k_ratio=0.1,
            min_size_to_compress=0,
        )
        grad = torch.randn(100)
        result = compressor.compress("test", grad)
        # Solo 10% deberia ser no-cero
        nonzero = (result != 0).sum().item()
        assert nonzero == 10

    def test_one_bit(self):
        compressor = ZGradientCompressor(
            method=GradientCompression.QUANTIZE_1BIT,
            min_size_to_compress=0,
        )
        grad = torch.randn(100)
        result = compressor.compress("test", grad)
        # Todos los valores deberian tener la misma magnitud
        unique_abs = result.abs().unique()
        assert len(unique_abs) <= 2  # 0 y magnitude

    def test_int8(self):
        compressor = ZGradientCompressor(
            method=GradientCompression.QUANTIZE_INT8,
        )
        grad = torch.randn(100)
        result = compressor.compress("test", grad)
        # Deberia ser similar al original
        assert torch.allclose(grad, result, atol=0.05)

    def test_svd_2d(self):
        compressor = ZGradientCompressor(
            method=GradientCompression.SVD,
            svd_rank=4,
        )
        grad = torch.randn(32, 32)
        result = compressor.compress("test", grad)
        assert result.shape == grad.shape

    def test_svd_1d_fallback(self):
        compressor = ZGradientCompressor(
            method=GradientCompression.SVD,
        )
        grad = torch.randn(100)  # 1D, no se puede SVD
        result = compressor.compress("test", grad)
        assert result.shape == grad.shape

    def test_error_feedback(self):
        compressor = ZGradientCompressor(
            method=GradientCompression.TOP_K,
            top_k_ratio=0.1,
            min_size_to_compress=0,
        )
        grad1 = torch.randn(100)
        compressor.compress("layer1", grad1)
        # Deberia tener error feedback guardado
        assert "layer1" in compressor._error_feedback

    def test_stats(self):
        compressor = ZGradientCompressor(
            method=GradientCompression.TOP_K,
            top_k_ratio=0.5,
            min_size_to_compress=0,
        )
        for i in range(5):
            grad = torch.randn(100)
            compressor.compress(f"layer{i}", grad)
        stats = compressor.get_stats()
        assert stats["total_compressed"] == 5

    def test_reset(self):
        compressor = ZGradientCompressor(method=GradientCompression.TOP_K, min_size_to_compress=0)
        grad = torch.randn(100)
        compressor.compress("test", grad)
        assert len(compressor._error_feedback) > 0
        compressor.reset()
        assert len(compressor._error_feedback) == 0
        assert compressor._stats["skipped_small"] == 0

    def test_zero_gradient(self):
        compressor = ZGradientCompressor(
            method=GradientCompression.QUANTIZE_INT8,
        )
        grad = torch.zeros(100)
        result = compressor.compress("test", grad)
        assert torch.allclose(result, grad)

    def test_topology_change_invalidates_error_buffer(self):
        """Si el shape del parametro cambia entre steps (crecimiento de
        rango, sleep/compactacion de ElasticRank, refactorizacion), el
        buffer de error feedback con shape viejo NO debe romper compress():
        se descarta y se cuenta en error_buffer_resets."""
        compressor = ZGradientCompressor(
            method=GradientCompression.TOP_K,
            top_k_ratio=0.1,
            min_size_to_compress=0,
            compress_error_buffer=False,   # buffer crudo: assert directo del shape
        )
        # Step 1: factor rango 48 -> guarda error feedback con ese shape.
        compressor.compress("U", torch.randn(1024, 48))
        assert compressor._error_feedback["U"].shape == (1024, 48)

        # Step 2: el rango crecio a 60 -> grad nuevo con shape distinto.
        result = compressor.compress("U", torch.randn(1024, 60))

        assert result.shape == (1024, 60)              # no crash, shape ok
        assert compressor._stats["error_buffer_resets"] == 1
        # Tras el reset, se vuelve a poblar con el shape nuevo.
        assert compressor._error_feedback["U"].shape == (1024, 60)
