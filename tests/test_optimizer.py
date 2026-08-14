"""
Tests para mztrain.optimizer - ZCompressedAdam.
"""

import pytest
import torch
import torch.nn as nn

from mztrain.optimizer import ZCompressedAdam


class TestZCompressedAdam:
    """Tests para ZCompressedAdam."""

    def test_creation(self):
        model = nn.Linear(64, 32)
        optimizer = ZCompressedAdam(model.parameters(), lr=1e-3)
        assert optimizer.compress_states is True
        assert optimizer._step_count == 0

    def test_step(self):
        model = nn.Linear(64, 32)
        optimizer = ZCompressedAdam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        assert optimizer._step_count == 1

    def test_weight_decay(self):
        model = nn.Linear(64, 32)
        w_before = model.weight.data.clone()
        optimizer = ZCompressedAdam(
            model.parameters(), lr=1e-3, weight_decay=0.1
        )
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        # Pesos deberian cambiar
        assert not torch.allclose(w_before, model.weight.data)

    def test_compression(self):
        model = nn.Linear(64, 32)
        optimizer = ZCompressedAdam(
            model.parameters(), lr=1e-3,
            compress_states=True, compression_interval=1,
        )
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Despues de 1 step con interval=1, deberia estar comprimido
        for p in model.parameters():
            state = optimizer.state.get(p, {})
            if len(state) > 0:
                assert state.get("compressed", False)

    def test_no_compression(self):
        model = nn.Linear(64, 32)
        optimizer = ZCompressedAdam(
            model.parameters(), lr=1e-3,
            compress_states=False,
        )
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            state = optimizer.state.get(p, {})
            if len(state) > 0:
                assert not state.get("compressed", True)

    def test_multiple_steps(self):
        model = nn.Linear(64, 32)
        optimizer = ZCompressedAdam(model.parameters(), lr=1e-3)
        for _ in range(10):
            x = torch.randn(4, 64)
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
        assert optimizer._step_count == 10

    def test_memory_stats(self):
        model = nn.Linear(64, 32)
        optimizer = ZCompressedAdam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        stats = optimizer.get_memory_stats()
        assert "total_params" in stats
        assert "state_memory_mb" in stats
        assert "full_memory_mb" in stats
        assert "memory_saved_pct" in stats
        assert stats["total_params"] > 0

    def test_sparse_grad_raises(self):
        embedding = nn.Embedding(100, 64, sparse=True)
        optimizer = ZCompressedAdam(embedding.parameters(), lr=1e-3)
        x = torch.tensor([1, 2, 3])
        loss = embedding(x).sum()
        loss.backward()
        with pytest.raises(RuntimeError, match="sparse"):
            optimizer.step()

    def test_compress_decompress(self):
        optimizer = ZCompressedAdam([torch.randn(10, requires_grad=True)])
        tensor = torch.randn(100)
        compressed = optimizer._compress_state(tensor)
        # Block-wise: tupla (quantized, scales, shape, orig_numel)
        assert len(compressed) == 4
        quantized, scales, shape, orig_numel = compressed
        assert quantized.dtype == torch.int8
        assert shape == tensor.shape
        assert orig_numel == tensor.numel()
        reconstructed = optimizer._decompress_state(compressed, torch.float32)
        assert reconstructed.shape == tensor.shape
        # Error de cuantizacion deberia ser pequeno (block-wise mejora vs scalar).
        assert torch.allclose(tensor, reconstructed, atol=0.05)

    def test_zero_tensor_compress(self):
        optimizer = ZCompressedAdam([torch.randn(10, requires_grad=True)])
        tensor = torch.zeros(100)
        compressed = optimizer._compress_state(tensor)
        quantized, scales, shape, orig_numel = compressed
        # Tensor cero: quantized debe ser todo cero, scales todo cero.
        assert (quantized == 0).all()
        assert (scales == 0).all()
        # Round-trip de un tensor cero da cero.
        reconstructed = optimizer._decompress_state(compressed, torch.float32)
        assert torch.allclose(reconstructed, tensor)

    def test_compress_v_log_preserves_dynamic_range(self):
        """Adam v tiene distribucion sesgada — la cuantizacion logaritmica
        debe preservar el rango dinamico para que sqrt(v) no colapse.
        """
        optimizer = ZCompressedAdam([torch.randn(10, requires_grad=True)])
        # v sesgada como en Adam real: outlier + valores tiny
        v = torch.zeros(2048)
        v[0] = 1e-3
        v[1:100] = torch.rand(99) * 1e-7
        v[100:] = torch.rand(1948) * 1e-9

        compressed = optimizer._compress_v(v)
        v_rec = optimizer._decompress_v(compressed, torch.float32)

        # Critico: ningun valor debe colapsar a cero (eso causaria denom=eps
        # y step explota).
        assert (v_rec > 0).all(), "Bug #8: v values collapsed to zero in log-quant"

        # Precision relativa por elemento debe ser razonable (<25% en general).
        rel_err = ((v - v_rec).abs() / v.clamp(min=1e-12))
        assert rel_err.median().item() < 0.10, \
            f"Median relative error too high: {rel_err.median().item():.3f}"

        # 1/sqrt(v) - el factor que multiplica el step de Adam - debe seguir
        # acotado (no saltar a 1/eps = 1e8).
        denom_orig = (v.sqrt() + 1e-8)
        denom_rec = (v_rec.sqrt() + 1e-8)
        # max(1/denom) no debe crecer mas de 10% vs original.
        ratio = (1.0 / denom_rec).max() / (1.0 / denom_orig).max()
        assert 0.5 < ratio.item() < 1.5, \
            f"1/sqrt(v) max ratio {ratio.item():.3f} (expected ~1)"

    def test_compress_v_zero_tensor(self):
        """v=0 inicial debe round-trippear a v=0 (o muy cerca)."""
        optimizer = ZCompressedAdam([torch.randn(10, requires_grad=True)])
        v = torch.zeros(64)
        compressed = optimizer._compress_v(v)
        v_rec = optimizer._decompress_v(compressed, torch.float32)
        # log(eps_floor).exp() == eps_floor, valores muy chicos pero positivos.
        # Lo importante: no exploten ni produzcan NaN.
        assert v_rec.isfinite().all()
        assert (v_rec >= 0).all()

    def test_closure(self):
        model = nn.Linear(64, 32)
        optimizer = ZCompressedAdam(model.parameters(), lr=1e-3)

        def closure():
            optimizer.zero_grad()
            x = torch.randn(4, 64)
            loss = model(x).sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure=closure)
        assert loss is not None
