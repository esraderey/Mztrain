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
    ZSparseFactorizedLinear,
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


# ---------- anclas de los residuos del peritaje ElasticShape (2026-09-26) ----------

def test_residuo_R1_reconstruct_weight_es_la_w_efectiva_del_forward():
    """R1: el export a denso usa reconstruct_weight; con wake_gate parcial
    (mini-warmup de ElasticRank) debe devolver la W del forward, U diag(S*gate) V."""
    torch.manual_seed(0)
    layer = ZFactorizedLinear(8, 6, rank=3, bias=False).double()
    x = torch.randn(4, 8, dtype=torch.float64)
    # control: con gate=1 es exactamente (U*S)@V, como antes
    assert torch.equal(layer.reconstruct_weight(), (layer.U * layer.S.unsqueeze(0)) @ layer.V)
    with torch.no_grad():
        layer.wake_gate.copy_(torch.tensor([1.0, 0.5, 0.0]))
    W = layer.reconstruct_weight()
    torch.testing.assert_close(x @ W.T, layer(x).detach(), atol=1e-12, rtol=0)


def test_residuo_R1_sparse_reconstruct_weight_incluye_wake_gate():
    """R1 (variante sparse): la parte low-rank de reconstruct_weight usa el gate."""
    torch.manual_seed(0)
    layer = ZSparseFactorizedLinear(8, 6, rank=3, bias=False, sparse_density=0.2).double()
    with torch.no_grad():
        layer.sparse_values.normal_()
        layer.wake_gate.copy_(torch.tensor([1.0, 0.5, 0.0]))
    x = torch.randn(4, 8, dtype=torch.float64)
    W = layer.reconstruct_weight()
    # la via sparse del forward corre en fp32: tolerancia fp32
    torch.testing.assert_close(x @ W.T, layer(x).detach(), atol=1e-5, rtol=1e-5)


def test_residuo_R2_grow_rank_resvd_resetea_gates():
    """R2: grow_rank(preserve_weights=False) re-SVD cambia la base; los gates
    viejos no corresponden a las direcciones nuevas y deben resetearse."""
    torch.manual_seed(0)
    layer = ZFactorizedLinear(8, 6, rank=3, bias=False)
    with torch.no_grad():
        layer.wake_gate.copy_(torch.tensor([1.0, 0.5, 0.0]))
        layer.sleep_mask.copy_(torch.tensor([False, True, False]))
    x = torch.randn(4, 8)
    y_before = layer(x).detach()
    layer.grow_rank(5, preserve_weights=False)
    assert torch.equal(layer.wake_gate, torch.ones(5))
    assert torch.equal(layer.sleep_mask, torch.zeros(5, dtype=torch.bool))
    # la re-SVD parte de la W efectiva: la funcion se conserva
    torch.testing.assert_close(layer(x).detach(), y_before, atol=1e-5, rtol=1e-5)
    # control: preserve_weights=True conserva el prefijo (indices preservados)
    keep = ZFactorizedLinear(8, 6, rank=3, bias=False)
    with torch.no_grad():
        keep.wake_gate.copy_(torch.tensor([1.0, 0.5, 0.0]))
        keep.sleep_mask.copy_(torch.tensor([False, True, False]))
    keep.grow_rank(5, preserve_weights=True)
    assert torch.equal(keep.wake_gate, torch.tensor([1.0, 0.5, 0.0, 1.0, 1.0]))
    assert torch.equal(keep.sleep_mask, torch.tensor([False, True, False, False, False]))


def test_residuo_R5b_forward_mixto_bf16_con_s_fp32_sin_autocast():
    """R5b: U/V bf16 con S fp32 (regla del proyecto) debe funcionar sin
    autocast y coincidir con el forward fp32 a tolerancia bf16."""
    torch.manual_seed(0)
    ref = ZFactorizedLinear(8, 6, rank=4, bias=True)
    x = torch.randn(3, 8)
    y_ref = ref(x).detach()
    lay = ZFactorizedLinear(8, 6, rank=4, bias=True)
    lay.load_state_dict(ref.state_dict())
    lay = lay.to(torch.bfloat16)
    lay.S.data = ref.S.data.clone()
    y = lay(x.to(torch.bfloat16)).detach()
    assert y.dtype == torch.bfloat16
    assert float((y.float() - y_ref).norm() / y_ref.norm()) < 5e-2
    W = lay.reconstruct_weight()
    assert W.dtype == torch.bfloat16
    W_ref = ref.reconstruct_weight()
    assert float((W.float() - W_ref).norm() / W_ref.norm()) < 5e-2


def test_residuo_R1_refactorize_continuo_con_gates_parciales():
    """R1: refactorize hace la SVD de la W EFECTIVA (con wake_gate) y el engine
    resetea los gates a 1: la funcion antes/despues debe ser continua."""
    from mztrain.refactorize import refactorize_model

    torch.manual_seed(1)
    layer = ZFactorizedLinear(16, 12, rank=4, bias=True)
    with torch.no_grad():
        layer.wake_gate.copy_(torch.tensor([1.0, 0.5, 0.0, 0.25]))
    x = torch.randn(6, 16)
    with torch.no_grad():
        y_before = layer(x)
    refactorize_model(layer, torch.optim.AdamW(layer.parameters(), lr=1e-3), 4, (ZFactorizedLinear,))
    layer.wake_gate = torch.ones(4)  # lo que hace ElasticRankController.reset_after_refactorize
    layer.sleep_mask = torch.zeros(4, dtype=torch.bool)
    with torch.no_grad():
        y_after = layer(x)
    assert float((y_after - y_before).norm() / y_before.norm()) < 1e-5


def test_residuo_R5b_forward_bajo_autocast_no_cambia():
    """D1.4: el guard de dtype no debe alterar el forward bajo autocast bf16
    (ruta empirica: parametros fp32 + autocast) ni con U/V bf16 y S fp32."""
    torch.manual_seed(0)
    layer = ZFactorizedLinear(16, 12, rank=4, bias=True)
    x = torch.randn(3, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y_fp32_params = layer(x)
        # referencia manual del mismo calculo bajo autocast, sin pasar por el guard
        h = torch.nn.functional.linear(x, layer.V) * layer._gated_s().unsqueeze(0)
        y_manual = torch.nn.functional.linear(h, layer.U) + layer.bias.to(torch.bfloat16)
    assert torch.equal(y_fp32_params, y_manual)
    mixed = ZFactorizedLinear(16, 12, rank=4, bias=True)
    mixed.load_state_dict(layer.state_dict())
    mixed = mixed.to(torch.bfloat16)
    mixed.S.data = layer.S.data.clone()  # S fp32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y_mixed = mixed(x.to(torch.bfloat16))
    assert y_mixed.dtype == torch.bfloat16
    assert float((y_mixed.float() - y_fp32_params.float()).norm() / y_fp32_params.float().norm()) < 5e-2
