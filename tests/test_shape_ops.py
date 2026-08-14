"""Tests de M1 shape_ops — criterios ejecutables de SPEC-elasticshape-v1.md."""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mztrain.shape_ops import (
    dense_to_factorized,
    pad_adam_entry,
    pad_state_tensor,
    scale_output_rows,
    widen_embedding,
    widen_factorized_linear,
    widen_layernorm,
    zero_block_outputs,
)
from mztrain import ZFactorizedLinear

torch.manual_seed(0)


def _layer(i=96, o=160, r=32, bias=False):
    torch.manual_seed(1)
    return ZFactorizedLinear(i, o, rank=r, bias=bias, init_method="svd")


# ---------- widen: exactitud (SPEC §1) ----------

def test_widen_prefix_exact_and_new_outputs_zero():
    lay = _layer()
    x = torch.randn(7, 96)
    y_old = lay(x)
    new = widen_factorized_linear(lay, 128, 224)
    x_pad = torch.cat([x, torch.randn(7, 32)], dim=1)  # basura en dims nuevas
    y_new = new(x_pad)
    assert torch.allclose(y_new[:, :160], y_old, atol=1e-6, rtol=1e-6)
    assert torch.equal(y_new[:, 160:], torch.zeros(7, 64))
    assert new.rank == lay.rank
    assert torch.equal(new.S.data, lay.S.data)


def test_widen_with_bias_and_maps():
    lay = _layer(i=10, o=12, r=6, bias=True)
    x = torch.randn(4, 10)
    y_old = lay(x)
    in_map = torch.arange(10) * 2       # dims viejas en posiciones pares
    out_map = torch.arange(12) + 5      # desplazadas
    new = widen_factorized_linear(lay, 20, 20, in_map=in_map, out_map=out_map)
    x_pad = torch.randn(4, 20)
    x_pad[:, in_map] = x
    y_new = new(x_pad)
    assert torch.allclose(y_new[:, out_map], y_old, atol=1e-6, rtol=1e-6)
    fresh = torch.ones(20, dtype=torch.bool)
    fresh[out_map] = False
    assert torch.equal(y_new[:, fresh], torch.zeros(4, int(fresh.sum())))


def test_widen_noise_bounded_and_old_dims_exact():
    lay = _layer()
    x = torch.randn(7, 96)
    y_old = lay(x)
    g = torch.Generator().manual_seed(7)
    new = widen_factorized_linear(lay, 128, 224, noise_scale=1e-3, generator=g)
    x_pad = torch.cat([x, torch.zeros(7, 32)], dim=1)
    y_new = new(x_pad)
    assert torch.allclose(y_new[:, :160], y_old, atol=1e-6, rtol=1e-6)
    fresh_norm = float(y_new[:, 160:].norm())
    assert fresh_norm > 0.0
    assert fresh_norm < 0.05 * float(y_old.norm())


# ---------- qkv con cabezas + correccion SDPA (SPEC §3-4) ----------

def _attn(qkv_layer, x, heads, head_dim):
    B, T, _ = x.shape
    qkv = qkv_layer(x).view(B, T, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
    return a.transpose(1, 2).reshape(B, T, heads * head_dim)


def test_widen_qkv_heads_with_sdpa_scale_correction():
    d, heads, hd = 8, 2, 4
    d2, hd2 = 12, 6
    torch.manual_seed(2)
    qkv = ZFactorizedLinear(d, 3 * d, rank=8, bias=False, init_method="svd")
    x = torch.randn(3, 5, d)
    a_old = _attn(qkv, x, heads, hd)

    # mapa: bloque q|k|v, insercion por cabeza (hd 4 -> 6)
    pos = []
    for blk in range(3):
        for h in range(heads):
            for j in range(hd):
                pos.append(blk * d2 + h * hd2 + j)
    out_map = torch.tensor(pos)
    new = widen_factorized_linear(qkv, d2, 3 * d2, out_map=out_map)
    # correccion exacta de escala SDPA: filas del bloque q x sqrt(hd2/hd)
    scale_output_rows(new, torch.arange(d2), math.sqrt(hd2 / hd))

    x_pad = torch.cat([x, torch.randn(3, 5, d2 - d)], dim=2)
    a_new = _attn(new, x_pad, heads, hd2)
    for h in range(heads):
        old_slice = a_old[..., h * hd:(h + 1) * hd]
        new_slice = a_new[..., h * hd2:h * hd2 + hd]
        assert torch.allclose(new_slice, old_slice, atol=1e-5, rtol=1e-5)
        extra = a_new[..., h * hd2 + hd:(h + 1) * hd2]
        assert torch.equal(extra, torch.zeros_like(extra))


# ---------- LN y embedding ----------

def test_widen_layernorm_shapes_and_copy():
    ln = nn.LayerNorm(16)
    with torch.no_grad():
        ln.weight.mul_(2.0); ln.bias.add_(0.5)
    new = widen_layernorm(ln, 24)
    assert torch.equal(new.weight.data[:16], ln.weight.data)
    assert torch.equal(new.bias.data[:16], ln.bias.data)
    assert torch.equal(new.weight.data[16:], torch.ones(8))
    assert torch.equal(new.bias.data[16:], torch.zeros(8))


def test_widen_embedding_zero_and_noise():
    emb = nn.Embedding(11, 6)
    ids = torch.randint(0, 11, (4, 3))
    e_old = emb(ids)
    new0 = widen_embedding(emb, 9)
    assert torch.equal(new0(ids)[..., :6], e_old)
    assert torch.equal(new0(ids)[..., 6:], torch.zeros(4, 3, 3))
    g = torch.Generator().manual_seed(3)
    new1 = widen_embedding(emb, 9, noise_scale=1e-3, generator=g)
    assert torch.equal(new1(ids)[..., :6], e_old)
    assert float(new1.weight.data[:, 6:].norm()) > 0.0


# ---------- denso -> factorizado exacto (SPEC §7) ----------

def test_dense_to_factorized_exact():
    torch.manual_seed(4)
    lin = nn.Linear(16, 24, bias=True)
    fac = dense_to_factorized(lin)
    assert fac.rank == 16
    x = torch.randn(9, 16)
    assert torch.allclose(fac(x), lin(x), atol=1e-5, rtol=1e-5)
    W_rec = fac.reconstruct_weight()
    rel = float((W_rec - lin.weight.data).norm() / lin.weight.data.norm())
    assert rel < 1e-5


# ---------- bloque identidad (SPEC §6) ----------

class _Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = ZFactorizedLinear(d, 3 * d, rank=d // 2, bias=False)
        self.proj = ZFactorizedLinear(d, d, rank=d // 2, bias=False)
        self.fc1 = ZFactorizedLinear(d, 4 * d, rank=d // 2, bias=False)
        self.fc2 = ZFactorizedLinear(4 * d, d, rank=d // 2, bias=False)
        self.heads = heads
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


def test_identity_block_exact():
    torch.manual_seed(5)
    blk = _Block(16, 2)
    zero_block_outputs(blk.proj, blk.fc2)
    x = torch.randn(3, 4, 16)
    assert torch.equal(blk(x), x)


def test_zero_block_outputs_rejects_dense():
    with pytest.raises(TypeError):
        zero_block_outputs(nn.Linear(4, 4))  # type: ignore[arg-type]


# ---------- estado Adam (SPEC §8) ----------

def test_pad_state_tensor_2d_and_1d():
    t = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = pad_state_tensor(t, (6, 5), row_map=[0, 2, 3, 5], col_map=[1, 2, 4])
    assert torch.equal(out[[0, 2, 3, 5]][:, [1, 2, 4]], t)
    assert float(out.sum()) == float(t.sum())  # el resto es cero
    v = torch.ones(4)
    out1 = pad_state_tensor(v, (7,), row_map=[1, 3, 5, 6])
    assert float(out1.sum()) == 4.0
    with pytest.raises(ValueError):
        pad_state_tensor(t, (6, 5, 2))


def test_pad_adam_entry_step_by_value_not_aliased():
    entry = {"step": torch.tensor(123.0),
             "exp_avg": torch.randn(4, 3), "exp_avg_sq": torch.rand(4, 3)}
    out = pad_adam_entry(entry, (6, 4), row_map=[0, 1, 2, 3], col_map=[0, 1, 2])
    assert out["step"] is not entry["step"]        # sin alias (G4-B item 1)
    assert float(out["step"]) == 123.0
    out["step"] += 1                                # AdamW incrementa in-place
    assert float(entry["step"]) == 123.0            # el viejo no se contamina
    assert torch.equal(out["exp_avg"][:4, :3], entry["exp_avg"])
    assert torch.equal(out["exp_avg_sq"][:4, :3], entry["exp_avg_sq"])


# ---------- validacion de mapas ----------

def test_map_validation_errors():
    lay = _layer(i=8, o=8, r=4)
    with pytest.raises(ValueError):
        widen_factorized_linear(lay, 6, 12)            # encoger
    with pytest.raises(ValueError):
        widen_factorized_linear(lay, 12, 12, in_map=[0, 0, 1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        widen_factorized_linear(lay, 12, 12, out_map=list(range(7)) + [12])
    with pytest.raises(ValueError):
        widen_factorized_linear(lay, 12, 12, in_map=[0, 1, 2])


# ---------- dinamica: gradiente vivo / muerto (SPEC §2) ----------

def test_new_dims_train_when_loss_reads_them():
    torch.manual_seed(6)
    lay = _layer(i=12, o=10, r=5)
    g = torch.Generator().manual_seed(8)
    new = widen_factorized_linear(lay, 16, 14, noise_scale=1e-3, generator=g)
    opt = torch.optim.AdamW(new.parameters(), lr=1e-2)
    x = torch.randn(32, 16); tgt = torch.randn(32, 14)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = (new(x) - tgt).square().mean()
        loss.backward(); opt.step(); losses.append(float(loss))
    assert losses[-1] < losses[0]
    assert all(l == l for l in losses)  # sin NaN
    assert float(new.U.grad[10:, :].abs().sum()) > 0.0  # filas nuevas con gradiente


# ---------- anclas de la doble revision G4 (bias, padding, CUDA, Adam) ----------

def test_scale_rows_with_bias_qkv_exact():
    """G4-A item 2: con bias=True (default real de mztrain), escalar solo U
    rompia la correccion SDPA. El fix escala U y bias."""
    d, heads, hd, d2, hd2 = 8, 2, 4, 12, 6
    torch.manual_seed(11)
    qkv = ZFactorizedLinear(d, 3 * d, rank=8, bias=True, init_method="svd")
    with torch.no_grad():
        qkv.bias.normal_(0, 0.5)
    x = torch.randn(3, 5, d)
    a_old = _attn(qkv, x, heads, hd)
    pos = [blk * d2 + h * hd2 + j
           for blk in range(3) for h in range(heads) for j in range(hd)]
    new = widen_factorized_linear(qkv, d2, 3 * d2, out_map=torch.tensor(pos))
    scale_output_rows(new, torch.arange(d2), math.sqrt(hd2 / hd))
    a_new = _attn(new, torch.cat([x, torch.randn(3, 5, d2 - d)], dim=2), heads, hd2)
    for h in range(heads):
        assert torch.allclose(a_new[..., h * hd2:h * hd2 + hd],
                              a_old[..., h * hd:(h + 1) * hd],
                              atol=1e-5, rtol=1e-5)


def test_identity_block_with_bias():
    """G4-A item 4: proj/fc2 con bias devolvian x + bias != x."""
    torch.manual_seed(12)
    blk = _Block(16, 2)
    blk.proj = ZFactorizedLinear(16, 16, rank=8, bias=True)
    blk.fc2 = ZFactorizedLinear(64, 16, rank=8, bias=True)
    with torch.no_grad():
        blk.proj.bias.normal_(0, 0.1); blk.fc2.bias.normal_(0, 0.1)
    zero_block_outputs(blk.proj, blk.fc2)
    x = torch.randn(3, 4, 16)
    assert torch.equal(blk(x), x)


def test_padding_idx_preserved_with_noise():
    """G4-B item 3: la fila de padding debe seguir siendo el vector cero."""
    emb = nn.Embedding(10, 6, padding_idx=0)
    g = torch.Generator().manual_seed(13)
    new = widen_embedding(emb, 9, noise_scale=1e-3, generator=g)
    assert new.padding_idx == 0
    assert torch.equal(new.weight.data[0], torch.zeros(9))
    assert float(new.weight.data[1:, 6:].norm()) > 0.0  # el ruido sigue vivo


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
def test_cuda_widen_with_cpu_generator():
    """G4-B item 2 (confirmado): randn(device=cuda, generator=cpu) revienta;
    el fix muestrea en el device del generator y mueve."""
    lay = _layer(i=16, o=12, r=6).cuda()
    x = torch.randn(4, 16, device="cuda")
    y_old = lay(x)
    g = torch.Generator().manual_seed(14)
    new = widen_factorized_linear(lay, 24, 20, noise_scale=1e-3, generator=g)
    y_new = new(torch.cat([x, torch.zeros(4, 8, device="cuda")], dim=1))
    assert torch.allclose(y_new[:, :12], y_old, atol=1e-5, rtol=1e-5)
    assert float(y_new[:, 12:].norm()) > 0.0
    assert new.U.device.type == "cuda"


def test_gates_copied_exact():
    lay = _layer(i=10, o=10, r=5)
    with torch.no_grad():
        lay.wake_gate.copy_(torch.tensor([1.0, 0.5, 1.0, 0.25, 1.0]))
        lay.sleep_mask.copy_(torch.tensor([False, True, False, True, False]))
    new = widen_factorized_linear(lay, 14, 14)
    assert torch.equal(new.wake_gate, lay.wake_gate)
    assert torch.equal(new.sleep_mask, lay.sleep_mask)
    assert new._use_fp8 == lay._use_fp8
    assert new._epsi_scaling == lay._epsi_scaling


def test_adamw_update_bounded_after_pad():
    """G4-A item 5: primer update de un param nuevo con step alto y m=v=0
    esta acotado por lr·(1-b1)/sqrt(1-b2) ~ 3.16·lr — no diverge."""
    lr = 1e-3
    p = nn.Parameter(torch.zeros(6, 4))
    opt = torch.optim.AdamW([p], lr=lr, weight_decay=0.0)
    old = {"step": torch.tensor(1000.0),
           "exp_avg": torch.zeros(4, 3), "exp_avg_sq": torch.zeros(4, 3)}
    opt.state[p] = pad_adam_entry(old, (6, 4),
                                  row_map=[0, 1, 2, 3], col_map=[0, 1, 2])
    p.grad = torch.ones(6, 4)
    before = p.data.clone()
    opt.step()
    delta = (p.data - before).abs().max()
    assert 0.0 < float(delta) <= 3.2 * lr


def test_qkv_map_heads3_tight_packing():
    """G4-B item 7: geometria con 3 cabezas y slack=1 por cabeza."""
    d, heads, hd, d2, hd2 = 9, 3, 3, 12, 4
    torch.manual_seed(15)
    qkv = ZFactorizedLinear(d, 3 * d, rank=9, bias=False, init_method="svd")
    x = torch.randn(2, 4, d)
    a_old = _attn(qkv, x, heads, hd)
    pos = [blk * d2 + h * hd2 + j
           for blk in range(3) for h in range(heads) for j in range(hd)]
    new = widen_factorized_linear(qkv, d2, 3 * d2, out_map=torch.tensor(pos))
    scale_output_rows(new, torch.arange(d2), math.sqrt(hd2 / hd))
    a_new = _attn(new, torch.cat([x, torch.randn(2, 4, d2 - d)], dim=2), heads, hd2)
    for h in range(heads):
        assert torch.allclose(a_new[..., h * hd2:h * hd2 + hd],
                              a_old[..., h * hd:(h + 1) * hd],
                              atol=1e-5, rtol=1e-5)


def test_new_dims_dead_without_downstream_readers():
    """Verdad de diseno que M2 debe manejar: si nadie lee las dims nuevas
    (V aguas abajo = 0 y la loss solo ve dims viejas), su gradiente es 0."""
    torch.manual_seed(9)
    a = _layer(i=8, o=8, r=4)
    b = _layer(i=8, o=8, r=4)
    ga = torch.Generator().manual_seed(10)
    a2 = widen_factorized_linear(a, 8, 12, noise_scale=1e-3, generator=ga)
    b2 = widen_factorized_linear(b, 12, 12)  # V cols nuevas = 0: no lee
    x = torch.randn(16, 8)
    out = b2(a2(x))
    loss = out[:, :8].square().mean()  # la loss solo ve dims viejas
    loss.backward()
    assert float(a2.U.grad[8:, :].abs().sum()) == 0.0
