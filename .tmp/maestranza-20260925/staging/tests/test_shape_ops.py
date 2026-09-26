"""Tests de M1 shape_ops — criterios ejecutables de SPEC-elasticshape-v1.md."""
import math
import warnings

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


# ---------- anclas del peritaje PER-SHAPE (ver ORDEN-PER-SHAPE.md) ----------

def test_ancla_PER_API_001():
    """PER-API-001: _validate_map debe rechazar mapas no enteros ANTES del
    cast (antes: floats truncados en silencio via as_tensor(dtype=long))."""
    ln = nn.LayerNorm(4)
    with torch.no_grad():
        ln.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        ln.bias.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    with pytest.raises(ValueError):
        widen_layernorm(ln, new_dim=6, dim_map=torch.tensor([0.1, 1.1, 2.1, 3.1]))
    with pytest.raises(ValueError):
        widen_layernorm(ln, new_dim=6, dim_map=[0, 1, True, 3])  # bool no es indice
    # control negativo: mapa entero real sigue funcionando
    new_ln = widen_layernorm(ln, new_dim=6, dim_map=torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(new_ln.weight.data[:4], ln.weight.data)


def test_ancla_PER_API_002():
    """PER-API-002: scale_output_rows debe exigir factor > 0 (su unico uso
    documentado, la correccion SDPA, siempre es sqrt(...) > 0)."""
    torch.manual_seed(0)
    layer = ZFactorizedLinear(8, 8, rank=4, bias=True, init_method="random")
    with torch.no_grad():
        layer.U.data.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4) + 1.0)
        layer.bias.data.copy_(torch.arange(8, dtype=torch.float32) + 1.0)
    with pytest.raises(ValueError):
        scale_output_rows(layer, rows=[0, 1, 2], factor=-2.5)
    with pytest.raises(ValueError):
        scale_output_rows(layer, rows=[0, 1, 2], factor=0.0)
    # control negativo: factor positivo sigue funcionando
    before = layer.U.data.clone()
    scale_output_rows(layer, rows=[0, 1, 2], factor=2.0)
    assert torch.allclose(layer.U.data[:3], before[:3] * 2.0)


def test_ancla_PER_API_004():
    """PER-API-004: pad_state_tensor debe rechazar new_shape que no sea una
    secuencia de int de Python (p.ej. un tensor) con ValueError explicito."""
    t = torch.randn(4)
    with pytest.raises(ValueError):
        pad_state_tensor(t, torch.tensor([6]))
    # control negativo: tupla de enteros funciona
    out = pad_state_tensor(t, (6,))
    assert out.shape == (6,)
    assert torch.equal(out[:4], t)


def test_ancla_PER_API_005():
    """PER-API-005: pad_adam_entry sin 'step' debe dar ValueError explicativo,
    no un KeyError crudo."""
    entry = {"exp_avg": torch.zeros(4), "exp_avg_sq": torch.zeros(4)}
    with pytest.raises(ValueError, match="step"):
        pad_adam_entry(entry, (6,))
    # control negativo: con 'step' presente funciona normalmente
    entry_ok = {"exp_avg": torch.zeros(4), "exp_avg_sq": torch.zeros(4), "step": torch.tensor(5.0)}
    out = pad_adam_entry(entry_ok, (6,))
    assert float(out["step"]) == 5.0


def test_ancla_PER_API_009():
    """PER-API-009 (y el resto del hallazgo 3): new_dim/new_in/new_out deben
    ser int (no bool, no float) en widen_layernorm/widen_embedding/
    widen_factorized_linear, con ValueError explicito en vez de un TypeError
    interno de PyTorch."""
    ln = nn.LayerNorm(4)
    with pytest.raises(ValueError):
        widen_layernorm(ln, new_dim=6.5)
    with pytest.raises(ValueError):
        widen_layernorm(ln, new_dim=True)  # bool no es entero valido
    emb = nn.Embedding(5, 4)
    with pytest.raises(ValueError):
        widen_embedding(emb, new_dim=6.0)
    lay = _layer(i=8, o=8, r=4)
    with pytest.raises(ValueError):
        widen_factorized_linear(lay, new_in=12.0, new_out=8)
    with pytest.raises(ValueError):
        widen_factorized_linear(lay, new_in=12, new_out=True)
    # control negativo: enteros reales funcionan
    new_ln = widen_layernorm(ln, new_dim=6)
    assert new_ln.normalized_shape == (6,)


def test_ancla_PER_API_011():
    """PER-API-011: pad_adam_entry solo debe paddear las claves explicitas de
    AdamW/AMSGrad; una clave ajena que coincide en forma por casualidad debe
    quedar intacta (deep-copiada), no expandida."""
    old_shape = (4,)
    entry = {
        "exp_avg": torch.zeros(old_shape),
        "exp_avg_sq": torch.zeros(old_shape),
        "step": torch.tensor(3.0),
        "custom_adapter_counter": torch.arange(4, dtype=torch.float32),
    }
    out = pad_adam_entry(entry, (6,))
    assert out["custom_adapter_counter"].shape == entry["custom_adapter_counter"].shape
    assert torch.equal(out["custom_adapter_counter"], entry["custom_adapter_counter"])
    # control negativo: las claves documentadas si se expanden
    assert out["exp_avg"].shape == (6,)


def test_ancla_PER_MAT_002():
    """PER-MAT-002: en fp16 con gauge extremo, la calibracion funcional puede
    colapsar a cero por underflow sin ningun aviso; debe lanzar NoiseError."""
    from mztrain.shape_ops import NoiseError

    torch.manual_seed(1)
    layer = ZFactorizedLinear(16, 16, rank=4, bias=False, init_method="random")
    with torch.no_grad():
        layer.U.normal_()
        layer.V.normal_()
        layer.S.fill_(1.0)
        gauge = 1e-3
        layer.U.mul_(gauge)  # mismo W: U*g, S/g
        layer.S.div_(gauge)
    layer = layer.to(torch.float16)
    layer.S.data = layer.S.data.float()  # regla del proyecto: S fp32
    g = torch.Generator().manual_seed(3)
    with pytest.raises(NoiseError):
        widen_factorized_linear(layer, 16, 32, noise_scale=1e-5, generator=g)


def test_ancla_PER_MAT_003():
    """PER-MAT-003: dense_to_factorized en baja precision (fp16/bf16) debe
    avisar del error de reconstruccion ~1e-3-1e-2; en fp32 no debe avisar.
    Residuo R5: en bf16 S queda en fp32 y el forward (sin autocast) funciona."""
    torch.manual_seed(0)
    lin_bf = nn.Linear(48, 144, bias=False).to(torch.bfloat16)
    with pytest.warns(RuntimeWarning):
        fac_bf = dense_to_factorized(lin_bf)
    assert fac_bf.S.dtype == torch.float32
    assert fac_bf.U.dtype == fac_bf.V.dtype == torch.bfloat16
    x = torch.randn(9, 48).to(torch.bfloat16)
    with torch.no_grad():
        y_ref, y = lin_bf(x).float(), fac_bf(x).float()
    assert float((y - y_ref).norm() / y_ref.norm()) < 1e-2
    lin_f32 = nn.Linear(48, 144, bias=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dense_to_factorized(lin_f32)
    assert not any(issubclass(w.category, RuntimeWarning) for w in rec)


def test_ancla_PER_MAT_004():
    """PER-MAT-004: widen_layernorm(variance_compensation=True) debe tambien
    reescalar eps (eps' = eps*old_dim/new_dim); si no, con sigma^2=eps queda
    un residuo no declarado de ~18%."""
    d, dn, eps = 8, 16, 1e-5
    torch.manual_seed(0)
    ln = nn.LayerNorm(d, eps=eps).double()
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.normal_()

    def centered(var):
        x = torch.randn(32, d, dtype=torch.float64)
        x = x - x.mean(-1, keepdim=True)  # mu = 0 exacto
        return x / x.std(-1, unbiased=False, keepdim=True) * math.sqrt(var)

    def err(new, x):
        xp = torch.cat([x, torch.zeros(x.shape[0], dn - d, dtype=x.dtype)], -1)
        with torch.no_grad():
            y_old, y_new = ln(x), new(xp)[:, :d]
            return ((y_new - y_old).norm() / (y_old - ln.bias).norm()).item()

    new = widen_layernorm(ln, dn, variance_compensation=True)
    e_bad = err(new, centered(eps))
    assert e_bad < 1e-6, f"residuo de eps no compensado: {e_bad:.3f}"
