"""D6 aislado: escala del gradiente de U_q/b_q tras widen + correccion SDPA, SIN LayerNorm en el camino.
Entrada h fija zero-padded (lo que produce un stream exacto), misma perdida escalar sobre la salida de atencion en dims viejas."""
import sys, math, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch, torch.nn.functional as F
import mztrain.elastic_shape as es
from mztrain.shape_ops import widen_factorized_linear, scale_output_rows
from mztrain.layers import ZFactorizedLinear
torch.manual_seed(0)
B, T, D, H, D2 = 2, 5, 8, 2, 16
hd, hd2 = D // H, D2 // H
c = math.sqrt(hd2 / hd)

def attn(qkv_layer, h, d, heads):
    qkv = qkv_layer(h).view(B, T, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
    return a.transpose(1, 2).reshape(B, T, d)

for bias in (False, True):
    lay = ZFactorizedLinear(D, 3 * D, rank=D, bias=bias, init_method="random").double()
    if bias:
        with torch.no_grad(): lay.bias.normal_()
    h = torch.randn(B, T, D, dtype=torch.float64)
    w = torch.randn(B, T, D, dtype=torch.float64)  # pesos de la perdida sobre dims viejas
    out0 = attn(lay, h, D, H); loss0 = (out0 * w).sum(); loss0.backward()
    gU0 = lay.U.grad[:D].clone(); gb0 = lay.bias.grad[:D].clone() if bias else None
    gV0, gS0 = lay.V.grad.clone(), lay.S.grad.clone()

    omap = es.qkv_out_map(D, D2, H); imap_attn = es.attn_in_map(D, D2, H)
    new = widen_factorized_linear(lay, D2, 3 * D2, in_map=torch.arange(D), out_map=omap, noise_scale=0.0)
    scale_output_rows(new, torch.arange(D2), c)
    h2 = torch.zeros(B, T, D2, dtype=torch.float64); h2[..., :D] = h
    out1 = attn(new, h2, D2, H)
    # exactitud funcional en dims viejas (interleaved por cabeza)
    exact = float((out1[..., imap_attn] - out0).abs().max())
    w2 = torch.zeros(B, T, D2, dtype=torch.float64); w2[..., imap_attn] = w
    loss1 = (out1 * w2).sum(); loss1.backward()
    qrows = omap[:D]
    gU1 = new.U.grad[qrows]; gb1 = new.bias.grad[qrows] if bias else None
    rU = float(gU1.norm() / gU0.norm())
    print(f"bias={bias}: forward exacto max|diff|={exact:.2e}; ||gU_q'||/||gU_q||={rU:.9f} (1/c={1/c:.9f}); elementwise ok={torch.allclose(gU1, gU0 / c, atol=1e-10)}")
    if bias:
        print(f"           grad bias_q ratio elementwise ok={torch.allclose(gb1, gb0 / c, atol=1e-10)}")
    print(f"           grad V invariante={torch.allclose(new.V.grad[:, :D], gV0, atol=1e-10)}  grad S invariante={torch.allclose(new.S.grad, gS0, atol=1e-10)}")
    # filas k y v viejas: deben tener gradiente identico (no re-escalado)
    krows, vrows = omap[D:2 * D], omap[2 * D:]
    print(f"           grad U_k invariante={torch.allclose(new.U.grad[krows], lay.U.grad[D:2*D], atol=1e-10)}  grad U_v invariante={torch.allclose(new.U.grad[vrows], lay.U.grad[2*D:], atol=1e-10)}")
