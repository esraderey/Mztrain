"""ElasticShape v1 — M1: primitivas puras de crecimiento de forma en espacio
factorizado. Ver SPEC-elasticshape-v1.md (contratos y matematica de preservacion).

Convenciones (mztrain): ZFactorizedLinear con U (out, r), S (r), V (r, in);
W_efectiva = U @ diag(S) @ V; forward y = ((x @ V^T) * S) @ U^T.

Preservacion: ensanchar es exacto A NIVEL DE CAPA (V nuevas cols = 0 ignoran
entradas nuevas; U nuevas filas = 0 emiten 0), salvo reasociacion FP del GEMM.
LayerNorm NO preserva a nivel de red (documentado en SPEC §5): eso lo mide el
probe del controller (M2), no estas primitivas.
"""
import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .layers import ZFactorizedLinear

IndexMap = Union[Sequence[int], torch.Tensor]


def _validate_map(m: Optional[IndexMap], old_dim: int, new_dim: int,
                  name: str) -> torch.Tensor:
    """Mapa de indices: posicion de cada dim vieja dentro del layout nuevo."""
    if m is None:
        m = torch.arange(old_dim)
    m = torch.as_tensor(m, dtype=torch.long)
    m = m.cpu()
    if m.ndim != 1 or m.numel() != old_dim:
        raise ValueError(f"{name}: se esperaban {old_dim} indices, hay {m.numel()}")
    if m.numel() != torch.unique(m).numel():
        raise ValueError(f"{name}: indices duplicados")
    if m.numel() and (int(m.min()) < 0 or int(m.max()) >= new_dim):
        raise ValueError(f"{name}: indices fuera de [0, {new_dim})")
    return m


def _new_rows_mask(n_new: int, used: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(n_new, dtype=torch.bool)
    mask[used] = False
    return mask


def _noise(shape, generator: Optional[torch.Generator], dev, dt) -> torch.Tensor:
    """Ruido reproducible independiente del device destino: se muestrea en el
    device del generator (CPU por defecto) y se mueve — randn(device=cuda,
    generator=cpu) revienta en torch."""
    gdev = generator.device if generator is not None else torch.device("cpu")
    n = torch.randn(*shape, generator=generator, device=gdev)
    return n.to(device=dev, dtype=dt)


def widen_factorized_linear(
    layer: ZFactorizedLinear,
    new_in: int,
    new_out: int,
    in_map: Optional[IndexMap] = None,
    out_map: Optional[IndexMap] = None,
    noise_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> ZFactorizedLinear:
    """Ensancha (in, out) -> (new_in, new_out) preservando la funcion en las
    dims viejas: V cols nuevas = 0, U filas nuevas = 0 (o ruido relativo
    noise_scale para mantener vivo el gradiente; SPEC §2). Rango intacto."""
    if new_in < layer.in_features or new_out < layer.out_features:
        raise ValueError("v1 solo crece: new_in/new_out >= actuales")
    imap = _validate_map(in_map, layer.in_features, new_in, "in_map")
    omap = _validate_map(out_map, layer.out_features, new_out, "out_map")
    dev, dt = layer.U.device, layer.U.dtype
    r = layer.rank

    new = ZFactorizedLinear(new_in, new_out, rank=r,
                            bias=layer.bias is not None, init_method="random")
    new = new.to(device=dev, dtype=dt)
    with torch.no_grad():
        V = torch.zeros(r, new_in, device=dev, dtype=dt)
        V[:, imap] = layer.V.data
        U = torch.zeros(new_out, r, device=dev, dtype=dt)
        U[omap, :] = layer.U.data
        if noise_scale > 0.0:
            fresh = _new_rows_mask(new_out, omap)
            if bool(fresh.any()):
                ref = float(layer.U.data.std())
                U[fresh] = _noise((int(fresh.sum()), r), generator, dev, dt) \
                    * (noise_scale * ref)
        new.U.data.copy_(U)
        new.S.data.copy_(layer.S.data)
        new.V.data.copy_(V)
        new.wake_gate.copy_(layer.wake_gate.to(dev))
        new.sleep_mask.copy_(layer.sleep_mask.to(dev))
        if layer.bias is not None:
            b = torch.zeros(new_out, device=dev, dtype=dt)
            b[omap] = layer.bias.data
            new.bias.data.copy_(b)
    # flags de runtime que el constructor resetea y deben sobrevivir al widen
    new._use_fp8 = layer._use_fp8
    new._epsi_scaling = layer._epsi_scaling
    return new


def widen_embedding(
    emb: nn.Embedding,
    new_dim: int,
    dim_map: Optional[IndexMap] = None,
    noise_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> nn.Embedding:
    """Ensancha la dim de embedding; cols nuevas = 0 (exacto) o ruido relativo
    (alimenta gradiente a las dims nuevas del stream via head atado; SPEC §2)."""
    if new_dim < emb.embedding_dim:
        raise ValueError("v1 solo crece: new_dim >= embedding_dim")
    dmap = _validate_map(dim_map, emb.embedding_dim, new_dim, "dim_map")
    dev, dt = emb.weight.device, emb.weight.dtype
    new = nn.Embedding(emb.num_embeddings, new_dim,
                       padding_idx=emb.padding_idx).to(device=dev, dtype=dt)
    with torch.no_grad():
        W = torch.zeros(emb.num_embeddings, new_dim, device=dev, dtype=dt)
        W[:, dmap] = emb.weight.data
        if noise_scale > 0.0:
            fresh = _new_rows_mask(new_dim, dmap)
            if bool(fresh.any()):
                ref = float(emb.weight.data.std())
                W[:, fresh] = _noise((emb.num_embeddings, int(fresh.sum())),
                                     generator, dev, dt) * (noise_scale * ref)
        new.weight.data.copy_(W)
        # el constructor zeroea la fila de padding pero copy_ la sobreescribe;
        # el contrato padding=vector cero debe sobrevivir al ruido
        if emb.padding_idx is not None:
            new.weight.data[emb.padding_idx].zero_()
    return new


def widen_layernorm(
    ln: nn.LayerNorm,
    new_dim: int,
    dim_map: Optional[IndexMap] = None,
    variance_compensation: bool = False,
) -> nn.LayerNorm:
    """Extiende LN: peso nuevo = 1, bias nuevo = 0, dims viejas copiadas.

    Con zero-pad del stream, LN diluye la varianza: sus salidas en dims viejas
    se reescalan ~sqrt(new/old). `variance_compensation=True` multiplica los
    pesos viejos por sqrt(old/new) y cancela ese factor dominante. El residuo
    (termino de shift de media, delta = k(1-d/d')·mu/sigma) es de PRIMER orden
    en mu — medido por G4-A: deriva de red 3.9-5.3% en configs toy 1.5x,
    1.3-4.1% en configs mayores, ~10% en growth 4x. Default False
    (comportamiento M1); el controller (M2) usa True. El probe mide la deriva
    real en cada growth."""
    if len(ln.normalized_shape) != 1:
        raise ValueError("solo LayerNorm 1-D")
    old_dim = ln.normalized_shape[0]
    if new_dim < old_dim:
        raise ValueError("v1 solo crece: new_dim >= dim actual")
    dmap = _validate_map(dim_map, old_dim, new_dim, "dim_map")
    dev, dt = ln.weight.device, ln.weight.dtype
    new = nn.LayerNorm(new_dim, eps=ln.eps,
                       elementwise_affine=ln.elementwise_affine)
    new = new.to(device=dev, dtype=dt)
    if ln.elementwise_affine:
        with torch.no_grad():
            w = torch.ones(new_dim, device=dev, dtype=dt)
            b = torch.zeros(new_dim, device=dev, dtype=dt)
            w[dmap] = ln.weight.data
            b[dmap] = ln.bias.data
            if variance_compensation:
                w[dmap] *= math.sqrt(old_dim / new_dim)
            new.weight.data.copy_(w)
            new.bias.data.copy_(b)
    return new


def scale_output_rows(layer: ZFactorizedLinear, rows: IndexMap,
                      factor: float) -> None:
    """Multiplica in-place las salidas `rows` por `factor`: filas de U Y sus
    entradas de bias (q_i = U_i·h + b_i; escalar solo U rompe la identidad con
    bias != 0). Uso: correccion exacta de la escala de SDPA al crecer head_dim
    — escalar el bloque q por sqrt(head_dim_nuevo / head_dim_viejo) (SPEC §4)."""
    if not math.isfinite(factor):
        raise ValueError("factor no finito")
    idx = torch.as_tensor(rows, dtype=torch.long).cpu()
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= layer.out_features):
        raise ValueError("rows fuera de rango")
    with torch.no_grad():
        layer.U.data[idx, :] *= factor
        if layer.bias is not None:
            layer.bias.data[idx] *= factor


def dense_to_factorized(linear: nn.Linear) -> ZFactorizedLinear:
    """Conversion EXACTA denso -> factorizado via SVD completa (rango = dim
    minima; error solo numerico, SPEC §7). Sin EPSI: el mapa se preserva."""
    W = linear.weight.data  # (out, in)
    out_f, in_f = W.shape
    m = min(out_f, in_f)
    dev, dt = W.device, W.dtype
    U_svd, S_svd, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    new = ZFactorizedLinear(in_f, out_f, rank=m,
                            bias=linear.bias is not None, init_method="random")
    new = new.to(device=dev, dtype=dt)
    with torch.no_grad():
        new.U.data.copy_(U_svd.to(dt))
        new.S.data.copy_(S_svd.to(dt))
        new.V.data.copy_(Vh.to(dt))
        if linear.bias is not None:
            new.bias.data.copy_(linear.bias.data)
        rel = float((new.reconstruct_weight() - W).norm() / W.norm().clamp_min(1e-12))
        new._reconstruction_error = rel
    return new


def zero_block_outputs(*layers: ZFactorizedLinear) -> None:
    """Anula in-place las U de las capas de SALIDA de un bloque residual
    (p.ej. proj y fc2): el bloque entero se vuelve la identidad EXACTA
    (x + 0 + 0), independientemente de LN internas (SPEC §6)."""
    for lay in layers:
        if not isinstance(lay, ZFactorizedLinear):
            raise TypeError("zero_block_outputs espera ZFactorizedLinear")
        with torch.no_grad():
            lay.U.data.zero_()
            # sin esto, el bloque devuelve x + bias_proj + bias_fc2 != x
            if lay.bias is not None:
                lay.bias.data.zero_()


def pad_state_tensor(
    t: torch.Tensor,
    new_shape: Sequence[int],
    row_map: Optional[IndexMap] = None,
    col_map: Optional[IndexMap] = None,
) -> torch.Tensor:
    """Zero-pad de un tensor de estado (m o v de Adam) a new_shape, colocando
    lo viejo segun los mapas. m=v=0 en zonas nuevas = parametro recien nacido
    para Adam (SPEC §8)."""
    if t.ndim != len(new_shape):
        raise ValueError("ndim distinto entre estado y new_shape")
    for k, (old_d, new_d) in enumerate(zip(t.shape, new_shape, strict=True)):
        if new_d < old_d:
            raise ValueError(f"solo crece: new_shape[{k}]={new_d} < {old_d}")
    out = torch.zeros(*new_shape, device=t.device, dtype=t.dtype)
    if t.ndim == 1:
        rmap = _validate_map(row_map, t.shape[0], new_shape[0], "row_map")
        out[rmap] = t
    elif t.ndim == 2:
        rmap = _validate_map(row_map, t.shape[0], new_shape[0], "row_map")
        cmap = _validate_map(col_map, t.shape[1], new_shape[1], "col_map")
        out[rmap.unsqueeze(1), cmap.unsqueeze(0)] = t
    else:
        raise ValueError("solo estados 1-D o 2-D")
    return out


def pad_adam_entry(
    entry: dict,
    new_shape: Sequence[int],
    row_map: Optional[IndexMap] = None,
    col_map: Optional[IndexMap] = None,
) -> dict:
    """Pad de una entrada de estado AdamW {step, exp_avg, exp_avg_sq}.
    `step` se conserva POR VALOR (clonado: AdamW lo incrementa in-place y un
    alias contaminaria el estado viejo). Nota honesta (SPEC §8): con step alto
    y m=v=0 el primer update del parametro nuevo es ~lr·(1-b1)/sqrt(1-b2)
    (≈3.16·lr con defaults) en vez del 1.0·lr de un estado step=0 — acotado,
    transitorio, y absorbido por el warmup post-growth del controller."""
    step = entry["step"]
    out = {"step": step.clone() if isinstance(step, torch.Tensor) else step}
    for k in ("exp_avg", "exp_avg_sq"):
        out[k] = pad_state_tensor(entry[k], new_shape, row_map, col_map)
    return out
