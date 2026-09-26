"""ElasticShape v1 — M1: primitivas puras de crecimiento de forma en espacio
factorizado. Ver SPEC-elasticshape-v1.md (contratos y matematica de preservacion).

Convenciones (mztrain): ZFactorizedLinear con U (out, r), S (r), V (r, in);
W_efectiva = U @ diag(S) @ V; forward y = ((x @ V^T) * S) @ U^T.

Preservacion: ensanchar es exacto A NIVEL DE CAPA (V nuevas cols = 0 ignoran
entradas nuevas; U nuevas filas = 0 emiten 0), salvo reasociacion FP del GEMM.
LayerNorm NO preserva a nivel de red (documentado en SPEC §5): eso lo mide el
probe del controller (M2), no estas primitivas.
"""

import copy
import math
import numbers
import warnings
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .layers import ZFactorizedLinear

IndexMap = Union[Sequence[int], torch.Tensor]

_INT_TENSOR_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


class NoiseError(ValueError):
    """Ruido funcional calibrado no representable en el dtype de destino
    (colapso a cero por underflow, o valores no finitos)."""


def _require_int(value, name: str) -> None:
    """Exige un int de Python (no bool, no float, no tensor 0-d)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} debe ser int (no bool, no float)")


def _validate_map(m: Optional[IndexMap], old_dim: int, new_dim: int, name: str) -> torch.Tensor:
    """Mapa de indices: posicion de cada dim vieja dentro del layout nuevo."""
    if m is None:
        m = torch.arange(old_dim)
    elif torch.is_tensor(m):
        if m.dtype not in _INT_TENSOR_DTYPES:
            raise ValueError(f"{name}: indices deben ser enteros")
    else:
        for v in m:
            if isinstance(v, bool) or not isinstance(v, numbers.Integral):
                raise ValueError(f"{name}: indices deben ser enteros")
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


def _validate_noise(noise_scale: float) -> None:
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale debe ser finito y >= 0")


@torch.no_grad()
def _functional_noise(
    layer: ZFactorizedLinear, count: int, noise_scale: float, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """Calibra ||delta_W||_F = noise_scale * ||W_efectiva||_F sin formar W.

    QR de (diag(S*gate)V)^T permite calcular ambas normas con matrices de
    ancho rank. La magnitud funcional no depende de std(U) ni de su gauge;
    la direccion aleatoria puede depender de la parametrizacion. La cota
    es de Frobenius por capa, no de error relativo para cada entrada/red.
    Se calcula en alta precision; el resultado tiene tolerancia del dtype.
    Un mapa efectivo nulo recibe ruido nulo.
    """
    dev, dt = layer.U.device, layer.U.dtype
    acc = torch.float32 if dev.type == "mps" else torch.float64
    u = layer.U.detach().to(acc)
    sv = layer.V.detach().to(acc) * layer._gated_s().detach().to(acc)[:, None]
    if not torch.isfinite(u).all() or not torch.isfinite(sv).all():
        raise ValueError("factores no finitos: no se puede calibrar el ruido")
    _, triangular = torch.linalg.qr(sv.T, mode="r")
    target = (u @ triangular.T).norm() * noise_scale
    noise = _noise((count, layer.rank), generator, dev, acc)
    size = (noise @ triangular.T).norm()
    if target == 0 or size == 0:
        return torch.zeros(count, layer.rank, device=dev, dtype=dt)
    noise = (noise * (target / size)).to(dt)
    # criterio elemento a elemento: la norma en dtype bajo subdesborda con
    # ruido minusculo pero no nulo (falso positivo detectado en revision)
    if target > 0 and bool((noise == 0).all()):
        raise NoiseError("ruido funcional colapsa a cero en el dtype de U")
    if not torch.isfinite(noise).all():
        raise NoiseError("ruido funcional no representable en el dtype de U")
    return noise


def _inherit_parameter_flags(old: nn.Module, new: nn.Module) -> None:
    new.train(old.training)
    for name, p in old.named_parameters(recurse=False):
        q = getattr(new, name, None)
        if q is not None:
            q.requires_grad_(p.requires_grad)


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
    dims viejas: V cols nuevas = 0, U filas nuevas = 0 o ruido calibrado
    por ||delta_W||_F / ||W_efectiva||_F = noise_scale. Rango intacto.
    No garantiza una deriva relativa pequena en cada entrada ni en la red."""
    _validate_noise(noise_scale)
    _require_int(new_in, "new_in")
    _require_int(new_out, "new_out")
    if new_in < layer.in_features or new_out < layer.out_features:
        raise ValueError("v1 solo crece: new_in/new_out >= actuales")
    imap = _validate_map(in_map, layer.in_features, new_in, "in_map")
    omap = _validate_map(out_map, layer.out_features, new_out, "out_map")
    dev, dt = layer.U.device, layer.U.dtype
    r = layer.rank

    new = ZFactorizedLinear(new_in, new_out, rank=r, bias=layer.bias is not None, init_method="random")
    new = new.to(device=dev, dtype=dt)
    with torch.no_grad():
        V = torch.zeros(r, new_in, device=dev, dtype=dt)
        V[:, imap] = layer.V.data
        U = torch.zeros(new_out, r, device=dev, dtype=dt)
        U[omap, :] = layer.U.data
        if noise_scale > 0.0:
            fresh = _new_rows_mask(new_out, omap)
            if bool(fresh.any()):
                U[fresh] = _functional_noise(layer, int(fresh.sum()), noise_scale, generator)
        new.U.data.copy_(U)
        # S/gates pueden ser fp32 con U/V bf16: no reducir su precision al
        # copiar el resto de la capa al dtype de U.
        new.S.data = layer.S.detach().clone()
        new.V.data.copy_(V)
        new.wake_gate = layer.wake_gate.detach().clone()
        new.sleep_mask.copy_(layer.sleep_mask.to(dev))
        if layer.bias is not None:
            b = torch.zeros(new_out, device=dev, dtype=dt)
            b[omap] = layer.bias.data
            new.bias.data.copy_(b)
    # flags de runtime que el constructor resetea y deben sobrevivir al widen
    new._use_fp8 = layer._use_fp8
    new._epsi_scaling = layer._epsi_scaling
    _inherit_parameter_flags(layer, new)
    return new


def widen_embedding(
    emb: nn.Embedding,
    new_dim: int,
    dim_map: Optional[IndexMap] = None,
    noise_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> nn.Embedding:
    """Ensancha el embedding con ||columnas nuevas||_F = noise_scale*||W||_F.

    El padding se excluye del ruido. La cota es global sobre la tabla, no
    sobre cada token; la sonda del controlador mide el efecto en la red.
    Con ||W||_F = 0 las columnas nuevas quedan en cero (calibracion relativa).
    """
    _validate_noise(noise_scale)
    _require_int(new_dim, "new_dim")
    if new_dim < emb.embedding_dim:
        raise ValueError("v1 solo crece: new_dim >= embedding_dim")
    dmap = _validate_map(dim_map, emb.embedding_dim, new_dim, "dim_map")
    dev, dt = emb.weight.device, emb.weight.dtype
    new = nn.Embedding(
        emb.num_embeddings,
        new_dim,
        padding_idx=emb.padding_idx,
        max_norm=emb.max_norm,
        norm_type=emb.norm_type,
        scale_grad_by_freq=emb.scale_grad_by_freq,
        sparse=emb.sparse,
    ).to(device=dev, dtype=dt)
    with torch.no_grad():
        W = torch.zeros(emb.num_embeddings, new_dim, device=dev, dtype=dt)
        W[:, dmap] = emb.weight.data
        if noise_scale > 0.0:
            fresh = _new_rows_mask(new_dim, dmap)
            if bool(fresh.any()):
                acc = torch.float64 if dev.type != "mps" else torch.float32
                noise = _noise((emb.num_embeddings, int(fresh.sum())), generator, dev, acc)
                if emb.padding_idx is not None:
                    noise[emb.padding_idx].zero_()
                norm = noise.norm()
                ref = emb.weight.detach().to(acc).norm()
                if not torch.isfinite(ref):
                    raise ValueError("embedding no finito")
                if norm > 0:
                    W[:, fresh] = (noise * (noise_scale * ref / norm)).to(dt)
        new.weight.data.copy_(W)
        # el constructor zeroea la fila de padding pero copy_ la sobreescribe;
        # el contrato padding=vector cero debe sobrevivir al ruido
        if emb.padding_idx is not None:
            new.weight.data[emb.padding_idx].zero_()
    _inherit_parameter_flags(emb, new)
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
    real en cada growth. Con variance_compensation=True el eps tambien se
    reescala (eps' = eps*old_dim/new_dim), cancelando exactamente ese termino."""
    if len(ln.normalized_shape) != 1:
        raise ValueError("solo LayerNorm 1-D")
    old_dim = ln.normalized_shape[0]
    _require_int(new_dim, "new_dim")
    if new_dim < old_dim:
        raise ValueError("v1 solo crece: new_dim >= dim actual")
    dmap = _validate_map(dim_map, old_dim, new_dim, "dim_map")
    if not ln.elementwise_affine or ln.bias is None:
        raise ValueError("widen_layernorm requiere weight y bias afines")
    dev, dt = ln.weight.device, ln.weight.dtype
    new_eps = ln.eps * old_dim / new_dim if variance_compensation else ln.eps
    new = nn.LayerNorm(new_dim, eps=new_eps, elementwise_affine=ln.elementwise_affine)
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
    _inherit_parameter_flags(ln, new)
    return new


def scale_output_rows(layer: ZFactorizedLinear, rows: IndexMap, factor: float) -> None:
    """Multiplica in-place las salidas `rows` por `factor`: filas de U Y sus
    entradas de bias (q_i = U_i·h + b_i; escalar solo U rompe la identidad con
    bias != 0). Uso: correccion exacta de la escala de SDPA al crecer head_dim
    — escalar el bloque q por sqrt(head_dim_nuevo / head_dim_viejo) (SPEC §4)."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("factor debe ser finito y > 0")
    idx = torch.as_tensor(rows, dtype=torch.long).cpu()
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= layer.out_features):
        raise ValueError("rows fuera de rango")
    with torch.no_grad():
        layer.U.data[idx, :] *= factor
        if layer.bias is not None:
            layer.bias.data[idx] *= factor


def dense_to_factorized(linear: nn.Linear) -> ZFactorizedLinear:
    """Conversion EXACTA denso -> factorizado via SVD completa (rango = dim
    minima; error solo numerico, SPEC §7). Sin EPSI: el mapa se preserva.

    En baja precision (fp16/bf16) U/V quedan en ese dtype y S se conserva en
    fp32 (regla del proyecto: valores singulares nunca en baja precision; el
    forward tolera el dtype mixto con y sin autocast). La reconstruccion tiene
    error ~1e-3-1e-2 por U/V; la ruta recomendada para conversion de alta
    fidelidad es hacerla en parametros fp32. En fp32/fp64 S queda en ese dtype.
    """
    W = linear.weight.data  # (out, in)
    out_f, in_f = W.shape
    m = min(out_f, in_f)
    dev, dt = W.device, W.dtype
    if dt in (torch.float16, torch.bfloat16):
        warnings.warn(
            f"dense_to_factorized en baja precision: U/V quedan en {dt} (S en fp32): la "
            "reconstruccion tiene error ~1e-3-1e-2; la ruta recomendada es parametros fp32",
            RuntimeWarning,
            stacklevel=2,
        )
    work = W if dt == torch.float64 else W.float()
    U_svd, S_svd, Vh = torch.linalg.svd(work, full_matrices=False)
    new = ZFactorizedLinear(in_f, out_f, rank=m, bias=linear.bias is not None, init_method="random")
    new = new.to(device=dev, dtype=dt)
    with torch.no_grad():
        new.U.data.copy_(U_svd.to(dt))
        if dt in (torch.float16, torch.bfloat16):
            new.S.data = S_svd.to(torch.float32)  # como widen_factorized_linear: S fp32
        else:
            new.S.data.copy_(S_svd.to(dt))
        new.V.data.copy_(Vh.to(dt))
        if linear.bias is not None:
            new.bias.data.copy_(linear.bias.data)
        rel = float((new.reconstruct_weight() - W).norm() / W.norm().clamp_min(1e-12))
        new._reconstruction_error = rel
    for p in (new.U, new.S, new.V):
        p.requires_grad_(linear.weight.requires_grad)
    if linear.bias is not None:
        new.bias.requires_grad_(linear.bias.requires_grad)
    new.train(linear.training)
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
    lo viejo segun los mapas. m=v=0 en zonas nuevas; si heredan un contador
    avanzado, no equivalen a un parametro recien nacido para Adam (SPEC §8)."""
    if torch.is_tensor(new_shape) or any(
        isinstance(d, bool) or not isinstance(d, numbers.Integral) for d in new_shape
    ):
        raise ValueError("new_shape debe ser una secuencia de int (no tensor)")
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


_ADAM_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def pad_adam_entry(
    entry: dict,
    new_shape: Sequence[int],
    row_map: Optional[IndexMap] = None,
    col_map: Optional[IndexMap] = None,
) -> dict:
    """Pad del estado AdamW/AMSGrad y copia independiente de sus metadatos.
    `step` se conserva POR VALOR (clonado: AdamW lo incrementa in-place y un
    alias contaminaria el estado viejo). Nota honesta (SPEC §8): con step alto
    y m=v=0 el primer update del parametro nuevo es ~lr·(1-b1)/sqrt(1-b2)
    (≈3.16·lr con defaults) en vez del 1.0·lr de un estado step=0. Es el
    limite del PRIMER paso: pasos siguientes pueden superar ese valor;
    warmup mitiga el transitorio, sin garantizar estabilidad."""
    old_shape = entry["exp_avg"].shape
    if entry["exp_avg_sq"].shape != old_shape or ("max_exp_avg_sq" in entry and entry["max_exp_avg_sq"].shape != old_shape):
        raise ValueError("shapes incompatibles en los momentos AdamW")
    if "step" not in entry:
        raise ValueError("entrada AdamW sin 'step'")
    out = {}
    for key, value in entry.items():
        if key in _ADAM_MOMENT_KEYS and isinstance(value, torch.Tensor):
            out[key] = pad_state_tensor(value, new_shape, row_map, col_map)
        else:
            out[key] = copy.deepcopy(value)
    return out
