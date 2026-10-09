"""ElasticShape v1 — crecimiento de forma en espacio factorizado.

Modelo GPT de referencia (convencion del arco empirico: heads fijos,
head_dim = d/heads, head atado, bias=False) + controller del morph:
conversion densa->factorizada exacta, ensanchado con ledger de migracion
Adam, profundizado identidad, probe de deriva, schedule y warmup.

Claim historico v1 (T8; docs/evidencia/T8-VEREDICTO.md): ~54% del reloj
a escala 11M-equiv. No valida esta revision de seguridad y ruido funcional:
ver docs/ELASTICSHAPE-SAFETY.md. API opt-in: nada de este modulo se activa
salvo llamada explicita.

Decisiones documentadas: LN con variance_compensation=True; estado q
re-escalado (1/c, 1/c^2) tras la correccion SDPA; conversion densa->
factorizada resetea el optimizer (parametrizaciones no conmensurables).
"""

import copy
import inspect
import math
import numbers
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ZFactorizedLinear
from .shape_ops import (
    NoiseError,
    dense_to_factorized,
    pad_adam_entry,
    scale_output_rows,
    widen_embedding,
    widen_factorized_linear,
    widen_layernorm,
    zero_block_outputs,
)


@contextmanager
def _shape_transaction(model, generator=None):
    """Rollback de la topologia por referencias, sin duplicar los pesos.

    Las cirugias solo reemplazan modulos/parametros, nunca escriben pesos
    originales. Se conservan identidades, gradientes, modos y RNG en fallos.
    El optimizer original se mantiene intacto mediante estados clonados.
    """
    snapshots = []
    for module in model.modules():
        state = module.__dict__.copy()
        state["_modules"] = module._modules.copy()
        snapshots.append((module, state))
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    gen_rng = generator.get_state() if generator is not None else None
    try:
        yield
    except BaseException:
        for module, state in snapshots:
            module.__dict__.clear()
            module.__dict__.update(state)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if gen_rng is not None:
            generator.set_state(gen_rng)
        raise


def _atomic_shape(function):
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        # Solo widen_gpt recibe un generator, al final de su firma publica.
        generator = kwargs.get("generator", args[2] if len(args) > 2 else None)
        with _shape_transaction(model, generator):
            return function(model, *args, **kwargs)

    return wrapped


def dense_lin(i: int, o: int) -> nn.Module:
    return nn.Linear(i, o, bias=False)


def fact_lin(r: int) -> Callable[[int, int], nn.Module]:
    def f(i: int, o: int) -> nn.Module:
        return ZFactorizedLinear(i, o, rank=r, bias=False, init_method="svd")

    return f


class Block(nn.Module):
    def __init__(self, d: int, heads: int, lin: Callable[[int, int], nn.Module], eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.ln2 = nn.LayerNorm(d, eps=eps)
        self.qkv = lin(d, 3 * d)
        self.proj = lin(d, d)
        self.fc1 = lin(d, 4 * d)
        self.fc2 = lin(4 * d, d)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab: int,
        seq: int,
        d: int,
        layers: int,
        heads: int,
        lin: Callable[[int, int], nn.Module],
        ln_eps: float = 1e-5,
        emb_std: Optional[float] = 0.02,
    ):
        super().__init__()
        if d % heads != 0:
            raise ValueError("d debe ser multiplo de heads")
        if emb_std is not None and not (
            isinstance(emb_std, numbers.Real) and not isinstance(emb_std, bool) and math.isfinite(emb_std) and emb_std > 0
        ):
            raise ValueError("emb_std debe ser None (init N(0, 1) de nn.Embedding) o un real finito > 0")
        self.vocab, self.seq, self.d, self.heads = vocab, seq, d, heads
        # eps de LayerNorm es configuracion (widen lo reescala y no viaja en el
        # state_dict): reconstruir con GPT(..., ln_eps=model.ln_eps). emb_std solo
        # decide el init (un state_dict cargado lo pisa); se guarda como traza.
        self.ln_eps = ln_eps
        self.emb_std = emb_std
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        # Linea base sana (v1.5.0; docs/evidencia/T20-VEREDICTO.md): tok y pos ~ N(0, emb_std^2).
        # Con la cabeza atada, el N(0, 1) de nn.Embedding arranca prediciendo el token actual con
        # margen ~0.645*d (BPC inicial de cientos a d=768); con 0.02 es el del uniforme. El N(0, 1)
        # recien sorteado se ESCALA (no se vuelve a sortear): no consume RNG, asi que a igual
        # semilla los bloques y el estado del RNG global son los de 1.4.0, y tok/pos son
        # exactamente emb_std por los de 1.4.0.
        if emb_std is not None:
            with torch.no_grad():
                self.tok.weight.mul_(emb_std)
                self.pos.weight.mul_(emb_std)
        self.blocks = nn.ModuleList([Block(d, heads, lin, eps=ln_eps) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d, eps=ln_eps)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        pending = getattr(self, "_mzshape_pending_recs", False) or getattr(self, "_mzshape_new_param_sources", None)
        if pending and torch.is_grad_enabled():
            warnings.warn(
                "migracion de optimizer pendiente: el optimizer anterior ya no referencia estos "
                "parametros; llama migrate_optimizer (o usa apply_event) antes de entrenar",
                RuntimeWarning,
                stacklevel=2,
            )
        B, T = idx.shape
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            h = b(h)
        h = self.lnf(h)
        logits = h @ self.tok.weight.T  # head atado
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))
        return logits, loss


@dataclass
class MigrationRec:
    """Un parametro reemplazado: de donde a donde, y con que mapas."""

    old: nn.Parameter
    new: nn.Parameter
    row_map: torch.Tensor
    col_map: Optional[torch.Tensor]  # None para estados 1-D


def _preflight_model(model: GPT, require_factorized=False) -> None:
    """Comprueba toda la arquitectura soportada antes de reemplazar nada."""
    d = model.d
    for name in ("tok", "pos"):
        emb = getattr(model, name)
        if not isinstance(emb, nn.Embedding) or emb.embedding_dim != d or emb.max_norm is not None:
            raise ValueError("GPT requiere embeddings de ancho d, sin max_norm in-place")
    norms = [model.lnf]
    for block in model.blocks:
        if block.heads != model.heads:
            raise ValueError("todos los bloques deben conservar heads")
        norms.extend((block.ln1, block.ln2))
        for name, expected in (("qkv", (d, 3 * d)), ("proj", (d, d)), ("fc1", (d, 4 * d)), ("fc2", (4 * d, d))):
            layer = getattr(block, name)
            allowed = (ZFactorizedLinear,) if require_factorized else (nn.Linear, ZFactorizedLinear)
            if type(layer) not in allowed:
                raise TypeError("GPT requiere capas lineales soportadas y bloques factorizados para crecer")
            if (layer.in_features, layer.out_features) != expected:
                raise ValueError(f"dimensiones incompatibles en {name}")
    for ln in norms:
        if not isinstance(ln, nn.LayerNorm) or ln.normalized_shape != (d,) or ln.weight is None or ln.bias is None:
            raise ValueError("GPT requiere LayerNorm 1-D con weight y bias")


def qkv_out_map(d_old: int, d_new: int, heads: int) -> torch.Tensor:
    """Posiciones de las salidas viejas del qkv fusionado [q|k|v] en el layout
    nuevo, con insercion por cabeza (head_dim crece, heads fijos)."""
    hd, hd2 = d_old // heads, d_new // heads
    pos = [blk * d_new + h * hd2 + j for blk in range(3) for h in range(heads) for j in range(hd)]
    return torch.tensor(pos, dtype=torch.long)


def attn_in_map(d_old: int, d_new: int, heads: int) -> torch.Tensor:
    """Posiciones de las dims viejas de la salida de atencion (entrada de
    proj), interleaved por cabeza."""
    hd, hd2 = d_old // heads, d_new // heads
    return torch.tensor([h * hd2 + j for h in range(heads) for j in range(hd)], dtype=torch.long)


def _recs_fact(
    old: ZFactorizedLinear, new: ZFactorizedLinear, in_map: torch.Tensor, out_map: torch.Tensor
) -> List[MigrationRec]:
    ar = torch.arange(old.rank)
    recs = [
        MigrationRec(old.U, new.U, out_map, ar),
        MigrationRec(old.S, new.S, ar, None),
        MigrationRec(old.V, new.V, ar, in_map),
    ]
    if old.bias is not None:
        recs.append(MigrationRec(old.bias, new.bias, out_map, None))
    return recs


@_atomic_shape
def factorize_gpt(model: GPT) -> int:
    """Convierte in-place cada nn.Linear de los bloques a ZFactorizedLinear
    EXACTA (SVD completa, r = dim minima). Devuelve cuantas capas convirtio:
    con 0, el caller NO debe resetear el optimizer (factorize redundante
    sobre un modelo ya factorizado vaciaria el estado en silencio; G4-A)."""
    _preflight_model(model)
    n = 0
    for blk in model.blocks:
        for name in ("qkv", "proj", "fc1", "fc2"):
            lay = getattr(blk, name)
            if isinstance(lay, nn.Linear):
                setattr(blk, name, dense_to_factorized(lay))
                n += 1
    return n


@_atomic_shape
def widen_gpt(
    model: GPT, new_d: int, noise_scale: float = 1e-3, generator: Optional[torch.Generator] = None
) -> List[MigrationRec]:
    """Ensancha el modelo in-place a new_d (heads fijos, head_dim crece).
    Requiere bloques factorizados. Devuelve el ledger de migracion Adam.
    Deja migracion pendiente: hasta migrate_optimizer, un forward con gradiente emite RuntimeWarning."""
    d, heads = model.d, model.heads
    if (
        not isinstance(noise_scale, numbers.Real)
        or isinstance(noise_scale, bool)
        or not math.isfinite(noise_scale)
        or noise_scale < 0
    ):
        raise ValueError("noise_scale debe ser finito y >= 0")
    if new_d % heads != 0 or new_d < d:
        raise ValueError("new_d debe ser multiplo de heads y >= d actual")
    if new_d == d:
        return []
    if getattr(model, "_mzshape_pending_recs", False):
        raise RuntimeError(
            "ledger de migracion pendiente: migra el optimizer "
            "(migrate_optimizer/apply_event) antes de otro widen — un ledger "
            "caduco pierde momentum en silencio (G4-B)"
        )
    if getattr(model, "_mzshape_new_param_sources", {}):
        raise RuntimeError("migracion pendiente de bloques nuevos: llama migrate_optimizer antes de widen")
    # preflight ANTES de mutar nada: un TypeError a mitad dejaria embeddings
    # ensanchados con bloques viejos = modelo roto permanente (G4-B)
    _preflight_model(model, require_factorized=True)
    hd, hd2 = d // heads, new_d // heads
    pref = torch.arange(d)
    pref4 = torch.arange(4 * d)
    omap_qkv = qkv_out_map(d, new_d, heads)
    imap_attn = attn_in_map(d, new_d, heads)
    recs: List[MigrationRec] = []

    # embeddings: ruido en cols nuevas — via head atado da senal Y gradiente
    # a las dims nuevas del stream (la verdad de diseno anclada en M1)
    for name in ("tok", "pos"):
        old = getattr(model, name)
        new = widen_embedding(old, new_d, noise_scale=noise_scale, generator=generator)
        setattr(model, name, new)
        recs.append(MigrationRec(old.weight, new.weight, torch.arange(old.num_embeddings), pref))

    old_lnf = model.lnf
    model.lnf = widen_layernorm(old_lnf, new_d, variance_compensation=True)
    model.ln_eps = model.lnf.eps  # vive en __dict__: la transaccion lo restaura en rollback
    recs += [
        MigrationRec(old_lnf.weight, model.lnf.weight, pref, None),
        MigrationRec(old_lnf.bias, model.lnf.bias, pref, None),
    ]

    for blk in model.blocks:
        for ln_name in ("ln1", "ln2"):
            o = getattr(blk, ln_name)
            n = widen_layernorm(o, new_d, variance_compensation=True)
            setattr(blk, ln_name, n)
            recs += [MigrationRec(o.weight, n.weight, pref, None), MigrationRec(o.bias, n.bias, pref, None)]

        o = blk.qkv
        blk.qkv = widen_factorized_linear(
            o, new_d, 3 * new_d, in_map=pref, out_map=omap_qkv, noise_scale=noise_scale, generator=generator
        )
        # correccion exacta de escala SDPA: SOLO las filas q viejas (las
        # nuevas son ruido; la exactitud SDPA no las necesita y asi el
        # presupuesto ||dW||=ns*||W|| de qkv se cumple; _rescale_q_state usa
        # exactamente estas mismas filas)
        scale_output_rows(blk.qkv, omap_qkv[:d], math.sqrt(hd2 / hd))
        recs += _recs_fact(o, blk.qkv, pref, omap_qkv)

        o = blk.proj
        blk.proj = widen_factorized_linear(
            o, new_d, new_d, in_map=imap_attn, out_map=pref, noise_scale=noise_scale, generator=generator
        )
        recs += _recs_fact(o, blk.proj, imap_attn, pref)

        o = blk.fc1
        blk.fc1 = widen_factorized_linear(
            o, new_d, 4 * new_d, in_map=pref, out_map=pref4, noise_scale=noise_scale, generator=generator
        )
        recs += _recs_fact(o, blk.fc1, pref, pref4)

        o = blk.fc2
        blk.fc2 = widen_factorized_linear(
            o, 4 * new_d, new_d, in_map=pref4, out_map=pref, noise_scale=noise_scale, generator=generator
        )
        recs += _recs_fact(o, blk.fc2, pref4, pref)

    model.d = new_d
    model._mzshape_pending_recs = recs
    return recs


@_atomic_shape
def deepen_gpt(model: GPT, n_new: int) -> None:
    """Anade n_new bloques IDENTIDAD exactos al final (pre-lnf). Requiere
    bloques factorizados (hereda el rango del primero).
    Deja migracion pendiente: hasta migrate_optimizer, un forward con gradiente emite RuntimeWarning."""
    if n_new <= 0:
        return
    _preflight_model(model, require_factorized=True)
    if not model.blocks:
        raise ValueError("deepen_gpt requiere al menos un bloque plantilla")
    first = model.blocks[0]
    if not isinstance(first.qkv, ZFactorizedLinear):
        raise TypeError("deepen_gpt requiere bloques factorizados")
    dev = first.qkv.U.device
    dt = first.qkv.U.dtype
    sources = dict(getattr(model, "_mzshape_new_param_sources", {}))
    for _ in range(n_new):
        templates = iter(getattr(first, name) for name in ("qkv", "proj", "fc1", "fc2"))

        def lin(inputs, outputs, templates=templates):
            template = next(templates)
            layer = ZFactorizedLinear(inputs, outputs, rank=template.rank, bias=template.bias is not None, init_method="svd")
            layer._use_fp8 = template._use_fp8
            layer._epsi_scaling = template._epsi_scaling
            return layer

        b = Block(model.d, model.heads, lin, eps=first.ln1.eps).to(device=dev, dtype=dt)
        template_params = dict(first.named_parameters())
        for name, p in b.named_parameters():
            template = template_params[name]
            p.data = p.data.to(template.dtype)
            p.requires_grad_(template.requires_grad)
            sources[p] = template
        template_modules = dict(first.named_modules())
        for name, module in b.named_modules():
            module.training = template_modules[name].training
        zero_block_outputs(b.proj, b.fc2)
        model.blocks.append(b)
    model._mzshape_new_param_sources = sources


def _require_adamw(opt):
    if type(opt) is not torch.optim.AdamW:
        raise TypeError("ElasticShape soporta torch.optim.AdamW; otros optimizers requieren un adaptador")


def _new_adamw(model, old_opt, replacements, lr=None, weight_decay=None):
    """Reconstruye grupos/opciones, sin copiar aun los buffers de momentos."""
    _require_adamw(old_opt)
    live = {id(p) for p in model.parameters()}
    groups, ownership = [], {}
    for index, group in enumerate(old_opt.param_groups):
        new_group = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        new_group["params"] = []
        for old in group["params"]:
            for new in replacements.get(old, [old]):
                if id(new) not in live:
                    raise ValueError("optimizer/ledger caduco: parametro fuera del modelo")
                if id(new) in ownership:
                    raise ValueError("parametro duplicado en los grupos migrados")
                new_group["params"].append(new)
                ownership[id(new)] = index
        if lr is not None:
            new_group["lr"] = copy.deepcopy(lr)
            if "_mzshape_base_lr" in new_group:
                new_group["_mzshape_base_lr"] = copy.deepcopy(lr)
        if weight_decay is not None:
            new_group["weight_decay"] = weight_decay
        groups.append(new_group)
    # Cada parametro de un bloque nuevo hereda el grupo de su homologo en
    # el bloque plantilla. Un homologo excluido sigue excluido, sin adivinar.
    for new, source in getattr(model, "_mzshape_new_param_sources", {}).items():
        if id(new) not in live or id(source) not in live:
            raise ValueError("fuentes de parametros nuevos caducas")
        index = ownership.get(id(source))
        if index is not None and id(new) not in ownership:
            groups[index]["params"].append(new)
            ownership[id(new)] = index
    names = {id(p): name for name, p in model.named_parameters()}
    for group in groups:
        if "param_names" in group:
            group["param_names"] = [names[id(p)] for p in group["params"]]
    defaults = copy.deepcopy(old_opt.defaults)
    if lr is not None:
        defaults["lr"] = copy.deepcopy(lr)
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    # Algunas versiones guardan defaults internos (decoupled_weight_decay)
    # que no son argumentos del constructor AdamW publico.
    accepted = inspect.signature(torch.optim.AdamW).parameters
    new_opt = torch.optim.AdamW(groups, **{k: v for k, v in defaults.items() if k in accepted})
    new_opt.defaults.update(defaults)
    return new_opt


def migrate_optimizer(
    model: GPT,
    old_opt: torch.optim.Optimizer,
    recs: List[MigrationRec],
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> torch.optim.AdamW:
    """Migra AdamW conservando grupos, opciones y estado completo sin alias.

    None hereda LR/WD de CADA grupo. Overrides explicitos se aplican a todos.
    Los nuevos bloques heredan grupos por parametro del primer bloque;
    parametros excluidos del optimizer permanecen excluidos. AMSGrad se
    rellena junto con m/v. El reescalado SDPA lo hace apply_event.
    """
    _require_adamw(old_opt)
    live = {id(p) for p in model.parameters()}
    pending = getattr(model, "_mzshape_pending_recs", [])
    if recs and not pending:
        raise ValueError("ledger ya consumido o no emitido por widen_gpt: nada que migrar")
    if pending and (len(pending) != len(recs) or any(a is not b for a, b in zip(pending, recs))):
        raise ValueError("ledger de migracion pendiente distinto del proporcionado")
    if len({id(r.old) for r in recs}) != len(recs) or len({id(r.new) for r in recs}) != len(recs):
        raise ValueError("ledger duplicado")
    if any(id(r.new) not in live for r in recs):
        raise ValueError("ledger caduco: parametro nuevo fuera del modelo")
    new_opt = _new_adamw(model, old_opt, {r.old: [r.new] for r in recs}, lr, weight_decay)
    owned = {id(p) for group in new_opt.param_groups for p in group["params"]}
    replaced = {id(r.old) for r in recs}
    for group in old_opt.param_groups:
        for p in group["params"]:
            if id(p) in owned and id(p) not in replaced and p in old_opt.state:
                new_opt.state[p] = copy.deepcopy(old_opt.state[p])
    for rec in recs:
        st = old_opt.state.get(rec.old)
        if st and id(rec.new) in owned:
            if st["exp_avg"].shape != rec.old.shape:
                raise ValueError("momento AdamW incompatible con el parametro original")
            new_opt.state[rec.new] = pad_adam_entry(st, tuple(rec.new.shape), rec.row_map, rec.col_map)
    model._mzshape_pending_recs = False
    model._mzshape_new_param_sources = {}
    return new_opt


def _rescale_q_state(new_opt: torch.optim.Optimizer, model: GPT, d_old: int, d_new: int) -> None:
    """Correccion del estado migrado del bloque q tras la correccion SDPA:
    U_q se multiplico por c=sqrt(hd'/hd), asi que el gradiente futuro escala
    1/c (medido exacto por G4-A) -> m /= c, v /= c^2 en las filas q migradas
    (patron de re-escalado consistente validado en T7)."""
    heads = model.heads
    c = math.sqrt((d_new // heads) / (d_old // heads))
    q_rows = qkv_out_map(d_old, d_new, heads)[:d_old]  # bloque q = primeras d_old
    for blk in model.blocks:
        for parameter in (blk.qkv.U, blk.qkv.bias):
            st = new_opt.state.get(parameter)
            if st:
                st["exp_avg"][q_rows] /= c
                st["exp_avg_sq"][q_rows] /= c * c
                if "max_exp_avg_sq" in st:
                    st["max_exp_avg_sq"][q_rows] /= c * c


@torch.no_grad()
def rel_drift(model: GPT, probe_x: torch.Tensor, ref_logits: torch.Tensor) -> float:
    """Deriva relativa de logits vs ref_logits; corre bajo no_grad: no dispara el aviso de migracion pendiente."""
    logits, _ = model(probe_x)
    denom = ref_logits.norm().clamp_min(1e-12)
    return float((logits - ref_logits).norm() / denom)


@dataclass
class GrowthEvent:
    step: int
    factorize: bool = False
    new_d: Optional[int] = None
    add_layers: int = 0
    noise_scale: float = 1e-3


class ShapeSchedule:
    """Schedule declarativo: eventos por paso, consumidos una sola vez."""

    def __init__(self, events: List[GrowthEvent]):
        self._by_step: Dict[int, List[GrowthEvent]] = {}
        for e in events:
            self._by_step.setdefault(e.step, []).append(e)

    def pop_due(self, step: int) -> List[GrowthEvent]:
        """Devuelve (y consume) TODOS los eventos con paso <= step — catch-up:
        un loop que salta pasos (grad accumulation, resume con offset) no debe
        perder eventos en silencio (G4-B). Dentro de un mismo paso se preserva
        el orden de insercion; para factorize+widen usa UN evento combinado
        (apply_event fija el orden interno), no dos eventos separados."""
        due_steps = sorted(s for s in self._by_step if s <= step)
        out: List[GrowthEvent] = []
        for s in due_steps:
            out.extend(self._by_step.pop(s))
        return out

    @property
    def pending(self) -> int:
        return sum(len(v) for v in self._by_step.values())


class _GrowthRejected(RuntimeError):
    """Rechazo recuperable por calidad/no finitos; no es un error del caller."""


@torch.no_grad()
def _probe_logits(model, probe_x):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        # Una capa con U/V bf16 y S fp32 necesita autocast para el segundo
        # GEMM. Fijar la politica por dtype evita heredar el contexto caller.
        dtype = model.tok.weight.dtype
        low_precision = dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(device_type=probe_x.device.type, enabled=low_precision, dtype=dtype if low_precision else None):
            logits, _ = model(probe_x)
        if not torch.isfinite(logits).all():
            raise _GrowthRejected("logits no finitos en la sonda")
        return logits
    finally:
        for module, mode in modes:
            module.training = mode


@torch.no_grad()
def _probe_metrics(before, after, targets=None):
    # FP64 evita que el redondeo del CALCULO de KL/loss domine; si los logits
    # ya son bf16, su cuantizacion fija un piso de ~3e-5 nats en la KL.
    acc = torch.float32 if before.device.type == "mps" else torch.float64
    before, after = before.to(acc), after.to(acc)
    log_p, log_q = F.log_softmax(before, dim=-1), F.log_softmax(after, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean().clamp_min(0)
    result = {"logits_drift_rel": float((after - before).norm() / before.norm().clamp_min(1e-12)), "kl_div": float(kl)}
    if targets is not None:
        vocab = before.shape[-1]
        first = F.cross_entropy(before.reshape(-1, vocab), targets.reshape(-1))
        last = F.cross_entropy(after.reshape(-1, vocab), targets.reshape(-1))
        result.update(loss_before=float(first), loss_after=float(last), loss_delta=float(last - first))
    if not all(math.isfinite(value) for value in result.values()):
        raise _GrowthRejected("metricas no finitas en la sonda")
    return result


def apply_event(
    model: GPT,
    opt: torch.optim.Optimizer,
    ev: GrowthEvent,
    lr: Optional[float] = None,
    probe_x: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    weight_decay: Optional[float] = None,
    *,
    probe_targets: Optional[torch.Tensor] = None,
    max_loss_increase: Optional[float] = None,
    max_kl: Optional[float] = None,
):
    """Cirugia atomica; devuelve (optimizer, reporte con accepted).

    Las excepciones restauran topologia, referencias y RNG antes de propagarse.
    Un limite de calidad excedido/no finitos devuelve el optimizer ORIGINAL y
    accepted=False, rolled_back=True. Loss es CE media en nats; KL es media
    por token KL(p_antes || p_despues). Los limites se eligen antes del evento:
    no hay un umbral universal. Sin limites se conserva la API diagnostica.
    La sonda es opcional salvo al activar limites; loss necesita targets.
    LR/WD None hereda cada grupo. Crear LrWarmup solo si accepted=True.
    La conversion densa reinicia momentos (politica historica), no opciones.
    """
    _require_adamw(opt)
    _preflight_model(model)
    for name, threshold in (("max_loss_increase", max_loss_increase), ("max_kl", max_kl)):
        if threshold is not None and (not math.isfinite(threshold) or threshold < 0):
            raise ValueError(f"{name} debe ser finito y >= 0")
    guarded = max_loss_increase is not None or max_kl is not None
    if guarded and probe_x is None:
        raise ValueError("los limites de calidad requieren probe_x")
    if max_loss_increase is not None and probe_targets is None:
        raise ValueError("max_loss_increase requiere probe_targets")
    if probe_targets is not None and (probe_x is None or probe_targets.shape != probe_x.shape):
        raise ValueError("probe_targets requiere la misma forma que probe_x")
    if probe_targets is not None:
        valid_dtype = probe_targets.dtype == torch.int64  # cross_entropy exige Long
        valid_values = valid_dtype and bool(
            ((probe_targets == -100) | ((probe_targets >= 0) & (probe_targets < model.vocab))).all()
        )
        if not (valid_dtype and valid_values):
            raise ValueError("probe_targets debe ser entero con valores en [0, vocab) o -100")
    if (
        not isinstance(ev.noise_scale, numbers.Real)
        or isinstance(ev.noise_scale, bool)
        or not math.isfinite(ev.noise_scale)
        or ev.noise_scale < 0
    ):
        raise ValueError("noise_scale debe ser finito y >= 0")
    if not isinstance(ev.add_layers, int) or ev.add_layers < 0:
        raise ValueError("add_layers debe ser un entero >= 0")
    if ev.new_d is not None and (not isinstance(ev.new_d, int) or ev.new_d < model.d or ev.new_d % model.heads):
        raise ValueError("new_d debe ser entero, multiplo de heads y >= d")
    if getattr(model, "_mzshape_pending_recs", False) or getattr(model, "_mzshape_new_param_sources", {}):
        raise RuntimeError("migracion pendiente: termina migrate_optimizer antes de otro evento")
    original_opt = opt
    report: Dict[str, object] = {"step": ev.step, "optimizer": [], "accepted": True, "rolled_back": False, "guarded": guarded}
    try:
        with _shape_transaction(model, generator):
            ref = None
            if probe_x is not None:
                dev = next(model.parameters()).device
                probe_x = probe_x.to(dev)
                if probe_targets is not None:
                    probe_targets = probe_targets.to(dev)
                ref = _probe_logits(model, probe_x)
            n_conv = 0
            if ev.factorize:
                old_layers = [
                    (block, name, getattr(block, name))
                    for block in model.blocks
                    for name in ("qkv", "proj", "fc1", "fc2")
                    if isinstance(getattr(block, name), nn.Linear)
                ]
                n_conv = factorize_gpt(model)
                if n_conv:
                    replacements = {}
                    for block, name, old in old_layers:
                        new = getattr(block, name)
                        replacements[old.weight] = [new.U, new.S, new.V]
                        if old.bias is not None:
                            replacements[old.bias] = [new.bias]
                    opt = _new_adamw(model, opt, replacements, lr, weight_decay)
                    report["optimizer"].append(f"reset de momentos (conversion densa->factorizada, {n_conv} capas)")
                else:
                    report["optimizer"].append("factorize redundante: 0 capas convertidas, estado INTACTO")
            d_old = model.d
            recs = []
            if ev.new_d is not None:
                try:
                    recs = widen_gpt(model, ev.new_d, noise_scale=ev.noise_scale, generator=generator)
                except NoiseError as e:
                    raise _GrowthRejected(f"ruido no representable: {e}") from e
            if ev.add_layers:
                deepen_gpt(model, ev.add_layers)
            if recs or ev.add_layers:
                opt = migrate_optimizer(model, opt, recs, lr, weight_decay)
                report["optimizer"].append("migrado (ledger aplicado, grupos y opciones conservados)")
                if recs:
                    _rescale_q_state(opt, model, d_old, model.d)
                    report["optimizer"].append("estado q re-escalado (1/c, 1/c^2; incluye AMSGrad)")
            elif lr is not None or weight_decay is not None:
                # sin cirugia (factorize redundante o new_d == d): sin esto, un
                # override explicito de lr/weight_decay se descartaba en
                # silencio pese a accepted=True (PER-LOG-002).
                if not n_conv:
                    opt = migrate_optimizer(model, opt, [], lr, weight_decay)
                    report["optimizer"].append("overrides lr/weight_decay aplicados sin cirugia")
            for parameter in model.parameters():
                if not torch.isfinite(parameter).all():
                    raise _GrowthRejected("parametros no finitos despues de la cirugia")
            for state in opt.state.values():
                if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in state.values()):
                    raise _GrowthRejected("estado del optimizer no finito")
            if ref is not None:
                report.update(_probe_metrics(ref, _probe_logits(model, probe_x), probe_targets))
                if max_loss_increase is not None and report["loss_delta"] > max_loss_increase:
                    raise _GrowthRejected("incremento de loss excede max_loss_increase")
                if max_kl is not None and report["kl_div"] > max_kl:
                    raise _GrowthRejected("KL excede max_kl")
    except _GrowthRejected as error:
        report.update(accepted=False, rolled_back=True, rejection_reason=str(error))
        report["optimizer"].append("rollback: optimizer original INTACTO")
        return original_opt, report
    return opt, report


class LrWarmup:
    """Rampa lineal floor->1.0 del LR tras un growth (absorbe el transitorio
    de LN residual y del estado Adam aproximado; SPEC §5/§8).

    Regla "la escritura externa gana": la base persistente de un warmup sin
    terminar solo se hereda si el LR del grupo sigue siendo el ultimo que
    escribio un warmup (`_mzshape_warmup_lr`); si alguien escribio el LR
    despues (scheduler, reanudacion con otro LR), la base es ese LR vigente.
    Coincidencia benigna: una escritura externa que deja exactamente el mismo
    valor que escribio el warmup se trata como no tocada."""

    def __init__(self, opt: torch.optim.Optimizer, steps: int, floor: float = 0.1):
        if steps <= 0:
            raise ValueError("steps > 0")
        if not (math.isfinite(floor) and 0.0 <= floor <= 1.0):
            raise ValueError("floor debe estar en [0, 1]")
        self.opt, self.steps, self.floor = opt, steps, floor
        # base PERSISTENTE en el param_group: un warmup construido sobre otro
        # sin terminar capturaria el lr ya escalado y perderia el objetivo
        # real para siempre (G4-B). La clave sobrevive entre instancias, pero
        # solo se hereda si nadie escribio el LR despues del ultimo warmup.
        for g in opt.param_groups:
            if "_mzshape_base_lr" in g and g.get("_mzshape_warmup_lr") == float(g["lr"]):
                base = g["_mzshape_base_lr"]  # warmup activo sin tocar (G4-B)
            else:
                base = g["lr"]  # sin warmup activo, o la escritura externa gana
            g["_mzshape_base_lr"] = base
        self.base = [g["_mzshape_base_lr"] for g in opt.param_groups]
        self.t = 0
        self._apply(floor)

    def _apply(self, s: float) -> None:
        for g, b in zip(self.opt.param_groups, self.base, strict=True):
            g["lr"] = b * s
            # copia ESCALAR (un LR tensor seria alias: fill_ de un scheduler externo no se detectaria)
            g["_mzshape_warmup_lr"] = float(g["lr"])

    def step(self) -> None:
        if self.done:
            return
        self.t += 1
        s = self.floor + (1.0 - self.floor) * min(1.0, self.t / self.steps)
        self._apply(s)
        if self.done:
            # warmup terminado: la base persistente ya cumplio su proposito
            # (G4-B, evitar que un warmup sin terminar pierda su objetivo).
            # Si sobrevive, el PROXIMO warmup rampea hasta este pico caduco
            # en vez de partir del LR vigente (PER-LOG-001).
            for g in self.opt.param_groups:
                g.pop("_mzshape_base_lr", None)
                g.pop("_mzshape_warmup_lr", None)

    @property
    def done(self) -> bool:
        return self.t >= self.steps
