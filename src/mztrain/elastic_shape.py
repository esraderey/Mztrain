"""ElasticShape v1 — crecimiento de forma en espacio factorizado.

Modelo GPT de referencia (convencion del arco empirico: heads fijos,
head_dim = d/heads, head atado, bias=False) + controller del morph:
conversion densa->factorizada exacta, ensanchado con ledger de migracion
Adam, profundizado identidad, probe de deriva, schedule y warmup.

Claim v1 (T8, preregistrado; docs/evidencia/T8-VEREDICTO.md): el morph
alcanza la calidad del from-scratch en ~54% del reloj a escala 11M-equiv.
Alcance y limites en docs/SPEC-elasticshape-v1.md. API opt-in: nada de
este modulo se activa salvo llamada explicita.

Decisiones documentadas: LN con variance_compensation=True; estado q
re-escalado (1/c, 1/c^2) tras la correccion SDPA; conversion densa->
factorizada resetea el optimizer (parametrizaciones no conmensurables).
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ZFactorizedLinear
from .shape_ops import (
    dense_to_factorized,
    pad_adam_entry,
    scale_output_rows,
    widen_embedding,
    widen_factorized_linear,
    widen_layernorm,
    zero_block_outputs,
)

def dense_lin(i: int, o: int) -> nn.Module:
    return nn.Linear(i, o, bias=False)


def fact_lin(r: int) -> Callable[[int, int], nn.Module]:
    def f(i: int, o: int) -> nn.Module:
        return ZFactorizedLinear(i, o, rank=r, bias=False, init_method="svd")
    return f


class Block(nn.Module):
    def __init__(self, d: int, heads: int, lin: Callable[[int, int], nn.Module]):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
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
    def __init__(self, vocab: int, seq: int, d: int, layers: int, heads: int,
                 lin: Callable[[int, int], nn.Module]):
        super().__init__()
        if d % heads != 0:
            raise ValueError("d debe ser multiplo de heads")
        self.vocab, self.seq, self.d, self.heads = vocab, seq, d, heads
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([Block(d, heads, lin) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
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


def qkv_out_map(d_old: int, d_new: int, heads: int) -> torch.Tensor:
    """Posiciones de las salidas viejas del qkv fusionado [q|k|v] en el layout
    nuevo, con insercion por cabeza (head_dim crece, heads fijos)."""
    hd, hd2 = d_old // heads, d_new // heads
    pos = [blk * d_new + h * hd2 + j
           for blk in range(3) for h in range(heads) for j in range(hd)]
    return torch.tensor(pos, dtype=torch.long)


def attn_in_map(d_old: int, d_new: int, heads: int) -> torch.Tensor:
    """Posiciones de las dims viejas de la salida de atencion (entrada de
    proj), interleaved por cabeza."""
    hd, hd2 = d_old // heads, d_new // heads
    return torch.tensor([h * hd2 + j for h in range(heads) for j in range(hd)],
                        dtype=torch.long)


def _recs_fact(old: ZFactorizedLinear, new: ZFactorizedLinear,
               in_map: torch.Tensor, out_map: torch.Tensor) -> List[MigrationRec]:
    ar = torch.arange(old.rank)
    return [
        MigrationRec(old.U, new.U, out_map, ar),
        MigrationRec(old.S, new.S, ar, None),
        MigrationRec(old.V, new.V, ar, in_map),
    ]


def factorize_gpt(model: GPT) -> int:
    """Convierte in-place cada nn.Linear de los bloques a ZFactorizedLinear
    EXACTA (SVD completa, r = dim minima). Devuelve cuantas capas convirtio:
    con 0, el caller NO debe resetear el optimizer (factorize redundante
    sobre un modelo ya factorizado vaciaria el estado en silencio; G4-A)."""
    n = 0
    for blk in model.blocks:
        for name in ("qkv", "proj", "fc1", "fc2"):
            lay = getattr(blk, name)
            if isinstance(lay, nn.Linear):
                setattr(blk, name, dense_to_factorized(lay))
                n += 1
    return n


def widen_gpt(model: GPT, new_d: int, noise_scale: float = 1e-3,
              generator: Optional[torch.Generator] = None) -> List[MigrationRec]:
    """Ensancha el modelo in-place a new_d (heads fijos, head_dim crece).
    Requiere bloques factorizados. Devuelve el ledger de migracion Adam."""
    d, heads = model.d, model.heads
    if new_d % heads != 0 or new_d < d:
        raise ValueError("new_d debe ser multiplo de heads y >= d actual")
    if new_d == d:
        return []
    if getattr(model, "_mzshape_pending_recs", False):
        raise RuntimeError(
            "ledger de migracion pendiente: migra el optimizer "
            "(migrate_optimizer/apply_event) antes de otro widen — un ledger "
            "caduco pierde momentum en silencio (G4-B)")
    # preflight ANTES de mutar nada: un TypeError a mitad dejaria embeddings
    # ensanchados con bloques viejos = modelo roto permanente (G4-B)
    for blk in model.blocks:
        if not isinstance(blk.qkv, ZFactorizedLinear):
            raise TypeError("widen_gpt requiere bloques factorizados "
                            "(llama factorize_gpt primero)")
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
        new = widen_embedding(old, new_d, noise_scale=noise_scale,
                              generator=generator)
        setattr(model, name, new)
        recs.append(MigrationRec(old.weight, new.weight,
                                 torch.arange(old.num_embeddings), pref))

    old_lnf = model.lnf
    model.lnf = widen_layernorm(old_lnf, new_d, variance_compensation=True)
    recs += [MigrationRec(old_lnf.weight, model.lnf.weight, pref, None),
             MigrationRec(old_lnf.bias, model.lnf.bias, pref, None)]

    for blk in model.blocks:
        for ln_name in ("ln1", "ln2"):
            o = getattr(blk, ln_name)
            n = widen_layernorm(o, new_d, variance_compensation=True)
            setattr(blk, ln_name, n)
            recs += [MigrationRec(o.weight, n.weight, pref, None),
                     MigrationRec(o.bias, n.bias, pref, None)]

        o = blk.qkv
        blk.qkv = widen_factorized_linear(o, new_d, 3 * new_d, in_map=pref,
                                          out_map=omap_qkv,
                                          noise_scale=noise_scale,
                                          generator=generator)
        # correccion exacta de escala SDPA sobre el bloque q completo
        scale_output_rows(blk.qkv, torch.arange(new_d), math.sqrt(hd2 / hd))
        recs += _recs_fact(o, blk.qkv, pref, omap_qkv)

        o = blk.proj
        blk.proj = widen_factorized_linear(o, new_d, new_d, in_map=imap_attn,
                                           out_map=pref,
                                           noise_scale=noise_scale,
                                           generator=generator)
        recs += _recs_fact(o, blk.proj, imap_attn, pref)

        o = blk.fc1
        blk.fc1 = widen_factorized_linear(o, new_d, 4 * new_d, in_map=pref,
                                          out_map=pref4,
                                          noise_scale=noise_scale,
                                          generator=generator)
        recs += _recs_fact(o, blk.fc1, pref, pref4)

        o = blk.fc2
        blk.fc2 = widen_factorized_linear(o, 4 * new_d, new_d, in_map=pref4,
                                          out_map=pref,
                                          noise_scale=noise_scale,
                                          generator=generator)
        recs += _recs_fact(o, blk.fc2, pref4, pref)

    model.d = new_d
    model._mzshape_pending_recs = True
    return recs


def deepen_gpt(model: GPT, n_new: int) -> None:
    """Anade n_new bloques IDENTIDAD exactos al final (pre-lnf). Requiere
    bloques factorizados (hereda el rango del primero)."""
    if n_new <= 0:
        return
    first = model.blocks[0]
    if not isinstance(first.qkv, ZFactorizedLinear):
        raise TypeError("deepen_gpt requiere bloques factorizados")
    r = first.qkv.rank
    dev = first.qkv.U.device
    for _ in range(n_new):
        b = Block(model.d, model.heads, fact_lin(r)).to(dev)
        zero_block_outputs(b.proj, b.fc2)
        model.blocks.append(b)


def migrate_optimizer(model: GPT, old_opt: torch.optim.Optimizer,
                      recs: List[MigrationRec], lr: Optional[float] = None,
                      weight_decay: Optional[float] = None) -> torch.optim.AdamW:
    """AdamW nuevo sobre el modelo crecido. lr/weight_decay None = HEREDAR del
    optimizer viejo (pisar wd en silencio con un default fue hallazgo G4-B).
    Params intactos (solo posible en deepen-solo: widen reemplaza el 100%)
    trasplantan su entrada tal cual; los reemplazados migran con zero-pad
    segun su ledger. Bloques nuevos entran recien nacidos."""
    g0 = old_opt.param_groups[0]
    if lr is None:
        lr = g0["lr"]
    if weight_decay is None:
        weight_decay = g0.get("weight_decay", 0.0)
    new_opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    live = {id(p) for p in model.parameters()}
    replaced = {id(r.old) for r in recs}
    for group in old_opt.param_groups:
        for p in group["params"]:
            if id(p) in live and id(p) not in replaced and p in old_opt.state:
                new_opt.state[p] = old_opt.state[p]
    for rec in recs:
        st = old_opt.state.get(rec.old)
        if not st:
            continue
        shape = tuple(rec.new.shape)
        if rec.new.ndim == 1:
            new_opt.state[rec.new] = pad_adam_entry(st, shape,
                                                    row_map=rec.row_map)
        else:
            new_opt.state[rec.new] = pad_adam_entry(st, shape,
                                                    row_map=rec.row_map,
                                                    col_map=rec.col_map)
    model._mzshape_pending_recs = False
    return new_opt


def _rescale_q_state(new_opt: torch.optim.Optimizer, model: GPT,
                     d_old: int, d_new: int) -> None:
    """Correccion del estado migrado del bloque q tras la correccion SDPA:
    U_q se multiplico por c=sqrt(hd'/hd), asi que el gradiente futuro escala
    1/c (medido exacto por G4-A) -> m /= c, v /= c^2 en las filas q migradas
    (patron de re-escalado consistente validado en T7)."""
    heads = model.heads
    c = math.sqrt((d_new // heads) / (d_old // heads))
    q_rows = qkv_out_map(d_old, d_new, heads)[:d_old]  # bloque q = primeras d_old
    for blk in model.blocks:
        st = new_opt.state.get(blk.qkv.U)
        if st:
            st["exp_avg"][q_rows, :] /= c
            st["exp_avg_sq"][q_rows, :] /= c * c


@torch.no_grad()
def rel_drift(model: GPT, probe_x: torch.Tensor,
              ref_logits: torch.Tensor) -> float:
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


def apply_event(model: GPT, opt: torch.optim.Optimizer, ev: GrowthEvent,
                lr: Optional[float] = None,
                probe_x: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None,
                weight_decay: Optional[float] = None):
    """Aplica un GrowthEvent completo. Devuelve (optimizer_nuevo, reporte).
    lr/weight_decay None = heredar del optimizer actual. El caller crea su
    LrWarmup con el optimizer DEVUELTO (nunca con el viejo)."""
    if lr is None:
        lr = opt.param_groups[0]["lr"]
    report: Dict[str, object] = {"step": ev.step, "optimizer": []}
    if probe_x is not None:
        probe_x = probe_x.to(next(model.parameters()).device)
    ref = None
    if probe_x is not None:
        with torch.no_grad():
            ref, _ = model(probe_x)
    if ev.factorize:
        n_conv = factorize_gpt(model)
        if n_conv > 0:
            opt = torch.optim.AdamW(
                model.parameters(), lr=lr,
                weight_decay=opt.param_groups[0].get("weight_decay", 0.0)
                if weight_decay is None else weight_decay)
            report["optimizer"].append(
                f"reset (conversion densa->factorizada, {n_conv} capas)")
        else:
            report["optimizer"].append(
                "factorize redundante: 0 capas convertidas, estado INTACTO")
    d_old = model.d
    recs: List[MigrationRec] = []
    if ev.new_d is not None:
        recs = widen_gpt(model, ev.new_d, noise_scale=ev.noise_scale,
                         generator=generator)
    if ev.add_layers > 0:
        deepen_gpt(model, ev.add_layers)
    if recs or ev.add_layers > 0:
        opt = migrate_optimizer(model, opt, recs, lr, weight_decay)
        report["optimizer"].append("migrado (ledger aplicado)")
        if recs and model.d != d_old:
            _rescale_q_state(opt, model, d_old, model.d)
            report["optimizer"].append("estado q re-escalado (1/c, 1/c^2)")
    if probe_x is not None and ref is not None:
        report["logits_drift_rel"] = rel_drift(model, probe_x, ref)
    return opt, report


class LrWarmup:
    """Rampa lineal floor->1.0 del LR tras un growth (absorbe el transitorio
    de LN residual y del estado Adam aproximado; SPEC §5/§8)."""
    def __init__(self, opt: torch.optim.Optimizer, steps: int,
                 floor: float = 0.1):
        if steps <= 0:
            raise ValueError("steps > 0")
        self.opt, self.steps, self.floor = opt, steps, floor
        # base PERSISTENTE en el param_group: un warmup construido sobre otro
        # sin terminar capturaria el lr ya escalado y perderia el objetivo
        # real para siempre (G4-B). La clave sobrevive entre instancias.
        for g in opt.param_groups:
            g.setdefault("_mzshape_base_lr", g["lr"])
        self.base = [g["_mzshape_base_lr"] for g in opt.param_groups]
        self.t = 0
        self._apply(floor)

    def _apply(self, s: float) -> None:
        for g, b in zip(self.opt.param_groups, self.base, strict=True):
            g["lr"] = b * s

    def step(self) -> None:
        if self.done:
            return
        self.t += 1
        s = self.floor + (1.0 - self.floor) * min(1.0, self.t / self.steps)
        self._apply(s)

    @property
    def done(self) -> bool:
        return self.t >= self.steps
