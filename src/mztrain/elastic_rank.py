"""
MZTrain - ElasticRank: rango bidireccional (grow / sleep / revive / prune).

El crecimiento progresivo de rango solo sabe AÑADIR capacidad. ElasticRank
hace el rango elastico: una direccion singular que deja de aportar puede
DORMIR (sus factores + momentum de Adam se mueven a un sleep bank en CPU y
baja precision), REVIVIR si vuelve a hacer falta capacidad, y PODARSE si
lleva demasiado tiempo dormida. No se borra informacion inmediatamente: una
direccion dormida conserva U_i, S_i, V_i y sus estados exp_avg/exp_avg_sq,
asi que revivirla parte de donde se quedo en lugar de un cold-start.

Diseño (acordado con el usuario):

  Señal de sleep = HIBRIDA, con AND (no producto):
    spectral_score_i = |S_i|*||U_:,i||*||V_i,:|| / quantile(raw, q)
    update_score_i   = energia del paso efectivo de Adam por direccion
                       (m / (sqrt(v)+eps)) propagada por la regla del
                       producto sobre U,S,V, normalizada por la mediana.
    dead_i = spectral_ema_i < tau_s AND update_ema_i < tau_u
             AND low_counter_i >= patience AND age_i >= min_age
             AND revive_cd_i == 0 AND no-grace AND no-warmup
  (Una direccion con score espectral bajo pero update alto sigue
   aprendiendo: el AND evita matarla.)

  [v2] Señal de redundancia funcional (camino independiente, opt-in,
  on por defecto): la señal v1 solo ve direcciones cuyo |S_i|·‖U_i‖·‖V_i‖
  decae; el sesgo low-rank de GD actua sobre el PRODUCTO W, no sobre cada
  triplete, asi que en entrenamiento desde cero las direcciones redundantes
  mantienen magnitud comparable y v1 no las ve (verificado en benchmark:
  0 dormidas desde cero). v2 añade leverage estadistico del Gram de los
  terminos rango-1 T_i = S_i·u_i·v_i^T: coherence_i -> 1 cuando T_i cae en
  el span de las otras (dependencia lineal del conjunto, no solo duplicado
  par-a-par). Esta via NO usa el gate de actividad.

  Cadencia = observar barato cada N steps, mutar topologia caro solo en el
  boundary de epoch (fusionado con grow/refactor: una sola reconstruccion de
  optimizer, una sola migracion de estados, un solo warmup). Entre medias,
  'soft sleep': congelar (grad=0) sin cambiar shapes.

  Revive = el scheduler global manda: cuando pide mas rango, se revive
  primero (mejores sleepers por wake_score) y solo se rellena con ruido el
  resto. Swap rank-neutral / crecimiento independiente quedan tras flags
  avanzados (off por defecto en v1).

Este modulo es puro (scoring, sleep bank, planificacion). La cirugia de
factores + reconstruccion/migracion del optimizer la orquesta el engine
(ZTrainEngine._apply_elastic_topology), reutilizando el mismo patron ya
probado del camino de crecimiento.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

logger = logging.getLogger("mztrain")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.elastic_rank. "
        "Instalar con: pip install torch>=2.0.0"
    )

_EPS = 1e-12
_STORE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ============================================================================
# DECOMPRESION DE ESTADOS DEL OPTIMIZER (optimizer-agnostico, best-effort)
# ============================================================================

def decompress_opt_state(
    optimizer: "torch.optim.Optimizer",
    state: dict,
    key: str,
    param: "torch.Tensor",
) -> Optional["torch.Tensor"]:
    """Devolver un tensor FP32 utilizable para state[key], o None.

    Maneja:
    - tensor plano con shape del parametro (Adam estandar / GaLore / adaptive),
    - tuplas comprimidas de ZCompressedAdam (INT8 lineal para m, INT8 log
      para v) usando los decompresores del propio optimizer,
    - cualquier otro caso -> None (el caller trata como zeros).
    """
    if not state:
        return None
    val = state.get(key)
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        if tuple(val.shape) == tuple(param.shape):
            return val.detach().to(torch.float32)
        return None
    if state.get("compressed", False):
        try:
            if key == "exp_avg_sq" and hasattr(optimizer, "_decompress_v"):
                return optimizer._decompress_v(val, torch.float32).detach().to(
                    torch.float32
                )
            if hasattr(optimizer, "_decompress_state"):
                return optimizer._decompress_state(val, torch.float32).detach().to(
                    torch.float32
                )
        except Exception as e:  # pragma: no cover - defensivo
            logger.debug(f"[ElasticRank] decompress fallback ({key}): {e}")
    return None


# ============================================================================
# SLEEP BANK
# ============================================================================

@dataclass
class SleepingDirection:
    """Una direccion singular dormida (almacenada en CPU, baja precision).

    Conserva los factores y el momentum de Adam para poder revivir desde
    donde se quedo. Los tensores van en CPU y store_dtype (fp16 por defecto):
    no necesitan precision alta, solo sirven como punto de partida.
    """
    layer_name: str
    u: torch.Tensor          # (out_features,)
    s: torch.Tensor          # escalar ()
    v: torch.Tensor          # (in_features,)
    m_u: torch.Tensor        # exp_avg de U[:, i]
    m_s: torch.Tensor        # exp_avg de S[i]
    m_v: torch.Tensor        # exp_avg de V[i, :]
    vsq_u: torch.Tensor      # exp_avg_sq de U[:, i]
    vsq_s: torch.Tensor      # exp_avg_sq de S[i]
    vsq_v: torch.Tensor      # exp_avg_sq de V[i, :]
    step: int
    spectral_at_sleep: float
    update_at_sleep: float
    epoch_slept: int
    revive_cooldown: int = 0

    def epochs_asleep(self, current_epoch: int) -> int:
        return max(0, current_epoch - self.epoch_slept)

    def wake_score(self, current_epoch: int, prune_after: int) -> float:
        """Prioridad para revivir. Mayor = mejor candidato.

        Combina la importancia espectral y la actividad de update que tenia
        justo antes de dormir, con un decaimiento por antiguedad (un sleeper
        muy viejo casi seguro ya no encaja en el subespacio actual).
        """
        freshness = 0.5 ** (self.epochs_asleep(current_epoch) / max(prune_after, 1))
        return (0.6 * self.spectral_at_sleep
                + 0.4 * self.update_at_sleep) * freshness


# ============================================================================
# ESTADO DE SCORING POR CAPA
# ============================================================================

@dataclass
class _LayerState:
    """EMAs y contadores por-direccion, alineados al rango activo actual."""
    spectral_ema: torch.Tensor   # (rank,)
    update_ema: torch.Tensor     # (rank,)
    low_counter: torch.Tensor    # (rank,) int
    age: torch.Tensor            # (rank,) int  (chequeos desde nacimiento)
    revive_cd: torch.Tensor      # (rank,) int  (cooldown anti-oscilacion)
    wake_left: torch.Tensor      # (rank,) int  (steps restantes de wake warmup)
    redund_ema: torch.Tensor     # (rank,) v2: coherence EMA en [0, 1)
    redund_counter: torch.Tensor  # (rank,) int  v2: paciencia de redundancia

    @staticmethod
    def fresh(rank: int) -> "_LayerState":
        return _LayerState(
            spectral_ema=torch.ones(rank),
            update_ema=torch.ones(rank),
            low_counter=torch.zeros(rank, dtype=torch.long),
            age=torch.zeros(rank, dtype=torch.long),
            revive_cd=torch.zeros(rank, dtype=torch.long),
            wake_left=torch.zeros(rank, dtype=torch.long),
            # redund_ema arranca en 0: una direccion nueva NO es redundante
            # por defecto (al reves que spectral/update, que arrancan en 1).
            redund_ema=torch.zeros(rank),
            redund_counter=torch.zeros(rank, dtype=torch.long),
        )


@dataclass
class LayerPlan:
    """Plan estructural para una capa en el boundary de epoch."""
    name: str
    keep: List[int]                     # indices activos viejos a conservar
    sleep: List[int]                    # indices activos viejos a compactar
    revive: List[SleepingDirection]     # sleepers a traer de vuelta
    grow: int                           # componentes nuevos (ruido)
    new_rank: int

    @property
    def changed(self) -> bool:
        return bool(self.sleep) or bool(self.revive) or self.grow > 0


@dataclass
class ElasticPlan:
    """Plan global (todas las capas) + contadores para logging."""
    layers: Dict[str, LayerPlan] = field(default_factory=dict)
    pruned: int = 0

    def has_changes(self) -> bool:
        return any(lp.changed for lp in self.layers.values())

    def changed_layers(self):
        return ((n, lp) for n, lp in self.layers.items() if lp.changed)

    def summary(self) -> Dict[str, int]:
        return {
            "layers_changed": sum(1 for lp in self.layers.values() if lp.changed),
            "slept": sum(len(lp.sleep) for lp in self.layers.values()),
            "revived": sum(len(lp.revive) for lp in self.layers.values()),
            "grown": sum(lp.grow for lp in self.layers.values()),
            "pruned": self.pruned,
        }


# ============================================================================
# CONTROLLER
# ============================================================================

class ElasticRankController:
    """Cerebro de ElasticRank: mide señales, decide, mantiene el sleep bank.

    Solo opera sobre ZFactorizedLinear puro (no ZSparseFactorizedLinear, cuyo
    componente sparse depende de W completo y complica la cirugia; queda para
    una version posterior). El engine excluye sparse simetricamente.

    Args:
        config: ZTrainConfig con los hiperparametros elastic_rank_*.
    """

    def __init__(self, config: Any):
        self.cfg = config
        self.store_dtype = _STORE_DTYPES.get(
            getattr(config, "elastic_rank_sleep_store_dtype", "float16"),
            torch.float16,
        )
        # Estado de scoring por nombre de capa (estable entre epochs).
        self._states: Dict[str, _LayerState] = {}
        # Sleep bank: nombre de capa -> lista de SleepingDirection.
        self.bank: Dict[str, List[SleepingDirection]] = {}
        # Gracia global tras growth/refactor (chequeos en los que NO se duerme).
        self._growth_grace: int = 0
        self._refactor_grace: int = 0
        # Stats acumuladas (para get_stats / logging).
        self._total_slept = 0
        self._total_revived = 0
        self._total_pruned = 0
        self._checks = 0
        # Historial del probe de loss-sensitivity (modo diagnostico).
        self._probe_history: List[Dict[str, Any]] = []
        # Cooldown por capa tras un rollback de compactacion (en epochs).
        self._rollback_cd: Dict[str, int] = {}
        self._total_rollbacks = 0

    # ---- helpers de modulos elegibles -------------------------------------

    @staticmethod
    def _eligible(module: nn.Module) -> bool:
        # Importacion diferida para evitar ciclo con layers.py.
        from .layers import ZFactorizedLinear, ZSparseFactorizedLinear
        return (isinstance(module, ZFactorizedLinear)
                and not isinstance(module, ZSparseFactorizedLinear))

    def _iter_layers(self, model: nn.Module):
        for name, module in model.named_modules():
            if self._eligible(module):
                yield name, module

    def _state_for(self, name: str, rank: int) -> _LayerState:
        st = self._states.get(name)
        if st is None or st.spectral_ema.numel() != rank:
            # Primera vez o el rango cambio por fuera (p. ej. refactorize):
            # reconstruir el estado de scoring a tamaño actual.
            st = _LayerState.fresh(rank)
            self._states[name] = st
        return st

    # ---- señales -----------------------------------------------------------

    @staticmethod
    def _adam_step(
        optimizer, param: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Paso efectivo de Adam por parametro: m / (sqrt(v) + eps).

        Si no hay estado todavia -> zeros (direccion sin actividad medible).
        Si solo hay m -> usar m (mejor que nada).
        """
        st = optimizer.state.get(param)
        zeros = torch.zeros_like(param.detach(), dtype=torch.float32)
        if not st:
            return zeros
        m = decompress_opt_state(optimizer, st, "exp_avg", param)
        if m is None:
            return zeros
        v = decompress_opt_state(optimizer, st, "exp_avg_sq", param)
        if v is None:
            return m
        return m / (v.sqrt() + eps)

    @torch.no_grad()
    def _scores(
        self, module: nn.Module, optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(spectral_score, update_score) por direccion, en CPU float32."""
        U = module.U.detach().float()
        S = module.S.detach().float()
        V = module.V.detach().float()

        u_norm = U.norm(dim=0)          # (rank,)
        v_norm = V.norm(dim=1)          # (rank,)
        s_abs = S.abs()

        raw = s_abs * u_norm * v_norm
        q = float(getattr(self.cfg, "elastic_rank_score_quantile", 0.95))
        if q >= 1.0:
            scale = raw.max()
        else:
            scale = torch.quantile(raw, q)
        scale = scale.clamp_min(_EPS)
        spectral = (raw / scale).clamp(max=1.0)

        step_U = self._adam_step(optimizer, module.U)
        step_S = self._adam_step(optimizer, module.S)
        step_V = self._adam_step(optimizer, module.V)

        # Regla del producto: como cambia |S_i|*||U_:,i||*||V_i,:|| si cada
        # factor da su paso de Adam (aprox de primer orden, suma de los tres
        # terminos en lugar de derivada exacta).
        update_raw = (
            s_abs * step_U.norm(dim=0) * v_norm
            + step_S.abs() * u_norm * v_norm
            + s_abs * u_norm * step_V.norm(dim=1)
        )
        med = update_raw.median().clamp_min(_EPS)
        update = update_raw / med

        return spectral.cpu(), update.cpu()

    @torch.no_grad()
    def _redundancy(self, module: nn.Module) -> torch.Tensor:
        """[v2] coherence_i en [0,1): cuanto del termino rango-1 i esta
        contenido en el span de los DEMAS (dependencia lineal del conjunto),
        ESCALA-INVARIANTE (independiente de la magnitud de la direccion).

        Se normaliza cada T_i a norma 1 y se usa la matriz de correlacion R
        (R_ii=1) con ridge uniforme pequeño. leverage lev_i = (R+λI)^{-1}_ii:
        =1 si T̂_i ⟂ resto (no redundante), crece sin cota si cae en su span.
        coherence_i = 1 - 1/lev_i ∈ [0,1).

        Por que escala-invariante: en un transformer real el espectro de S
        es muy sesgado; un Gram con magnitud + ridge global ciega a las
        direcciones redundantes de BAJA energia (justo las utiles de dormir).
        La colinealidad no depende de la magnitud, asi que se mide sobre
        direcciones unitarias.

        Coste: dos GEMM rank×rank + una inversa rank×rank, solo cada
        check_interval. Trivial para rank<=256.
        """
        r = int(module.S.numel())
        if r < 2:
            return torch.zeros(r)
        U = module.U.detach().float()
        S = module.S.detach().float()
        V = module.V.detach().float()

        # ESCALA-INVARIANTE (fix v2.1): el Gram con magnitud G_ij = S_iS_j
        # (u_i·u_j)(v_i·v_j) tiene G_ii = raw_i^2 con muchos ordenes de
        # magnitud de diferencia en un transformer real (espectro sesgado).
        # Un ridge global lam~mean(diag) lo fija la direccion MAYOR, y
        # ahoga a las direcciones redundantes de baja energia (lev->~0 ->
        # clamp(1) -> coherence 0): justo las que querriamos dormir. La
        # colinealidad NO depende de la magnitud, asi que normalizamos cada
        # termino rango-1 a norma 1 (T_i = sgn(S_i)·u_i/‖u_i‖ ⊗ v_i/‖v_i‖)
        # y trabajamos sobre la matriz de CORRELACION R (R_ii = 1), con un
        # ridge uniforme pequeño. Detecta dependencia lineal del conjunto
        # sea cual sea la energia de cada direccion.
        un = U / U.norm(dim=0, keepdim=True).clamp_min(_EPS)     # (out, r)
        vn = V / V.norm(dim=1, keepdim=True).clamp_min(_EPS)     # (r, in)
        sgn = torch.sign(S)
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        R = (un.t() @ un) * (vn @ vn.t()) * (sgn.unsqueeze(0) * sgn.unsqueeze(1))
        if not torch.isfinite(R).all():
            return torch.zeros(r)

        lam = 1e-6
        try:
            Rinv = torch.linalg.inv(
                R + lam * torch.eye(r, device=R.device, dtype=R.dtype)
            )
        except Exception:
            return torch.zeros(r)
        # R_ii == 1 por construccion -> lev_i = (R+λI)^{-1}_ii.
        lev = torch.diagonal(Rinv).clamp_min(1.0)
        coherence = 1.0 - 1.0 / lev
        coherence[~torch.isfinite(coherence)] = 0.0
        return coherence.clamp(0.0, 1.0).cpu()

    # ---- observacion intra-epoch (barata) ---------------------------------

    @torch.no_grad()
    def observe(
        self,
        model: nn.Module,
        optimizer,
        in_warmup: bool,
        epoch: int,
    ) -> None:
        """Un chequeo: actualizar EMAs/contadores y marcar soft-sleep.

        Llamado por el engine cada elastic_rank_check_interval steps. NO
        cambia shapes ni toca el optimizer (eso es el boundary de epoch).
        """
        self._checks += 1
        cfg = self.cfg
        beta = float(cfg.elastic_rank_ema_beta)
        tau_s = float(cfg.elastic_rank_sleep_spectral_threshold)
        tau_u = float(cfg.elastic_rank_sleep_update_threshold)
        patience = int(cfg.elastic_rank_sleep_patience_checks)
        min_age = int(cfg.elastic_rank_min_age_checks)
        min_per_layer = int(cfg.elastic_rank_min_per_layer)
        use_red = bool(getattr(cfg, "elastic_rank_use_redundancy_signal", False))
        tau_coh = float(getattr(cfg, "elastic_rank_redundancy_threshold", 0.9))
        red_patience = int(
            getattr(cfg, "elastic_rank_redundancy_patience_checks", 3)
        )

        # Gracia global se consume aqui (un chequeo = una unidad de gracia).
        grace_active = self._growth_grace > 0 or self._refactor_grace > 0
        self._growth_grace = max(0, self._growth_grace - 1)
        self._refactor_grace = max(0, self._refactor_grace - 1)
        suppress = grace_active or in_warmup

        for name, module in self._iter_layers(model):
            rank = int(module.S.numel())
            st = self._state_for(name, rank)
            if module.wake_gate.numel() != rank:
                # Buffer desincronizado por mutador externo: no arriesgar.
                continue

            spectral, update = self._scores(module, optimizer)

            st.spectral_ema.mul_(beta).add_(spectral, alpha=1.0 - beta)
            st.update_ema.mul_(beta).add_(update, alpha=1.0 - beta)
            if use_red:
                coh = self._redundancy(module)
                st.redund_ema.mul_(beta).add_(coh, alpha=1.0 - beta)
            st.age += 1
            st.revive_cd.clamp_(min=0)
            st.revive_cd[st.revive_cd > 0] -= 1

            if suppress:
                # Fase de estabilizacion: acumular señal pero NO contar hacia
                # dormir (no confundir 'estabilizando' con 'inutil').
                st.low_counter.zero_()
                st.redund_counter.zero_()
                continue

            # Via v1: contribucion espectral baja Y sin actividad de update
            # (AND: una direccion con update alto sigue aprendiendo).
            dead = (st.spectral_ema < tau_s) & (st.update_ema < tau_u)
            st.low_counter = torch.where(
                dead, st.low_counter + 1,
                torch.zeros_like(st.low_counter),
            )
            spec_cand = st.low_counter >= patience

            # Via v2 (independiente): redundante = casi en el span de las
            # otras. NO se condiciona a actividad baja: una direccion dentro
            # del span del resto no aporta capacidad aunque su update sea alto.
            if use_red:
                redundant = st.redund_ema > tau_coh
                st.redund_counter = torch.where(
                    redundant, st.redund_counter + 1,
                    torch.zeros_like(st.redund_counter),
                )
                red_cand = st.redund_counter >= red_patience
            else:
                red_cand = torch.zeros_like(spec_cand)

            mask = module.sleep_mask
            eligible = (
                (spec_cand | red_cand)
                & (st.age >= min_age)
                & (st.revive_cd == 0)
                & (~mask.cpu())
            )
            cand = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
            if not cand:
                continue

            # Respetar rango activo minimo: nunca dejar la capa con < min.
            awake = int((~mask).sum().item())
            can_sleep = max(0, awake - min_per_layer)
            if can_sleep <= 0:
                continue
            # Dormir primero las mas debiles (menor spectral_ema).
            cand.sort(key=lambda i: float(st.spectral_ema[i]))
            # sleep_mask = "candidata a compactacion en el boundary". El
            # CONGELAR intra-epoch (grad=0) lo decide soft_sleep en
            # mask_gradients, no aqui: con soft_sleep=False la direccion sigue
            # entrenando hasta que se compacta al final del epoch.
            for i in cand[:can_sleep]:
                module.sleep_mask[i] = True

    # ---- mascara de gradientes (soft sleep, cada step) --------------------

    @torch.no_grad()
    def mask_gradients(self, model: nn.Module) -> None:
        """Congelar (grad=0) las direcciones marcadas. Muy barato.

        Solo actua si elastic_rank_soft_sleep=True. Con soft_sleep=False las
        direcciones marcadas siguen entrenando hasta que se compactan en el
        boundary de epoch (la flag pasa a tener efecto funcional real).
        """
        if not bool(getattr(self.cfg, "elastic_rank_soft_sleep", True)):
            return
        for _name, module in self._iter_layers(model):
            mask = module.sleep_mask
            rank = int(module.S.numel())
            if mask.numel() != rank or not bool(mask.any()):
                continue
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if module.U.grad is not None:
                module.U.grad[:, idx] = 0
            if module.S.grad is not None:
                module.S.grad[idx] = 0
            if module.V.grad is not None:
                module.V.grad[idx, :] = 0

    # ---- wake warmup (cada step) ------------------------------------------

    @torch.no_grad()
    def tick_wake(self, model: nn.Module) -> None:
        """Avanzar el mini-warmup de direcciones revividas (gate 0 -> 1)."""
        for name, module in self._iter_layers(model):
            st = self._states.get(name)
            if st is None:
                continue
            rank = int(module.S.numel())
            if st.wake_left.numel() != rank or module.wake_gate.numel() != rank:
                continue
            if not bool((st.wake_left > 0).any()):
                continue
            total = max(1, int(self.cfg.elastic_rank_wake_warmup_steps))
            st.wake_left[st.wake_left > 0] -= 1
            gate = (1.0 - st.wake_left.float() / total).clamp_(0.0, 1.0)
            module.wake_gate.copy_(gate.to(module.wake_gate.device))

    # ---- eventos de gracia -------------------------------------------------

    def note_growth(self) -> None:
        self._growth_grace = max(
            self._growth_grace,
            int(self.cfg.elastic_rank_post_growth_grace_checks),
        )

    def note_refactorize(self) -> None:
        self._refactor_grace = max(
            self._refactor_grace,
            int(self.cfg.elastic_rank_post_refactor_grace_checks),
        )

    @torch.no_grad()
    def reset_after_refactorize(self, model: nn.Module) -> None:
        """Resetear el estado por-capa tras una refactorizacion.

        refactorize_model() reconstruye U/S/V con una SVD FRESCA: el indice i
        pasa a ser una direccion completamente distinta. El estado de scoring
        (_states), sleep_mask y wake_gate quedan indexados a la base ANTIGUA;
        sin resetear, mask_gradients congelaria direcciones equivocadas (corre
        cada step, no lo frena la gracia) y los EMAs mezclarian bases.

        Limpia sleep_mask (-> nada congelado), wake_gate (-> 1, nada apagado)
        y descarta _states (se reinicia limpio y bien dimensionado en el
        siguiente observe). El sleep bank NO se toca: es almacenamiento
        dormido independiente de los factores activos. La gracia
        (note_refactorize) sigue aplicando para que los EMAs nuevos se
        estabilicen antes de volver a dormir.
        """
        for name, module in self._iter_layers(model):
            r = int(module.S.numel())
            dev = module.S.device
            module.sleep_mask = torch.zeros(r, dtype=torch.bool, device=dev)
            module.wake_gate = torch.ones(r, device=dev)
            self._states.pop(name, None)

    def note_rollback(self, plan: "ElasticPlan", epoch: int) -> None:
        """Tras revertir una compactacion (loss-guard): poner cooldown a las
        capas afectadas para no re-proponer compactacion de inmediato."""
        cd = int(getattr(self.cfg,
                          "elastic_rank_loss_guard_cooldown_epochs", 3))
        for name, _lp in plan.changed_layers():
            self._rollback_cd[name] = cd
        self._total_rollbacks += 1

    # ---- planificacion en el boundary de epoch ----------------------------

    @torch.no_grad()
    def build_plan(
        self,
        model: nn.Module,
        requested_rank: int,
        grow_requested: bool,
        epoch: int,
    ) -> ElasticPlan:
        """Construir el plan estructural para este boundary de epoch.

        - Compactar (dormir de verdad) las direcciones soft-dormidas cuando
          una capa acumula >= compact_min_dirs (evita reconstruir el optimizer
          por una sola direccion), respetando min_per_layer.
        - Si el scheduler global crece, rellenar los slots nuevos reviviendo
          primero los mejores sleepers y solo el resto con ruido.
        - Podar sleepers que llevan >= prune_after_epochs dormidos.
        """
        cfg = self.cfg
        compact_min = int(cfg.elastic_rank_compact_min_dirs)
        min_per_layer = int(cfg.elastic_rank_min_per_layer)
        prune_after = int(cfg.elastic_rank_prune_after_epochs)
        revive_cd = int(cfg.elastic_rank_revive_cooldown_checks)
        plan = ElasticPlan()

        # 1. Poda del sleep bank (libera memoria CPU definitivamente).
        for lname, sleepers in list(self.bank.items()):
            survivors = []
            for sd in sleepers:
                if sd.epochs_asleep(epoch) >= prune_after:
                    plan.pruned += 1
                    self._total_pruned += 1
                else:
                    if sd.revive_cooldown > 0:
                        sd.revive_cooldown -= 1
                    survivors.append(sd)
            self.bank[lname] = survivors

        for name, module in self._iter_layers(model):
            rank = int(module.S.numel())
            max_rank = int(getattr(module, "_max_possible_rank", rank))
            mask = module.sleep_mask
            if mask.numel() != rank:
                plan.layers[name] = LayerPlan(name, list(range(rank)), [], [], 0, rank)
                continue
            # Cooldown post-rollback: no proponer cambios en esta capa.
            if self._rollback_cd.get(name, 0) > 0:
                plan.layers[name] = LayerPlan(
                    name, list(range(rank)), [], [], 0, rank)
                continue
            st = self._state_for(name, rank)

            masked = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            sleep_idx: List[int] = []
            if len(masked) >= max(1, compact_min):
                # No bajar de min_per_layer: si hay demasiadas dormidas,
                # compactar solo las mas debiles y dejar el resto en soft.
                max_compact = max(0, rank - min_per_layer)
                masked.sort(key=lambda i: float(st.spectral_ema[i]))
                sleep_idx = sorted(masked[:max_compact])

            sleep_set = set(sleep_idx)
            keep = [i for i in range(rank) if i not in sleep_set]

            # Revive/grow SOLO si el scheduler global pide crecer. (El flag
            # allow_independent_growth esta vetado en config.validate porque
            # rellenaria hasta requested_rank cada epoch, anulando la
            # reduccion; no se referencia aqui a proposito.)
            revive: List[SleepingDirection] = []
            grow = 0
            if grow_requested:
                target = max(min_per_layer, min(int(requested_rank), max_rank))
                slots = max(0, target - len(keep))
                if slots > 0:
                    pool = sorted(
                        self.bank.get(name, []),
                        key=lambda sd: sd.wake_score(epoch, prune_after),
                        reverse=True,
                    )
                    pool = [sd for sd in pool if sd.revive_cooldown == 0]
                    revive = pool[:slots]
                    grow = max(0, slots - len(revive))

            new_rank = len(keep) + len(revive) + grow
            new_rank = max(min_per_layer, min(new_rank, max_rank))
            plan.layers[name] = LayerPlan(
                name=name, keep=keep, sleep=sleep_idx,
                revive=revive, grow=grow, new_rank=new_rank,
            )
            # Reservar los sleepers elegidos (se quitan del bank al aplicar).
            for sd in revive:
                sd.revive_cooldown = revive_cd

        # Consumir 1 epoch del cooldown de rollback al final (un build_plan =
        # un epoch). cooldown_epochs=N bloquea exactamente N epochs.
        if self._rollback_cd:
            self._rollback_cd = {
                n: c - 1 for n, c in self._rollback_cd.items() if c - 1 > 0
            }

        return plan

    # ---- consumo por el engine al aplicar la cirugia ----------------------

    def take_revivals(self, name: str, revive: List[SleepingDirection]) -> None:
        """Quitar del bank los sleepers que el plan va a revivir."""
        if name not in self.bank or not revive:
            return
        chosen = set(id(sd) for sd in revive)
        self.bank[name] = [
            sd for sd in self.bank[name] if id(sd) not in chosen
        ]
        self._total_revived += len(revive)

    def add_sleeper(self, name: str, sd: SleepingDirection) -> None:
        self.bank.setdefault(name, []).append(sd)
        self._total_slept += 1

    def remap_state(
        self,
        name: str,
        plan: LayerPlan,
    ) -> None:
        """Reconstruir _LayerState para el nuevo orden de direcciones.

        Nuevo orden = [keep...] + [revive...] + [grow...]. Las revividas
        arrancan con su EMA previa al sleep, edad 0, cooldown de revive y el
        wake warmup armado. Las nuevas (ruido) arrancan limpias y con gate
        completo (entrenan normal, como el grow legacy).
        """
        cfg = self.cfg
        old = self._states.get(name)
        nr = plan.new_rank
        new = _LayerState.fresh(nr)
        warmup = int(cfg.elastic_rank_wake_warmup_steps)
        revive_cd = int(cfg.elastic_rank_revive_cooldown_checks)

        pos = 0
        if old is not None and old.spectral_ema.numel() >= 1:
            for old_i in plan.keep:
                if pos >= nr:
                    break
                if old_i < old.spectral_ema.numel():
                    new.spectral_ema[pos] = old.spectral_ema[old_i]
                    new.update_ema[pos] = old.update_ema[old_i]
                    new.low_counter[pos] = old.low_counter[old_i]
                    new.age[pos] = old.age[old_i]
                    new.revive_cd[pos] = old.revive_cd[old_i]
                    new.wake_left[pos] = old.wake_left[old_i]
                    new.redund_ema[pos] = old.redund_ema[old_i]
                    new.redund_counter[pos] = old.redund_counter[old_i]
                pos += 1
        else:
            pos = min(len(plan.keep), nr)

        for sd in plan.revive:
            if pos >= nr:
                break
            new.spectral_ema[pos] = float(sd.spectral_at_sleep)
            new.update_ema[pos] = float(sd.update_at_sleep)
            new.low_counter[pos] = 0
            new.age[pos] = 0
            new.revive_cd[pos] = revive_cd
            new.wake_left[pos] = warmup
            pos += 1
        # Las 'grow' restantes quedan con fresh() (ema=1, gate completo).
        self._states[name] = new

    # ---- stats -------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        sleeping = sum(len(v) for v in self.bank.values())
        max_coh = 0.0
        for st in self._states.values():
            if st.redund_ema.numel():
                max_coh = max(max_coh, float(st.redund_ema.max()))
        return {
            "checks": self._checks,
            "sleeping_now": sleeping,
            "total_slept": self._total_slept,
            "total_revived": self._total_revived,
            "total_pruned": self._total_pruned,
            "total_rollbacks": self._total_rollbacks,
            "layers_in_cooldown": len(self._rollback_cd),
            "layers_tracked": len(self._states),
            "redundancy_enabled": bool(
                getattr(self.cfg, "elastic_rank_use_redundancy_signal", False)
            ),
            "max_coherence_ema": round(max_coh, 4),
        }

    # ---- PROBE de loss-sensitivity (modo diagnostico, NO muta topologia) ---

    @staticmethod
    def _spearman(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
        """Correlacion de Spearman (rango). Robusta a no-linealidad; lo que
        importa para decisiones de sleep es el ORDEN, no la escala."""
        n = x.numel()
        if n < 3:
            return None
        xr = x.argsort().argsort().float()
        yr = y.argsort().argsort().float()
        xr = xr - xr.mean()
        yr = yr - yr.mean()
        denom = xr.norm() * yr.norm()
        if float(denom) < 1e-12:
            return 0.0
        return round(float((xr @ yr) / denom), 4)

    @staticmethod
    def _dist(t: torch.Tensor) -> Dict[str, float]:
        t = t.float()
        p10 = float(t.quantile(0.10))
        p90 = float(t.quantile(0.90))
        mean = float(t.mean())
        std = float(t.std(unbiased=False))
        return {
            "median": round(float(t.median()), 6),
            "p10": round(p10, 6),
            "p90": round(p90, 6),
            "max": round(float(t.max()), 6),
            "cv": round(std / max(abs(mean), 1e-12), 4),
            "p90_p10_ratio": round(p90 / max(abs(p10), 1e-12), 3),
        }

    @torch.no_grad()
    def probe_loss_sensitivity(
        self,
        model: nn.Module,
        optimizer,
        loss_eval: Callable[[], float],
        epoch: int,
    ) -> Dict[str, Any]:
        """Mide, por ablacion, cuanto sube la loss al quitar cada direccion.

        Ablacion = poner wake_gate[i]=0 temporalmente (la contribucion de la
        direccion sale del forward por _gated_s) sobre un batch sonda FIJO,
        en eval/no_grad, y restaurar el gate exacto. NO toca optimizer,
        sleep_mask, _states ni shapes: es puramente lectura.

        Reporta por capa la distribucion de rel_delta y su correlacion de
        Spearman contra los proxies (spectral/update/coherence), mas un
        agregado global y un veredicto heuristico (plano vs heterogeneo).
        """
        cfg = self.cfg
        cap = int(cfg.elastic_rank_probe_max_dirs_per_layer)
        eps = 1e-8
        was_training = model.training
        model.eval()
        try:
            loss_base = float(loss_eval())
            layers_out: List[Dict[str, Any]] = []
            all_rel: List[float] = []
            all_spec, all_upd, all_coh = [], [], []

            for name, module in self._iter_layers(model):
                r = int(module.S.numel())
                if r < 2 or module.wake_gate.numel() != r:
                    continue
                if r <= cap:
                    idx = list(range(r))
                else:
                    idx = (torch.linspace(0, r - 1, steps=cap)
                           .round().long().unique().tolist())

                spectral, update = self._scores(module, optimizer)
                coh = self._redundancy(module)

                rel = torch.empty(len(idx))
                for j, i in enumerate(idx):
                    g = float(module.wake_gate[i])
                    module.wake_gate[i] = 0.0
                    try:
                        l = float(loss_eval())
                    finally:
                        module.wake_gate[i] = g          # restaurar EXACTO
                    rel[j] = (l - loss_base) / max(abs(loss_base), eps)

                sp = spectral[idx]
                up = update[idx]
                ch = coh[idx]
                all_rel += rel.tolist()
                all_spec += sp.tolist()
                all_upd += up.tolist()
                all_coh += ch.tolist()

                layers_out.append({
                    "layer": name,
                    "rank": r,
                    "probed": len(idx),
                    "rel_delta": self._dist(rel),
                    "spearman_vs_spectral": self._spearman(rel, sp),
                    "spearman_vs_update": self._spearman(rel, up),
                    "spearman_vs_coherence": self._spearman(rel, ch),
                })
        finally:
            model.train(was_training)

        rel_t = torch.tensor(all_rel) if all_rel else torch.zeros(1)
        g_ratio = self._dist(rel_t)["p90_p10_ratio"]
        g_cv = self._dist(rel_t)["cv"]
        # Heterogeneo SOLO si el efecto absoluto es no trivial (p90 de
        # |rel_delta| > min_effect) Y la dispersion relativa es alta. Sin la
        # condicion de magnitud, CV puro sobre valores ~0 da falsos positivos.
        min_effect = 1e-3
        p90_abs = float(rel_t.abs().quantile(0.90))
        heterogeneous = (p90_abs > min_effect) and (g_ratio > 2.0)
        sp_corr = self._spearman(rel_t, torch.tensor(all_spec or [0.0]))
        up_corr = self._spearman(rel_t, torch.tensor(all_upd or [0.0]))
        ch_corr = self._spearman(rel_t, torch.tensor(all_coh or [0.0]))

        record = {
            "epoch": epoch,
            "loss_base": round(loss_base, 6),
            "n_directions": len(all_rel),
            "global": self._dist(rel_t),
            "global_spearman": {
                "spectral": sp_corr, "update": up_corr, "coherence": ch_corr,
            },
            "verdict": {
                "heterogeneous": bool(heterogeneous),
                "p90_abs_rel_delta": round(p90_abs, 6),
                "min_effect": min_effect,
                "reading": (
                    "HETEROGENEO -> mercado global justificable"
                    if heterogeneous else
                    "PLANO/EFECTO~0 -> ElasticRank solo puede aspirar a "
                    "'no dañar' (rollback es el nucleo correcto)"
                ),
                "proxies_predictive": {
                    k: (v is not None and abs(v) >= 0.5)
                    for k, v in (("spectral", sp_corr),
                                 ("update", up_corr),
                                 ("coherence", ch_corr))
                },
            },
            "layers": layers_out,
        }
        self._probe_history.append(record)
        try:
            out = Path(cfg.elastic_rank_probe_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self._probe_history, indent=2))
        except Exception as e:  # pragma: no cover - IO defensivo
            logger.warning(f"[ElasticRank probe] no se pudo escribir JSON: {e}")
        logger.info(
            f"[ElasticRank probe] epoch {epoch}: {record['verdict']['reading']} "
            f"(p90/p10={record['global']['p90_p10_ratio']}, "
            f"cv={record['global']['cv']}, "
            f"corr spectral/update/coh="
            f"{sp_corr}/{up_corr}/{ch_corr})"
        )
        return record
