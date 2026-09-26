"""
MZTrain - VRAM Governor (MVP).

Alcance DELIBERADAMENTE acotado. La propuesta completa (precision autopilot,
checkpoint controller, rank-budget en bytes, scoring multiobjetivo,
acoplamiento con ElasticRank) NO se implementa porque:
  - la capa de sensores CUDA no es validable sin GPU (este entorno es CPU),
  - el rank-budget heredaria la debilidad probada de ElasticRank
    (Spearman(rel_loss, proxies) ~= 0.13: las señales no predicen utilidad),
  - el "loss guard" de esa propuesta ya existe (rollback por perdida).

Lo que SI hace el MVP, que es incondicionalmente correcto y testeable sin
GPU (la logica es pura dado un lector de memoria inyectable):

  observa memoria -> presion (EMA) -> modo (con histeresis)
                  -> BLOQUEA el crecimiento de rango si hay presion
                  -> recupera de un OOM transitorio (retry una vez)

Esto gobierna la feature ORIGINAL que si funciona y si puede causar OOM
(crecimiento de rango), sin depender de la redundancia de ElasticRank.

Roadmap (no implementado; requiere validacion en GPU real): acciones
estructurales agresivas, control de precision/checkpointing/compresion,
presupuesto de rango cuantitativo, modos QUALITY/DEFENSIVE.
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("mztrain")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.vram_governor. "
        "Instalar con: pip install torch>=2.0.0"
    )


@dataclass
class ZVRAMSnapshot:
    """Lectura instantanea de memoria + presion derivada."""
    allocated: int
    reserved: int
    peak: int
    free: int
    total: int
    pressure: float          # max(reserved/total, peak/total)
    pressure_ema: float
    mode: str


class ZVRAMGovernor:
    """Gobernador de presupuesto de VRAM (MVP).

    Si no hay CUDA (o no se inyecta un lector), es INERTE: observe() devuelve
    None y no se bloquea nada. Esto es lo correcto en CPU y mantiene el
    comportamiento del resto del codebase (todo CUDA esta guardado).

    Args:
        config: ZTrainConfig con los campos vram_*.
    """

    def __init__(self, config: Any):
        self.cfg = config
        self.pressure_ema: Optional[float] = None
        self.mode: str = "normal"
        self.block_growth: bool = False
        # Histeresis: no cambiar de modo por un spike aislado.
        self._pending_band: Optional[str] = None
        self._pending_count: int = 0
        # Stats / observabilidad.
        self._mode_history: List[str] = []
        self._pressure_history: List[float] = []
        self._counters: Dict[str, int] = {
            "checks": 0,
            "blocked_grows": 0,
            "allowed_grows": 0,
            "oom_events": 0,
            "oom_recoveries": 0,
        }

    # ---- sensores ---------------------------------------------------------

    def _read_memory(
        self,
        device: Optional["torch.device"],
        mem_reader: Optional[Callable[[], Dict[str, int]]] = None,
    ) -> Optional[Dict[str, int]]:
        """Lectura cruda de memoria. `mem_reader` (tests / mocks) tiene
        prioridad; si no, usa torch.cuda; si no hay CUDA -> None (inerte)."""
        if mem_reader is not None:
            return mem_reader()
        if not torch.cuda.is_available():
            return None
        try:
            free, total = torch.cuda.mem_get_info(device)
            return {
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
                "peak": int(torch.cuda.max_memory_allocated(device)),
                "free": int(free),
                "total": int(total),
            }
        except Exception as e:  # pragma: no cover - defensivo (sin GPU aqui)
            logger.debug(f"[VRAM Governor] lectura de memoria fallo: {e}")
            return None

    # ---- observacion + decision ------------------------------------------

    def observe(
        self,
        device: Optional["torch.device"] = None,
        mem_reader: Optional[Callable[[], Dict[str, int]]] = None,
    ) -> Optional[ZVRAMSnapshot]:
        """Medir, actualizar presion (EMA) y decidir modo. Devuelve el
        snapshot o None si no hay informacion de memoria (inerte)."""
        raw = self._read_memory(device, mem_reader)
        if raw is None:
            return None

        total = max(int(raw["total"]), 1)
        pressure = max(raw["reserved"] / total, raw["peak"] / total)

        beta = float(self.cfg.vram_governor_ema_beta)
        if self.pressure_ema is None:
            self.pressure_ema = pressure
        else:
            self.pressure_ema = beta * self.pressure_ema + (1 - beta) * pressure

        self._counters["checks"] += 1
        self._decide(self.pressure_ema)

        snap = ZVRAMSnapshot(
            allocated=raw["allocated"], reserved=raw["reserved"],
            peak=raw["peak"], free=raw["free"], total=total,
            pressure=pressure, pressure_ema=self.pressure_ema, mode=self.mode,
        )
        self._mode_history.append(self.mode)
        self._pressure_history.append(round(self.pressure_ema, 4))
        # Acotar historiales (observabilidad, no auditoria completa).
        if len(self._mode_history) > 512:
            self._mode_history = self._mode_history[-512:]
            self._pressure_history = self._pressure_history[-512:]
        return snap

    def _band(self, p: float) -> str:
        if p >= float(self.cfg.vram_emergency_threshold):
            return "emergency"
        if p >= float(self.cfg.vram_preventive_threshold):
            return "preventive"
        return "normal"

    def _decide(self, pressure_ema: float) -> None:
        """Transicion de modo con histeresis; fija block_growth."""
        band = self._band(pressure_ema)
        if band == self.mode:
            self._pending_band = None
            self._pending_count = 0
        else:
            if band == self._pending_band:
                self._pending_count += 1
            else:
                self._pending_band = band
                self._pending_count = 1
            if self._pending_count >= int(self.cfg.vram_hysteresis_checks):
                self.mode = band
                self._pending_band = None
                self._pending_count = 0
        # En MVP: tanto preventive como emergency bloquean crecimiento.
        # Las acciones estructurales agresivas de emergency son roadmap.
        self.block_growth = self.mode in ("preventive", "emergency")

    # ---- gating del crecimiento de rango ---------------------------------

    def approve_rank_growth(self, requested_rank: int,
                            current_rank: int) -> int:
        """Devuelve el rango aprobado. Si hay presion, no se crece."""
        if requested_rank <= current_rank:
            return requested_rank
        if self.block_growth:
            self._counters["blocked_grows"] += 1
            pe = ("n/a" if self.pressure_ema is None
                  else f"{self.pressure_ema:.3f}")
            logger.info(
                f"[MZTrain VRAM] mode={self.mode} pressure_ema={pe}: "
                f"crecimiento de rango {current_rank}->{requested_rank} "
                f"BLOQUEADO por presion"
            )
            return current_rank
        self._counters["allowed_grows"] += 1
        return requested_rank

    # ---- recuperacion de OOM ---------------------------------------------

    def oom_guarded(self, fn: Callable[[], Any]) -> Any:
        """Ejecuta fn(); ante OutOfMemoryError vacia cache y reintenta UNA
        vez (si vram_oom_retry). Si vuelve a fallar, re-lanza."""
        try:
            return fn()
        except torch.cuda.OutOfMemoryError as e:
            self._counters["oom_events"] += 1
            logger.warning("[MZTrain VRAM] OOM capturado")
            # Guardar la excepcion solo si NO vamos a reintentar (para re-lanzarla
            # con su traza). Si reintentamos, dejamos que Python borre 'e' al salir
            # del handler: su __traceback__ retiene los frames del step fallido (y
            # con ellos sus tensores), asi que empty_cache() no los liberaria si
            # limpiaramos aqui dentro.
            saved = None if bool(self.cfg.vram_oom_retry) else e
        # Fuera del handler: 'e' ya fue liberado, el traceback no retiene tensores.
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass
        if saved is not None:
            raise saved
        result = fn()  # reintento unico; si vuelve a fallar, propaga
        self._counters["oom_recoveries"] += 1
        logger.info("[MZTrain VRAM] recuperado tras OOM (retry)")
        return result

    # ---- stats ------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "pressure_ema": (round(self.pressure_ema, 4)
                             if self.pressure_ema is not None else None),
            "mode_history": list(self._mode_history),
            "pressure_history": list(self._pressure_history),
            "actions": dict(self._counters),
            "scope": "mvp: growth-gating + oom-retry; "
                     "precision/checkpoint/rank-budget = roadmap",
        }
