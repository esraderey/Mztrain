"""
MZTrain - Scheduler de rango progresivo.

Controla el crecimiento del rango de factorizacion durante el
entrenamiento, permitiendo empezar con modelos pequenos/rapidos
y crecer progresivamente hacia mayor capacidad.

Incluye:
- ZRankScheduler: schedules fijos (linear, exponential, cosine, adaptive)
- ZSpectralRankScheduler: crece basandose en analisis espectral de pesos (T4)

Referencias:
- Dynamic Rank Adjustment: Oct 2025 (arXiv:2508.08625v3)
- AdaLoRA: ICLR 2023 (arXiv:2303.10512)
- DR-LoRA: Enero 2026 (arXiv:2601.04823)
"""

import math
import logging
import random
from typing import List, Optional

logger = logging.getLogger("mztrain")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import RankSchedule


class ZRankScheduler:
    """Controla el crecimiento progresivo de rango durante entrenamiento.

    Empieza con rango bajo (rapido, baja memoria) y crece progresivamente
    hasta el rango maximo, dando al modelo mas capacidad conforme converge.

    Analogia: como progressive resizing en vision (empezar 64x64, crecer
    a 512x512) pero para la dimension de factorizacion.

    Args:
        initial_rank: Rango inicial.
        max_rank: Rango maximo.
        total_epochs: Total de epochs planificados.
        schedule: Tipo de schedule de crecimiento.
        growth_interval: Intervalo de epochs para evaluar crecimiento.
        growth_factor: Factor multiplicativo de crecimiento.
        adaptive_threshold: Umbral de mejora para crecimiento adaptativo.

    Example:
        >>> scheduler = ZRankScheduler(32, 256, total_epochs=100)
        >>> for epoch in range(100):
        ...     rank = scheduler.get_rank(epoch)
        ...     if rank > scheduler.current_rank:
        ...         model.grow_rank(rank)
    """

    def __init__(
        self,
        initial_rank: int,
        max_rank: int,
        total_epochs: int,
        schedule: RankSchedule = RankSchedule.EXPONENTIAL,
        growth_interval: int = 10,
        growth_factor: float = 2.0,
        adaptive_threshold: float = 0.01,
    ):
        self.initial_rank = initial_rank
        self.max_rank = max_rank
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.growth_interval = growth_interval
        self.growth_factor = growth_factor
        self.adaptive_threshold = adaptive_threshold
        self.current_rank = initial_rank
        self._loss_history: List[float] = []

    def get_rank(self, epoch: int, current_loss: Optional[float] = None) -> int:
        """Obtener rango para epoch dado.

        Args:
            epoch: Epoch actual.
            current_loss: Loss actual (requerido para schedule ADAPTIVE).

        Returns:
            Rango recomendado para este epoch.
        """
        if self.schedule == RankSchedule.CONSTANT:
            return self.initial_rank

        elif self.schedule == RankSchedule.LINEAR:
            # progress llega a 1.0 en la ultima epoch (epoch == total_epochs-1)
            # y se satura ahi, asi el rango alcanza max_rank dentro del plan.
            progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
            rank = int(
                self.initial_rank + progress * (self.max_rank - self.initial_rank)
            )

        elif self.schedule == RankSchedule.EXPONENTIAL:
            doublings = epoch // max(self.growth_interval, 1)
            rank = int(self.initial_rank * (self.growth_factor ** doublings))

        elif self.schedule == RankSchedule.COSINE:
            progress = min(epoch / max(self.total_epochs - 1, 1), 1.0)
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            rank = int(
                self.initial_rank + cosine_factor * (self.max_rank - self.initial_rank)
            )

        elif self.schedule == RankSchedule.ADAPTIVE:
            rank = self._adaptive_rank(epoch, current_loss)

        else:
            rank = self.initial_rank

        rank = max(self.initial_rank, min(rank, self.max_rank))
        # NOTE: do NOT mutate self.current_rank here. The engine compares the
        # returned rank against current_rank and only updates it after actually
        # calling _grow_model_rank(). Mutating here would short-circuit that
        # check and prevent any layer growth from happening.
        return rank

    def _adaptive_rank(self, epoch: int, current_loss: Optional[float]) -> int:
        """Crecer rango adaptativamente basado en convergencia del loss."""
        if current_loss is not None:
            self._loss_history.append(current_loss)

        if len(self._loss_history) < self.growth_interval:
            return self.current_rank

        recent = self._loss_history[-self.growth_interval:]
        improvement = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-8)

        if improvement < self.adaptive_threshold:
            new_rank = int(self.current_rank * self.growth_factor)
            new_rank = min(new_rank, self.max_rank)
            if new_rank > self.current_rank:
                logger.info(
                    f"[ZRankScheduler] Loss estancado (mejora {improvement:.4f}), "
                    f"creciendo rango: {self.current_rank} -> {new_rank}"
                )
            return new_rank

        return self.current_rank

    def should_grow(self, epoch: int) -> bool:
        """Verificar si el rango deberia crecer en este epoch.

        Args:
            epoch: Epoch actual.

        Returns:
            True si el rango deberia incrementarse.
        """
        if self.schedule == RankSchedule.CONSTANT:
            return False
        new_rank = self.get_rank(epoch)
        return new_rank > self.current_rank


class ZSpectralRankScheduler:
    """Scheduler de rango basado en analisis espectral de pesos (T4).

    En lugar de seguir un schedule fijo, analiza la distribucion de valores
    singulares de los pesos para decidir si el rango actual es suficiente.

    Algoritmo:
        CADA growth_interval epochs:
        1. Muestrear K capas factorizadas
        2. Sobre los r valores singulares ALMACENADOS (S del factor) calcular un
           proxy de "planitud" del espectro: ratio = 1 - sigma_min/sigma_max
           (0 = plano, 1 = empinado). AVISO: es un proxy de numero de condicion
           sobre el rango actual, funcion SOLO de los dos extremos; NO es la
           energia acumulada E(r)=sum(s_i^2)/sum(s_j^2) ni un SVD del peso
           reconstruido. Es sensible a un unico sigma_min pequeno (un outlier
           puede hacer que un espectro plano lea ~1).
        3. SI ratio < energy_threshold (plano) -> CRECER
        4. SI ratio > energy_ceiling (empinado) Y loss estancado -> NO crecer, LR
        5. Zona intermedia -> usar growth_factor como fallback

    Interfaz compatible con ZRankScheduler: get_rank(epoch, current_loss).

    Args:
        model: Modelo con capas factorizadas (para analisis espectral).
        initial_rank: Rango inicial.
        max_rank: Rango maximo.
        total_epochs: Total de epochs planificados.
        energy_threshold: Umbral bajo de energia (< esto = crecer).
        energy_ceiling: Umbral alto de energia (> esto = no crecer).
        sample_layers: Numero de capas a muestrear por evaluacion.
        growth_interval: Cada N epochs, evaluar.
        growth_factor: Factor multiplicativo si se decide crecer.
        factorized_cls: Clase(s) de capas factorizadas a analizar.

    Example:
        >>> scheduler = ZSpectralRankScheduler(
        ...     model, initial_rank=32, max_rank=256, total_epochs=100
        ... )
        >>> rank = scheduler.get_rank(epoch=10, current_loss=0.5)
    """

    def __init__(
        self,
        model: nn.Module,
        initial_rank: int,
        max_rank: int,
        total_epochs: int,
        energy_threshold: float = 0.90,
        energy_ceiling: float = 0.99,
        sample_layers: int = 4,
        growth_interval: int = 10,
        growth_factor: float = 2.0,
        factorized_cls=None,
    ):
        self.model = model
        self.initial_rank = initial_rank
        self.max_rank = max_rank
        self.total_epochs = total_epochs
        self.energy_threshold = energy_threshold
        self.energy_ceiling = energy_ceiling
        self.sample_layers = sample_layers
        self.growth_interval = growth_interval
        self.growth_factor = growth_factor
        self.current_rank = initial_rank
        self._loss_history: List[float] = []

        # Importar aqui para evitar circular
        if factorized_cls is None:
            from .layers import ZFactorizedLinear
            self._factorized_cls = ZFactorizedLinear
        else:
            self._factorized_cls = factorized_cls

    def get_rank(self, epoch: int, current_loss: Optional[float] = None) -> int:
        """Obtener rango para epoch dado, basado en analisis espectral.

        Args:
            epoch: Epoch actual.
            current_loss: Loss actual (para detectar estancamiento).

        Returns:
            Rango recomendado para este epoch.
        """
        if current_loss is not None:
            self._loss_history.append(current_loss)

        # Solo evaluar en intervalos de crecimiento
        if epoch == 0 or epoch % self.growth_interval != 0:
            return self.current_rank

        # Analizar espectro
        energy_ratio = self._compute_spectral_energy()
        if energy_ratio is None:
            return self.current_rank

        loss_stalled = self._is_loss_stalled()

        if energy_ratio < self.energy_threshold:
            # Rango insuficiente — CRECER
            new_rank = int(self.current_rank * self.growth_factor)
            new_rank = min(new_rank, self.max_rank)
            if new_rank > self.current_rank:
                logger.info(
                    f"[ZSpectralScheduler] Energia {energy_ratio:.3f} < threshold "
                    f"{self.energy_threshold}, creciendo: {self.current_rank} -> {new_rank}"
                )
            # No mutar current_rank aqui. El engine compara el valor retornado
            # contra current_rank y solo lo actualiza despues de crecer capas.
            return new_rank

        elif energy_ratio > self.energy_ceiling and loss_stalled:
            # Rango suficiente y loss estancado — NO crecer
            logger.info(
                f"[ZSpectralScheduler] Energia {energy_ratio:.3f} > ceiling "
                f"{self.energy_ceiling} y loss estancado — no crecer"
            )
            return self.current_rank

        else:
            # Zona intermedia — fallback conservador
            # Solo crecer si llevamos suficientes epochs sin crecimiento
            return self.current_rank

    @torch.no_grad()
    def _compute_spectral_energy(self) -> Optional[float]:
        """Proxy de suficiencia de rango: promedio de (1 - sigma_min/sigma_max)
        sobre los valores singulares ALMACENADOS de las capas muestreadas.

        NO es la energia acumulada E(r)=sum(s_i^2)/sum(s_j^2) (que sobre los r
        valores retenidos daria 1 trivialmente): es un proxy de numero de
        condicion, funcion solo de los dos extremos del espectro. Heuristico.

        Returns:
            Proxy promedio en [0,1), o None si no hay capas factorizadas.
        """
        # Recoger capas factorizadas
        factorized_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, self._factorized_cls):
                factorized_layers.append((name, module))

        if not factorized_layers:
            return None

        # Muestrear K capas
        k = min(self.sample_layers, len(factorized_layers))
        sampled = random.sample(factorized_layers, k)

        energy_ratios = []
        for name, module in sampled:
            r = module.rank

            # Para capas factorizadas, W = U@diag(S)@V es rank-r exacto.
            # La energy ratio clasica (top-r sigma^2 / total sigma^2) seria 1.0
            # trivialmente. En su lugar, usamos la TASA DE DECAIMIENTO ESPECTRAL
            # de los valores singulares almacenados como proxy:
            #
            #   energy_ratio = 1 - (S[r-1] / S[0])
            #
            # - Espectro EMPINADO (S[r-1] << S[0]): energy alta -> rank suficiente
            # - Espectro PLANO   (S[r-1] ~ S[0]):  energy baja -> rank insuficiente
            #   (todos los componentes igual de importantes = probablemente hay mas
            #    componentes significativos mas alla del rank actual)
            S = module.S.detach().float().abs()
            if not torch.isfinite(S).all() or S.numel() < 2:
                continue

            try:
                s_max = float(S.max())
                s_min = float(S.min())

                if s_max < 1e-12:
                    continue

                decay = s_min / s_max  # 0 to 1
                ratio = 1.0 - decay    # 1 = steep (sufficient), 0 = flat (insufficient)

                energy_ratios.append(ratio)

                logger.debug(
                    f"[ZSpectralScheduler] {name}: E({r})={ratio:.4f} "
                    f"(S_max={s_max:.4f}, S_min={s_min:.4f}, decay={decay:.4f})"
                )
            except Exception as e:
                logger.warning(f"[ZSpectralScheduler] SVD failed for {name}: {e}")
                continue

        if not energy_ratios:
            return None

        avg_energy = sum(energy_ratios) / len(energy_ratios)
        logger.debug(
            f"[ZSpectralScheduler] Average energy ratio: {avg_energy:.4f} "
            f"(from {len(energy_ratios)} layers)"
        )
        return avg_energy

    def _is_loss_stalled(self, window: int = None) -> bool:
        """Detectar si el loss esta estancado.

        Args:
            window: Ventana de epochs para evaluar. Default: growth_interval.

        Returns:
            True si la mejora relativa es < 1%.
        """
        window = window or self.growth_interval
        if len(self._loss_history) < window:
            return False

        recent = self._loss_history[-window:]
        improvement = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-8)
        return improvement < 0.01
