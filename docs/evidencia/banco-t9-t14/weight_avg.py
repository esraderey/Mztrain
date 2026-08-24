"""Promedio de pesos para ElasticShape: SWA canonico, acumulado EN PARALELO.

Evidencia que fija el diseno (T13, 2026-08-22, `T13-VEREDICTO.md`):

  - Promediar los ultimos checkpoints MEJORA el modelo: 6/6 semillas, -0.0175 BPC
    de media, con coste de computo nulo.
  - **SUSTITUIR** los pesos de entrenamiento por el promedio a mitad de camino es
    CONTRAPRODUCENTE: muerte por regla preregistrada (2/6 semillas mejoran,
    ganancia -2.58%). Diagnostico: el promedio no CREA progreso, lo ADELANTA — deja
    al modelo donde la trayectoria iba a llegar sola en ~250 pasos, en terreno mas
    llano; ademas el gradiente cae un 44% en ese punto (dano transitorio, se iguala
    hacia el paso 50, pero el progreso perdido no se recupera).
  - La version HACIA ADELANTE (extrapolar la trayectoria, theta + a*(theta_t -
    theta_{t-K})) esta FALSADA en tres regimenes: empeora monotonamente con a.

Por eso este modulo implementa **la unica variante que la evidencia no refuta**:
acumular el promedio en paralelo, sin tocar jamas los pesos de entrenamiento, y
usarlo solo para evaluar o desplegar.

**Condicion de uso descubierta al probarlo (medida, ventanas de 3 snapshots cada
50 pasos, tras una cirugia 192->384):**

    pasos tras el growth    ultimo    promedio    efecto
    0-150                   3.0408    3.0809      +0.0401  EMPEORA
    150-300                 2.9636    2.9492      -0.0144  mejora
    300-450                 2.8404    2.8338      -0.0066  mejora
    600-750                 2.6015    2.5553      -0.0462  mejora
    1200-1350               2.3511    2.2610      -0.0901  mejora

El promedio ayuda cuando la trayectoria **oscila alrededor de un valle**, y hace
dano mientras esta en **descenso rapido y sistematico**: ahi la media arrastra
hacia atras, hacia pesos claramente peores.

La distancia al growth resulto ser solo un proxy. **El predictor real es la
velocidad de descenso local** (medido sobre ventanas de 3 snapshots cada 50 pasos):

    caida de BPC por 50 pasos     delta del promedio    ayuda?
    0.0440                        +0.0338               no
    0.0149                        +0.0028               no
    -0.0217                       -0.0895               si
    -0.0455                       -0.1709               si
    0.0031 .. 0.0059              -0.017 .. -0.029      si

Frontera practica: **~0.015 BPC por ventana**. Por encima, no promediar; por debajo,
el promedio paga y a veces mucho. `umbral_descenso` codifica esa regla usando la
metrica que el caller ya calcula.

API opt-in: nada de esto se activa salvo llamada explicita del caller.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torch.nn as nn


class WeightAverager:
    """Promedio de pesos en paralelo. NO toca los parametros del modelo.

    Args:
        model: modelo del que se toma la forma inicial.
        decay: None (por defecto) = media aritmetica acumulada (SWA); un valor en
            (0, 1) = media exponencial (EMA), util cuando interesa olvidar el
            pasado lejano. La sonda de T13 midio que incluir puntos demasiado
            viejos EMPEORA (la media de 6 snapshots ya era peor que la de 2), asi
            que el olvido no es un capricho: es la forma de no arrastrar posiciones
            de cuando el valle estaba en otro sitio.
        device: donde vive el promedio. "cpu" lo saca de la VRAM (a 118M params son
            ~475 MB liberados) a cambio de una transferencia por update.
        omitir_tras_reset: cuantas llamadas a update() se ignoran justo despues de
            un reset, para cubrir el transitorio en que promediar hace dano (ver la
            tabla de arriba). **Por defecto 0: el guard es OPT-IN.** Un update() que
            no hace nada en silencio es justo el modo de fallo que este proyecto
            trata como grave — la primera version puso 4 por defecto y rompio seis
            pruebas del propio banco al instante, dejandolas creyendo que
            promediaban. Con updates cada 50 pasos, `omitir_tras_reset=4` cubre los
            200 de la rampa del controller; pero se pide explicitamente.

    Los tensores NO flotantes (mascaras bool, indices, contadores) no se promedian:
    se conserva el ultimo valor visto. Promediar un bool lo destruiria.
    """

    def __init__(self, model: nn.Module, decay: Optional[float] = None,
                 device: Optional[str] = None, omitir_tras_reset: int = 0,
                 umbral_descenso: Optional[float] = None):
        if decay is not None and not 0.0 < decay < 1.0:
            raise ValueError("decay debe estar en (0, 1), o ser None para media aritmetica")
        if omitir_tras_reset < 0:
            raise ValueError("omitir_tras_reset debe ser >= 0")
        self.decay = decay
        self.device = device
        self.omitir_tras_reset = omitir_tras_reset
        self.umbral_descenso = umbral_descenso
        self.omitidas = 0
        self._metrica_prev: Optional[float] = None
        self._avg: Dict[str, torch.Tensor] = {}
        self._otros: Dict[str, torch.Tensor] = {}
        self.n = 0
        self.reset(model)

    # ---------------------------------------------------------------- interno
    def _firma(self, sd: Dict[str, torch.Tensor]):
        return {k: tuple(v.shape) for k, v in sd.items()}

    def _guardar(self, sd: Dict[str, torch.Tensor]) -> None:
        self._avg, self._otros = {}, {}
        for k, v in sd.items():
            t = v.detach().clone()
            if self.device is not None:
                t = t.to(self.device)
            (self._avg if v.dtype.is_floating_point else self._otros)[k] = t

    # ---------------------------------------------------------------- publico
    def reset(self, model: nn.Module) -> None:
        """Re-siembra el promedio con los pesos actuales. OBLIGATORIO tras un
        growth: al cambiar la forma, el promedio acumulado deja de ser conmensurable
        con el modelo y mezclarlos no significa nada."""
        self._guardar(model.state_dict())
        self.n = 1
        self._por_omitir = self.omitir_tras_reset
        self._metrica_prev = None

    @torch.no_grad()
    def update(self, model: nn.Module, metrica: Optional[float] = None) -> bool:
        """Incorpora los pesos actuales al promedio. No modifica el modelo.

        Devuelve False si la llamada se omitio por caer dentro del transitorio
        posterior a un reset (ver `omitir_tras_reset`); True si se incorporo.
        Durante la omision se RE-SIEMBRA el promedio con los pesos actuales, para
        que la ventana empiece limpia cuando termine el transitorio.
        """
        sd = model.state_dict()
        if self._firma(sd) != self._firma({**self._avg, **self._otros}):
            raise RuntimeError(
                "la forma del modelo cambio desde el ultimo reset del promedio "
                "(tipicamente por un growth de ElasticShape): el promedio acumulado "
                "no es conmensurable con el modelo actual. Llama a reset(model) "
                "despues de cada apply_event para empezar un promedio nuevo.")
        if self.umbral_descenso is not None and metrica is not None:
            if (self._metrica_prev is not None
                    and self._metrica_prev - metrica > self.umbral_descenso):
                # todavia en descenso rapido: promediar aqui haria dano
                self._metrica_prev = metrica
                self.omitidas += 1
                self._guardar(sd)
                self.n = 1
                return False
            self._metrica_prev = metrica
        if self._por_omitir > 0:
            self._por_omitir -= 1
            self.omitidas += 1
            self._guardar(sd)          # re-siembra: la ventana arranca limpia
            self.n = 1
            return False
        self.n += 1
        for k, v in sd.items():
            if not v.dtype.is_floating_point:
                self._otros[k].copy_(v)
                continue
            w = v.detach().to(self._avg[k].device, dtype=self._avg[k].dtype)
            if self.decay is None:
                self._avg[k].add_((w - self._avg[k]) / self.n)     # media incremental
            else:
                self._avg[k].mul_(self.decay).add_(w, alpha=1.0 - self.decay)
        return True

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """El modelo promediado, listo para load_state_dict o para guardar."""
        return {**{k: v.clone() for k, v in self._avg.items()},
                **{k: v.clone() for k, v in self._otros.items()}}

    @contextmanager
    def evaluado_en(self, model: nn.Module):
        """Carga el promedio en el modelo, cede el control, y **restaura los pesos
        originales bit a bit** al salir (tambien si el bloque lanza excepcion).

        Uso previsto:
            with avg.evaluado_en(model):
                bpc = evaluar(model)
        """
        respaldo = {k: v.detach().clone() for k, v in model.state_dict().items()}
        prom = {k: v.to(next(model.parameters()).device)
                for k, v in self.state_dict().items()}
        try:
            model.load_state_dict(prom)
            yield model
        finally:
            model.load_state_dict(respaldo)

    def __repr__(self) -> str:
        modo = "SWA" if self.decay is None else f"EMA(decay={self.decay})"
        return (f"WeightAverager({modo}, n={self.n}, tensores={len(self._avg)}, "
                f"omitidas={self.omitidas})")
