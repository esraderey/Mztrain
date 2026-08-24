"""Guarda de convergencia: impide comparar BPC finales de corridas que no convergieron.

Origen (cribado del 2026-08-22): se compararon los BPC finales de T14 al paso 6000 y
se sacaron tres conclusiones —"el modelo chico gana al grande", "la tasa optima cambia
de signo con la escala", "5.1x parametros valen 0.130"— sin comprobar que NINGUNA de
las corridas habia convergido: todas seguian cayendo con pendientes de +0.016 a +0.063
BPC por cada 500 pasos. Medido despues, las tres afirmaciones cambian con el corte:

    paso    G_param   G_lr(7.6M)   G_lr(38.8M)   quien gana
    2500    +0.3631   +0.5715      +0.3116       -
    3000    +0.3336   +0.5057      +0.1964       el GRANDE
    4000    +0.2474   +0.3460      +0.0195       el chico
    4500    +0.2112   +0.2775      -0.0035       el chico
    6000    +0.1301   +0.1668      -0.0687       el chico

Comparar BPC finales sin convergencia no mide calidad alcanzable: mide quien va
delante en un corte arbitrario. Este modulo pone la comprobacion donde se cometio el
error — en el momento de comparar — y obliga a declarar explicitamente cuando la
comparacion es "a presupuesto fijo de N pasos", que es una afirmacion legitima y
distinta.

Umbral por defecto 0.005 BPC por cada 500 pasos: un orden de magnitud por debajo del
efecto mas pequeno que el arco ha reportado (0.017).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

UMBRAL_POR_500 = 0.005


def pendiente_final(curva: Sequence[Sequence[float]], por_pasos: int = 500) -> float:
    """Caida de BPC por cada `por_pasos` pasos, medida sobre los dos ultimos tramos.

    `curva` es la lista [paso, reloj, bpc] que producen los bancos del arco. Positivo
    = todavia mejorando. Devuelve 0.0 si no hay puntos suficientes.
    """
    if len(curva) < 3:
        return 0.0
    (p0, _, b0), (p1, _, b1) = curva[-3], curva[-1]
    if p1 <= p0:
        return 0.0
    return (b0 - b1) / (p1 - p0) * por_pasos


def convergido(curva: Sequence[Sequence[float]], umbral: float = UMBRAL_POR_500) -> bool:
    """True si la curva ya no mejora de forma apreciable."""
    return abs(pendiente_final(curva)) < umbral


class NoConvergido(RuntimeError):
    """Se intento comparar BPC finales de corridas que siguen mejorando."""


def comparar_final(curvas: Dict[str, Sequence[Sequence[float]]],
                   umbral: float = UMBRAL_POR_500,
                   presupuesto_fijo: bool = False) -> Dict[str, object]:
    """Compara los BPC finales de varias corridas, con la guarda puesta.

    Args:
        curvas: {nombre: curva}.
        umbral: pendiente por debajo de la cual se considera convergida.
        presupuesto_fijo: si True, la comparacion se permite aunque nadie haya
            convergido, y el resultado queda ETIQUETADO como "a presupuesto fijo de N
            pasos" — que es una afirmacion legitima sobre velocidad, no sobre calidad
            alcanzable.

    Raises:
        NoConvergido: si alguna corrida sigue mejorando y no se declaro
            `presupuesto_fijo=True`. El mensaje nombra las corridas y sus pendientes.
    """
    pendientes = {n: pendiente_final(c) for n, c in curvas.items()}
    sin_converger = {n: p for n, p in pendientes.items() if abs(p) >= umbral}
    if sin_converger and not presupuesto_fijo:
        detalle = ", ".join(f"{n} ({p:+.4f}/500 pasos)"
                            for n, p in sorted(sin_converger.items()))
        raise NoConvergido(
            f"{len(sin_converger)} de {len(curvas)} corridas siguen mejorando: "
            f"{detalle}. Comparar sus BPC finales mide quien va delante en el corte, "
            f"no calidad alcanzable. Entrena hasta convergencia, o pasa "
            f"presupuesto_fijo=True para etiquetar la comparacion como 'a presupuesto "
            f"fijo de N pasos'.")
    pasos = {n: c[-1][0] for n, c in curvas.items()}
    finales = {n: c[-1][2] for n, c in curvas.items()}
    mejor = min(finales, key=finales.get)
    return {
        "finales": finales,
        "pendientes": pendientes,
        "mejor": mejor,
        "pasos": pasos,
        "convergido": not sin_converger,
        "etiqueta": ("calidad final (todas convergidas)" if not sin_converger
                     else f"a presupuesto fijo de {max(pasos.values())} pasos "
                          f"(NINGUNA convergida)"),
        "sin_converger": sorted(sin_converger),
    }


def tabla(curvas: Dict[str, Sequence[Sequence[float]]],
          umbral: float = UMBRAL_POR_500) -> List[Tuple[str, int, float, float, bool]]:
    """(nombre, paso final, bpc final, pendiente, convergido) para inspeccion."""
    return [(n, c[-1][0], c[-1][2], pendiente_final(c), convergido(c, umbral))
            for n, c in sorted(curvas.items())]
