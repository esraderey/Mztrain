"""Anclas de las dos correcciones del cribado del 2026-08-22.

P1 - metrica de evaluacion duplicada y divergente (t14 topaba en 4000 ventanas frente
     a las 4461 del conjunto completo).
P2 - ausencia de guarda al comparar BPC finales de corridas no convergidas.

Cada ancla reproduce el defecto o fija el contrato del arreglo. La criterio 1 es la
que protege TODO el arco: unificar la metrica no debe alterar ni un bit los resultados
sobre los datos int64 que usaron T9-T13.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank
import t14
from convergencia import (NoConvergido, comparar_final, convergido,
                          pendiente_final, tabla)

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLA ") + msg, flush=True)
    if not cond:
        FALLOS.append(msg)


def main():
    print("=== P1: una sola metrica de evaluacion ===", flush=True)
    check(t14.val_bpc is bank.val_bpc_full,
          "1. t14.val_bpc ES bank.val_bpc_full (no hay copia divergente)")

    tr, va, V = bank.load_data("cuda")
    ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "ref_bpc.json")))["bpc_referencia"]
    m = bank.build(V, 384, 6, 6, 192, "cuda", 0)
    m.load_state_dict(torch.load("ref_modelo.pt", map_location="cuda"))
    ahora = bank.val_bpc_full(m, va)
    check(repr(ahora) == ref,
          f"2. NO REGRESION: mismo modelo, mismo BPC bit a bit tras el cambio "
          f"({ahora!r} == {ref})")

    # la metrica unica funciona sobre int16 (el formato de T14)
    tr16, va16, V16 = t14.cargar("wt2")
    check(va16.dtype == torch.int16, "3. el corpus de T14 vive en int16")
    b16 = t14.val_bpc(m, va16)
    check(abs(b16 - ahora) < 1e-9,
          f"4. la metrica unica da lo MISMO sobre int16 e int64 ({b16:.10f})")

    n_ventanas = (len(va) - 1) // 256
    check(n_ventanas == 4461, f"5. se evaluan las {n_ventanas} ventanas completas")
    del m
    torch.cuda.empty_cache()

    print("\n=== P2: guarda de convergencia ===", flush=True)
    plana = [[i * 500, 0.0, 2.0 + 0.0001 * (10 - i)] for i in range(11)]
    cayendo = [[i * 500, 0.0, 3.0 - 0.05 * i] for i in range(11)]
    check(convergido(plana), "6. una curva plana se reconoce como convergida")
    check(not convergido(cayendo), "7. una curva que cae NO se da por convergida")
    check(abs(pendiente_final(cayendo) - 0.05) < 1e-9,
          f"8. la pendiente se mide bien ({pendiente_final(cayendo):.4f} por 500 pasos)")

    # ANCLA DEL DEFECTO: las curvas reales de T14 deben ser rechazadas
    d14 = json.load(open("t14_results.json"))["runs"]
    curvas = {k: v["curva"] for k, v in d14.items()}
    try:
        comparar_final(curvas)
        check(False, "9. comparar los BPC finales de T14 deberia lanzar NoConvergido")
    except NoConvergido as e:
        check("siguen mejorando" in str(e) and "presupuesto_fijo" in str(e),
              "9. ANCLA: comparar los BPC finales de T14 lanza NoConvergido con la salida")

    r = comparar_final(curvas, presupuesto_fijo=True)
    check(r["convergido"] is False and "presupuesto fijo" in r["etiqueta"],
          f"10. con presupuesto_fijo=True se permite y queda ETIQUETADO: "
          f"'{r['etiqueta']}'")
    check(len(r["sin_converger"]) == len(curvas),
          f"11. las {len(curvas)} corridas de T14 estan sin converger, todas")

    r2 = comparar_final({"a": plana, "b": [[i * 500, 0.0, 2.1] for i in range(11)]})
    check(r2["convergido"] is True and r2["mejor"] == "a",
          "12. con todas convergidas compara sin protestar y nombra la mejor")

    print("\n  pendientes finales de T14 (BPC por cada 500 pasos):", flush=True)
    for n, paso, bpc, pend, conv in tabla(curvas):
        print(f"    {n:28s} paso {paso} bpc {bpc:.4f} pendiente {pend:+.4f} "
              f"{'convergida' if conv else 'SIGUE CAYENDO'}", flush=True)

    print(f"\nfallos: {len(FALLOS)}", flush=True)
    for f in FALLOS:
        print("  *", f, flush=True)
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())
