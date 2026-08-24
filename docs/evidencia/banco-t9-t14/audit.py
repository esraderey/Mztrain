"""Auditoria del banco T9/T10/T11, con el sesgo deliberado de buscar errores que
FAVOREZCAN las conclusiones. Ejecutable desde el directorio del banco.

Checks 1-3 necesitan GPU (verifican la mecanica); 4-7 solo leen los JSON.
"""
import json
import os

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FALLOS = []


def chk(cond, msg):
    if not cond:
        FALLOS.append(msg)
    return cond


# ---------------------------------------------------------------- 1-3 (GPU)
def checks_mecanica():
    import torch
    from bank import Clock, build, load_data, make_opt, train_steps
    from mztrain.elastic_shape import GrowthEvent, LrWarmup, apply_event

    # 1: la rampa hace lo declarado
    m = build(1118, 384, 6, 6, 192, "cuda", 0)
    o = make_opt(m, lr=1.2e-3)
    w = LrWarmup(o, 200, 0.1)
    lrs = []
    for _ in range(260):
        lrs.append(o.param_groups[0]["lr"])
        w.step()
    chk(abs(lrs[0] - 1.2e-4) < 1e-9, "1: el suelo de la rampa no es 0.1*base")
    chk(abs(lrs[200] - 1.2e-3) < 1e-9, "1: la rampa no llega a la base")
    chk(abs(lrs[259] - 1.2e-3) < 1e-9, "1: el LR no se queda en la base tras la rampa")
    chk(all(lrs[i] <= lrs[i + 1] + 1e-12 for i in range(199)), "1: rampa no monotona")
    del m, o
    torch.cuda.empty_cache()

    # 3: el LR y el wd sobreviven a la cirugia (apply_event crea un optimizador NUEVO)
    train, val, V = load_data("cuda")
    m = build(V, 192, 6, 6, None, "cuda", 0)
    o = make_opt(m, lr=1.2e-3)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000)
    train_steps(m, o, train, g, 20, Clock())
    o, _ = apply_event(m, o, GrowthEvent(step=0, factorize=True, new_d=384,
                                         noise_scale=1e-3))
    chk(abs(o.param_groups[0]["lr"] - 1.2e-3) < 1e-12, "3: el LR no sobrevive a la cirugia")
    chk(abs(o.param_groups[0]["weight_decay"] - 0.01) < 1e-12,
        "3: el weight decay no sobrevive a la cirugia")
    del m, o
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 4-7 (datos)
def checks_datos(ficheros=("t9_results.json", "t10_results.json", "t11_results.json")):
    finales = []
    for f in ficheros:
        path = os.path.join(HERE, f)
        if not os.path.exists(path):
            continue
        runs = json.load(open(path, encoding="utf-8"))["runs"]
        for k, r in runs.items():
            c = r.get("curve")
            if not chk(bool(c), f"4: {f}:{k} curva vacia"):
                continue
            chk(abs(r["final_bpc"] - c[-1][2]) < 1e-9,
                f"4: {f}:{k} final_bpc != ultimo punto de la curva")
            chk(all(c[i][1] <= c[i + 1][1] for i in range(len(c) - 1)),
                f"4: {f}:{k} reloj no monotono")
            chk(all(c[i][0] < c[i + 1][0] for i in range(len(c) - 1)),
                f"4: {f}:{k} pasos no monotonos")
            chk(all(b == b for _, _, b in c), f"4: {f}:{k} NaN en la curva")
            if r.get("T_M") is not None:
                hit = next((t for _, t, b in c if b <= r["Q"]), None)
                chk(hit is not None and abs(hit - r["T_M"]) < 1e-6,
                    f"4: {f}:{k} T_M almacenado != recalculado desde la curva")
            if r.get("Q") is not None and r.get("T_M") is None:
                chk(all(b > r["Q"] for _, _, b in c),
                    f"4: {f}:{k} marcado como 'no alcanza Q' pero la curva cruza")
            esperado = 4000 if r["kind"].startswith("R") else 6000
            chk(c[-1][0] == esperado, f"4: {f}:{k} no llego a {esperado} pasos")
            if "degen" in f or True:
                finales.append((r["final_bpc"], f"{f}:{k}"))

    # 7: el umbral de degeneracion no debe ser determinante. El margen se mide POR
    # REGIMEN: lo sano depende del LR (a 3e-4 lo sano vive en ~2.9; a 1.2e-3 en ~2.5),
    # asi que una banda fija seria un error del auditor, no de los datos.
    for f in ficheros:
        path = os.path.join(HERE, f)
        if not os.path.exists(path):
            continue
        v = sorted(r["final_bpc"] for r in
                   json.load(open(path, encoding="utf-8"))["runs"].values())
        sanos = [x for x in v if x <= 3.20]
        malos = [x for x in v if x > 3.20]
        margen = (min(malos) - max(sanos)) if malos else (3.20 - max(sanos))
        chk(margen >= 0.20,
            f"7: {f} margen al umbral demasiado fino ({margen:.3f} BPC): "
            "el umbral estaria decidiendo la clasificacion")
    return finales


def main():
    if "--sin-gpu" not in sys.argv:
        checks_mecanica()
    finales = checks_datos()
    print(f"runs auditados: {len(finales)}")
    print(f"fallos: {len(FALLOS)}")
    for x in FALLOS:
        print("  *", x)
    if finales:
        sanos = [b for b, _ in finales if b <= 3.20]
        malos = [b for b, _ in finales if b > 3.20]
        print(f"separacion bimodal: {len(sanos)} runs <= {max(sanos):.4f} | "
              f"{len(malos)} runs >= {min(malos):.4f}" if malos else
              f"todos sanos (max {max(sanos):.4f})")
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())
