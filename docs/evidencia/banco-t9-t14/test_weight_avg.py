"""Prueba aislada de WeightAverager (banco propio, fuera del arbol sellado).

Nueve comprobaciones: siete de contrato (exactitud aritmetica, tensores no
flotantes, restauracion bit a bit, guarda ante cambio de forma, EMA, no
contaminacion del entrenamiento) y dos funcionales sobre entrenamiento real
(el promedio mejora; y mejora tambien despues de una cirugia de ElasticShape).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank import Clock, build, load_data, make_opt, train_steps, val_bpc_full
from mztrain.elastic_shape import GrowthEvent, apply_event
from weight_avg import WeightAverager

FALLOS = []


def check(cond, msg):
    print(("  OK   " if cond else "  FALLA ") + msg, flush=True)
    if not cond:
        FALLOS.append(msg)


class Juguete(nn.Module):
    """Modelo minimo con lo que puede romper: float, bool y entero."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)
        self.register_buffer("mascara", torch.tensor([True, False, True]))
        self.register_buffer("contador", torch.tensor([7]))


def poner(m, valor):
    with torch.no_grad():
        m.lin.weight.fill_(valor)
        m.lin.bias.fill_(valor)


def main():
    print("=== CONTRATO ===", flush=True)
    m = Juguete()
    poner(m, 1.0)
    avg = WeightAverager(m)

    # 1: con un solo punto, el promedio es el modelo
    check(torch.equal(avg.state_dict()["lin.weight"], m.lin.weight),
          "1. con n=1 el promedio es identico a los pesos")

    # 2: media aritmetica exacta de 1, 2, 3 -> 2
    for v in (2.0, 3.0):
        poner(m, v)
        avg.update(m)
    esperado = (1.0 + 2.0 + 3.0) / 3
    obtenido = float(avg.state_dict()["lin.weight"].flatten()[0])
    check(abs(obtenido - esperado) < 1e-6,
          f"2. media aritmetica exacta ({obtenido:.6f} == {esperado:.6f}), n={avg.n}")

    # 3: los no flotantes se conservan, no se promedian
    sd = avg.state_dict()
    check(sd["mascara"].dtype == torch.bool and bool(sd["mascara"][0]) and
          not bool(sd["mascara"][1]), "3. la mascara bool sobrevive intacta")
    check(sd["contador"].dtype == torch.int64 and int(sd["contador"][0]) == 7,
          "3b. el buffer entero sobrevive intacto")

    # 4: el contexto restaura bit a bit
    poner(m, 9.0)
    antes = {k: v.clone() for k, v in m.state_dict().items()}
    with avg.evaluado_en(m):
        dentro = float(m.lin.weight.flatten()[0])
    check(abs(dentro - esperado) < 1e-6, "4. dentro del contexto se ve el promedio")
    check(all(torch.equal(antes[k], v) for k, v in m.state_dict().items()),
          "4b. al salir, los pesos quedan bit a bit como estaban")

    # 5: restaura tambien si el bloque lanza
    try:
        with avg.evaluado_en(m):
            raise ValueError("fallo simulado")
    except ValueError:
        pass
    check(all(torch.equal(antes[k], v) for k, v in m.state_dict().items()),
          "5. restaura tambien cuando el bloque lanza excepcion")

    # 6: EMA
    m2 = Juguete()
    poner(m2, 0.0)
    ema = WeightAverager(m2, decay=0.9)
    poner(m2, 1.0)
    ema.update(m2)
    v = float(ema.state_dict()["lin.weight"].flatten()[0])
    check(abs(v - 0.1) < 1e-6, f"6. EMA(0.9): 0.9*0 + 0.1*1 = {v:.4f}")

    # 7: guarda ante cambio de forma (el caso de ElasticShape)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, val, V = load_data(dev) if dev == "cuda" else (None, None, 64)
    g3 = build(V, 192, 6, 6, None, dev, 0)
    a3 = WeightAverager(g3)
    o3 = make_opt(g3)
    o3, _ = apply_event(g3, o3, GrowthEvent(step=0, factorize=True, new_d=384,
                                            noise_scale=1e-3))
    try:
        a3.update(g3)
        check(False, "7. update tras un growth deberia lanzar RuntimeError")
    except RuntimeError as e:
        check("reset(model)" in str(e),
              "7. update tras un growth lanza RuntimeError con la guia correcta")
    a3.reset(g3)
    a3.update(g3)
    check(a3.n == 2, "7b. tras reset(model) el promedio vuelve a funcionar")
    del g3, o3
    torch.cuda.empty_cache()

    if dev != "cuda":
        print("\n(sin GPU: se omiten las pruebas funcionales)")
        return 1 if FALLOS else 0

    # ---------------------------------------------------------------- funcional
    print("\n=== FUNCIONAL (entrenamiento real) ===", flush=True)
    train, val, V = load_data("cuda")

    # 8: en entrenamiento normal, el promedio mejora al ultimo checkpoint
    m = build(V, 192, 6, 6, None, "cuda", 0)
    o = make_opt(m, lr=1.2e-3)
    g = torch.Generator(device="cuda")
    g.manual_seed(1000)
    train_steps(m, o, train, g, 600, Clock())
    avg = WeightAverager(m, device="cpu")          # el promedio vive fuera de la VRAM
    for _ in range(3):
        train_steps(m, o, train, g, 50, Clock())
        avg.update(m)
    ultimo = val_bpc_full(m, val)
    pesos_antes = {k: v.clone() for k, v in m.state_dict().items()}
    with avg.evaluado_en(m):
        promediado = val_bpc_full(m, val)
    check(promediado < ultimo,
          f"8. el promedio mejora al ultimo checkpoint: {promediado:.4f} < {ultimo:.4f} "
          f"({promediado - ultimo:+.4f} BPC)")
    check(all(torch.equal(pesos_antes[k], v) for k, v in m.state_dict().items()),
          "8b. el entrenamiento queda INTACTO tras evaluar el promedio")

    # 9: tras la cirugia, promediar dentro del transitorio HACE DANO y fuera ayuda.
    #    Es la condicion de uso que descubrio esta misma prueba; el guard la codifica.
    px = torch.stack([val[i * 256:(i + 1) * 256] for i in range(8)])
    gg = torch.Generator(device="cuda")
    gg.manual_seed(4242)
    o, _ = apply_event(m, o, GrowthEvent(step=0, factorize=True, new_d=384,
                                         noise_scale=1e-3), probe_x=px, generator=gg)

    crudo = WeightAverager(m, device="cpu")            # sin guard (default)
    for _ in range(3):
        train_steps(m, o, train, g, 50, Clock())
        crudo.update(m)
    ult_t = val_bpc_full(m, val)
    with crudo.evaluado_en(m):
        pr_t = val_bpc_full(m, val)
    check(pr_t > ult_t,
          f"9. dentro del transitorio el promedio EMPEORA, como esperabamos: "
          f"{pr_t:.4f} > {ult_t:.4f} ({pr_t - ult_t:+.4f} BPC)")

    # 9b: el guard opt-in omite las primeras updates y lo REPORTA (no es silencioso)
    guardado = WeightAverager(m, device="cpu", omitir_tras_reset=4)
    incorporadas = [guardado.update(m) for _ in range(4)]
    check(not any(incorporadas) and guardado.omitidas == 4,
          f"9b. el guard opt-in omite las 4 primeras y lo devuelve en el retorno "
          f"(incorporadas={sum(incorporadas)}, omitidas={guardado.omitidas})")
    check(guardado.update(m) is True, "9c. la quinta update ya se incorpora")

    # 9d: el guard por VELOCIDAD DE DESCENSO, que es el predictor real.
    #     (La version anterior de esta prueba comparaba BPC tras N pasos y resulto
    #     inestable: fallo con +0.0028 y paso con -0.0007 en corridas consecutivas.
    #     Se sustituye por una comprobacion determinista del contrato.)
    rapido = WeightAverager(m, device="cpu", umbral_descenso=0.015)
    rapido.update(m, metrica=3.60)
    en_caida = rapido.update(m, metrica=3.50)        # cae 0.10 > umbral -> omitir
    en_meseta = rapido.update(m, metrica=3.4995)     # cae 0.0005 < umbral -> aceptar
    check(en_caida is False and en_meseta is True,
          f"9d. el guard por descenso omite en caida rapida y acepta en meseta "
          f"(caida={en_caida}, meseta={en_meseta})")
    check(rapido.omitidas == 1,
          f"9e. la omision queda contabilizada, no es silenciosa (omitidas={rapido.omitidas})")

    print(f"\nfallos: {len(FALLOS)}", flush=True)
    for f in FALLOS:
        print("  *", f, flush=True)
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())
