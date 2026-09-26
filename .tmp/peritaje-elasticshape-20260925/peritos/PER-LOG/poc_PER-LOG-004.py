"""PER-LOG-004: probe_targets solo se valida por forma; dtype/rango invalidos
se descubren DESPUES de la cirugia completa, dentro de _probe_metrics.

Propiedad: un probe_targets invalido (indices fuera de [0, vocab) distintos de
ignore_index, o dtype no entero) se rechaza en la fase de validacion de
apply_event, antes de ejecutar widen_gpt/deepen_gpt/migrate_optimizer.
En CPU el rollback funciona (solo se pierde la cirugia); en CUDA la misma
entrada dispara un device-side assert que envenena el contexto CUDA y deja el
rollback (RNG CUDA / generator) sin garantia. La variante CUDA se ejecuta
solo con --cuda y en un subproceso aislado.
Exit 0 = defecto no reproducido; AssertionError = defecto confirmado.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)

import torch  # noqa: E402

import mztrain.elastic_shape as shape  # noqa: E402

calls = {"widen": 0}
_orig_widen = shape.widen_gpt


def counting_widen(*args, **kwargs):
    calls["widen"] += 1
    return _orig_widen(*args, **kwargs)


shape.widen_gpt = counting_widen  # apply_event resuelve el global en tiempo de llamada


def setup(device="cpu"):
    torch.manual_seed(3)
    model = shape.GPT(64, 16, 24, 1, 2, shape.fact_lin(8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 16), device=device)
    return model, opt, x


# Control negativo: targets validos -> cirugia una vez, aceptado.
model, opt, x = setup()
new, report = shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=36), probe_x=x, probe_targets=x, max_kl=10.0)
assert report["accepted"] and calls["widen"] == 1
print("[control OK] targets validos aceptados")

results = {}
for label, bad in [
    ("fuera de rango", torch.full((2, 16), 999, dtype=torch.long)),
    ("dtype float", torch.zeros(2, 16, dtype=torch.float32)),
]:
    calls["widen"] = 0
    model, opt, x = setup()
    err = None
    try:
        shape.apply_event(model, opt, shape.GrowthEvent(1, new_d=36), probe_x=x, probe_targets=bad, max_kl=10.0)
    except Exception as e:  # noqa: BLE001
        err = e
    results[label] = (type(err).__name__ if err else None, calls["widen"], model.d)
    print(f"[{label}] excepcion={results[label][0]} cirugias_ejecutadas={calls['widen']} model.d={model.d}")

if "--cuda" in sys.argv and torch.cuda.is_available():
    code = f"""
import sys; sys.path.insert(0, {SRC!r})
import torch, mztrain.elastic_shape as shape
torch.manual_seed(3)
m = shape.GPT(64, 16, 24, 1, 2, shape.fact_lin(8)).cuda()
o = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randint(0, 64, (2, 16), device='cuda')
g = torch.Generator(device='cuda').manual_seed(5)
try:
    shape.apply_event(m, o, shape.GrowthEvent(1, new_d=36), probe_x=x,
                      probe_targets=torch.full((2, 16), 999, device='cuda'), max_kl=10.0, generator=g)
except Exception as e:
    print('EXC', type(e).__name__, str(e)[:120])
try:
    torch.zeros(1, device='cuda').add_(1).item(); print('CUDA_OK')
except Exception as e:
    print('CUDA_ENVENENADO', type(e).__name__)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    print("[cuda subproceso]", out.stdout.strip(), out.stderr.strip()[-300:])

late = {k: v for k, v in results.items() if v[1] > 0}
assert not late, (
    "PER-LOG-004 CONFIRMADO: probe_targets invalidos detectados despues de ejecutar la cirugia "
    f"(excepcion, n_widen, d_restaurado): {late}"
)
print("PER-LOG-004 no reproducido")
