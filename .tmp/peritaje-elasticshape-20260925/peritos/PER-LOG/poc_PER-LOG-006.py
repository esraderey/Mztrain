"""PER-LOG-006: tras widen_gpt/deepen_gpt directos con migracion pendiente,
el modelo sigue entrenable con el optimizer viejo y NO aprende, sin error.

Propiedad (modelo de amenazas (c) "ninguna ruta degrada en silencio:
migracion sin aplicar"; el propio widen_gpt declara que un ledger pendiente
"pierde momentum en silencio"): mientras _mzshape_pending_recs o
_mzshape_new_param_sources esten pendientes, un paso de entrenamiento o falla
ruidosamente o actualiza todos los parametros entrenables del modelo.
Exit 0 = defecto no reproducido; AssertionError = defecto confirmado.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))

import torch  # noqa: E402

import mztrain.elastic_shape as shape  # noqa: E402


def setup():
    torch.manual_seed(5)
    model = shape.GPT(64, 16, 24, 1, 2, shape.fact_lin(8))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 64, (2, 16))
    return model, opt, x


def steps(model, opt, x, n=3):
    for _ in range(n):
        opt.zero_grad()
        model(x, x)[1].backward()
        opt.step()


def frozen_params(model, opt, x, params):
    before = [p.detach().clone() for p in params]
    raised = None
    try:
        steps(model, opt, x)
    except Exception as e:  # noqa: BLE001
        raised = e
    unchanged = [i for i, (a, p) in enumerate(zip(before, params)) if torch.equal(a, p.detach())]
    return raised, unchanged


# Control negativo: widen + migrate -> todos los parametros se actualizan.
model, opt, x = setup()
recs = shape.widen_gpt(model, 36, noise_scale=1e-3)
opt = shape.migrate_optimizer(model, opt, recs)
raised, unchanged = frozen_params(model, opt, x, list(model.parameters()))
assert raised is None and not unchanged, (raised, unchanged)
print("[control OK] widen+migrate entrena todos los parametros")

problems = []
# Defecto A: widen sin migrar, entrenamiento con el optimizer viejo.
model, opt, x = setup()
shape.widen_gpt(model, 36, noise_scale=1e-3)
params = list(model.parameters())
raised, unchanged = frozen_params(model, opt, x, params)
print(f"[widen sin migrar] excepcion={raised!r} congelados={len(unchanged)}/{len(params)}")
if raised is None and unchanged:
    problems.append(f"widen: {len(unchanged)}/{len(params)} parametros nunca se actualizan")

# Defecto B: deepen sin migrar: el bloque nuevo nunca entrena.
model, opt, x = setup()
shape.deepen_gpt(model, 1)
params = list(model.blocks[-1].parameters())
steps(model, opt, x, 1)  # un paso para dar gradiente a la salida (U de proj/fc2)
raised, unchanged = frozen_params(model, opt, x, params)
print(f"[deepen sin migrar] excepcion={raised!r} congelados={len(unchanged)}/{len(params)}")
if raised is None and unchanged:
    problems.append(f"deepen: {len(unchanged)}/{len(params)} parametros del bloque nuevo congelados")

assert not problems, "PER-LOG-006 CONFIRMADO: " + "; ".join(problems)
print("PER-LOG-006 no reproducido")
