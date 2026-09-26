"""PER-LOG-005: migrate_optimizer acepta un ledger YA consumido junto con el
optimizer anterior a la migracion y transplanta momentos caducos en silencio.

Propiedad (guardas G4-B "ledger caduco pierde momentum en silencio";
migrate_optimizer rechaza "ledger caduco"): una vez migrado un ledger, volver
a migrarlo desde el optimizer PRE-migracion debe fallar ruidosamente; no
debe producir un optimizer cuyo estado retrocede K pasos.
Exit 0 = defecto no reproducido; AssertionError = defecto confirmado.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))

import torch  # noqa: E402

import mztrain.elastic_shape as shape  # noqa: E402


def train(model, opt, x, n):
    for _ in range(n):
        opt.zero_grad()
        model(x, x)[1].backward()
        opt.step()


torch.manual_seed(4)
model = shape.GPT(64, 16, 24, 1, 2, shape.fact_lin(8))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 64, (2, 16))
train(model, opt, x, 3)

recs = shape.widen_gpt(model, 36, noise_scale=0.0)
opt2 = shape.migrate_optimizer(model, opt, recs)
# Control negativo: la primera migracion conserva step=3 y limpia el ledger.
assert float(opt2.state[model.tok.weight]["step"]) == 3.0
assert not model._mzshape_pending_recs
print("[control OK] primera migracion: step=3")

train(model, opt2, x, 4)  # el optimizer vivo avanza a step=7
live_step = float(opt2.state[model.tok.weight]["step"])

opt3 = None
try:
    opt3 = shape.migrate_optimizer(model, opt, recs)  # optimizer PRE-migracion + ledger consumido
except (ValueError, RuntimeError) as e:
    print("[rechazado ruidosamente]", e)

if opt3 is not None:
    stale_step = float(opt3.state[model.tok.weight]["step"])
    raise AssertionError(
        f"PER-LOG-005 CONFIRMADO: re-migracion con ledger consumido aceptada; step {stale_step} "
        f"vs {live_step} del optimizer vivo ({live_step - stale_step:.0f} pasos de momento perdidos)"
    )
print("PER-LOG-005 no reproducido")
