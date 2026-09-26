"""PER-API-007 (ALTO): apply_event, con `max_loss_increase` activado y un
`probe_targets` donde TODOS los tokens usan ignore_index=-100 (ningun
target valido), es ACEPTADO en silencio, contradiciendo la promesa
explicita de docs/ELASTICSHAPE-SAFETY.md:

    "Un conjunto sin targets validos produce una metrica no finita y se
    rechaza."

Causa: F.cross_entropy(..., reduction='mean') en PyTorch >= 1.10, cuando
TODOS los elementos estan enmascarados por ignore_index, devuelve 0.0 (no
NaN) por diseno (se corrigio para evitar 0/0 -> NaN). `_probe_metrics`
(elastic_shape.py) calcula loss_before y loss_after con exactamente esa
llamada; con ambos en 0.0, loss_delta = 0.0, que ES finito, así que:
  - no dispara `_GrowthRejected("metricas no finitas en la sonda")`
  - `report["loss_delta"] > max_loss_increase` es 0.0 > umbral, casi
    siempre False -> el growth se ACEPTA con `accepted=True`.

Un caller que confía en el guard (p.ej. porque su batch de sonda quedo
completamente enmascarado por un bug de padding en su propio pipeline)
cree que tuvo una evaluacion de calidad real y no la tuvo: el guard
"fallo abierto" exactamente en el caso que la documentacion dice que
"se rechaza".

Ubicacion:
    docs/ELASTICSHAPE-SAFETY.md: "Un conjunto sin targets validos produce
    una metrica no finita y se rechaza."
    src/mztrain/elastic_shape.py:523-538 (_probe_metrics)
    src/mztrain/elastic_shape.py:635-636 (chequeo de max_loss_increase)

Control negativo: con targets validos, el mismo evento se evalua con una
metrica de perdida real (no forzosamente 0.0).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import math

import torch

from mztrain.elastic_shape import GPT, GrowthEvent, apply_event, dense_lin

torch.manual_seed(0)

VOCAB, SEQ, D, HEADS = 64, 16, 24, 2


def build():
    model = GPT(VOCAB, SEQ, D, layers=1, heads=HEADS, lin=dense_lin)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


probe_x = torch.randint(0, VOCAB, (2, SEQ))

# --- caso adverso: probe_targets todo -100 (ningun target valido) --------
model, opt = build()
probe_targets_all_ignored = torch.full((2, SEQ), -100, dtype=torch.long)

ev = GrowthEvent(step=0, factorize=True, new_d=48, noise_scale=1e-3)
opt2, report = apply_event(
    model,
    opt,
    ev,
    probe_x=probe_x,
    probe_targets=probe_targets_all_ignored,
    max_loss_increase=0.0,  # cualquier incremento de perdida rechaza
)

print(
    "adverso: accepted=%r loss_before=%r loss_after=%r loss_delta=%r"
    % (
        report["accepted"],
        report.get("loss_before"),
        report.get("loss_after"),
        report.get("loss_delta"),
    )
)

assert not report["accepted"], (
    "DEFECTO PER-API-007: con probe_targets TODO -100 (ignore_index, sin targets "
    "validos), SAFETY.md promete 'una metrica no finita y se rechaza', pero "
    f"apply_event ACEPTO el growth (accepted=True) con loss_delta="
    f"{report.get('loss_delta')!r} (finito, tipicamente 0.0 porque "
    "F.cross_entropy con reduccion 'mean' y todo enmascarado devuelve 0.0 en "
    "PyTorch moderno, no NaN)."
)

# --- control negativo: targets validos producen una metrica real ---------
model_b, opt_b = build()
probe_targets_valid = torch.randint(0, VOCAB, (2, SEQ))
ev_b = GrowthEvent(step=0, factorize=True, new_d=48, noise_scale=1e-3)
opt2_b, report_b = apply_event(
    model_b,
    opt_b,
    ev_b,
    probe_x=probe_x,
    probe_targets=probe_targets_valid,
    max_loss_increase=100.0,  # umbral laxo: solo queremos ver una metrica real
)
print(
    "control: accepted=%r loss_before=%r loss_after=%r"
    % (report_b["accepted"], report_b.get("loss_before"), report_b.get("loss_after"))
)
assert math.isfinite(report_b["loss_before"]) and report_b["loss_before"] > 0.0, (
    "control negativo roto: con targets validos se espera una cross-entropy real "
    "(> 0), no el caso degenerado de todo-ignorado"
)

print("PER-API-007: OK (si ves esto sin AssertionError arriba, el defecto existe)")
