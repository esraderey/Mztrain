"""PER-API-003: widen_embedding, cuando la tabla original tiene norma de
Frobenius 0 (||W||_F == 0), calibra el ruido de las columnas nuevas
proporcional a esa norma, produciendo ruido EXACTAMENTE cero pese a pedir
`noise_scale > 0`. Esto contradice en silencio SPEC-elasticshape-v1.md #2
("Gradientes muertos si todo es 0 ... el gradiente fluye desde el primer
paso") para el caso de embeddings, y es asimetrico con el mismo caso limite
en _functional_noise (usado por widen_factorized_linear), que SI lo
documenta explicitamente: "Un mapa efectivo nulo recibe ruido nulo."
widen_embedding no menciona este caso en ningun docstring ni en SAFETY.md.

Ubicacion: src/mztrain/shape_ops.py:178-190
    ref = emb.weight.detach().to(acc).norm()
    ...
    if norm > 0:
        W[:, fresh] = (noise * (noise_scale * ref / norm)).to(dt)
Si ref == 0, la rama `if norm > 0` puede ejecutarse pero el escalar
`noise_scale * ref / norm` es 0, así que W[:, fresh] se queda en su valor
de zeros() inicial: ruido nulo.

Control negativo: la misma llamada con una tabla NO nula produce columnas
nuevas con ruido no-nulo.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.nn as nn

from mztrain.shape_ops import widen_embedding

torch.manual_seed(0)

# --- caso adverso: tabla de embedding completamente en cero --------------
emb = nn.Embedding(5, 4)
with torch.no_grad():
    emb.weight.zero_()

new_emb = widen_embedding(emb, new_dim=8, noise_scale=1e-3, generator=torch.Generator().manual_seed(1))
new_cols = new_emb.weight.data[:, 4:]

assert new_cols.abs().sum().item() > 0, (
    "DEFECTO PER-API-003: con noise_scale=1e-3 (>0) pero ||W||_F == 0 (tabla toda "
    "cero), widen_embedding produce columnas nuevas EXACTAMENTE cero. El remedio "
    "de SPEC #2 (ruido para que 'el gradiente fluye desde el primer paso') falla "
    "en silencio: no hay excepcion, ni aviso, ni documentacion de este caso limite "
    "para widen_embedding (a diferencia de _functional_noise, que SI lo declara "
    "explicitamente para widen_factorized_linear)."
)

# --- control negativo: tabla no nula produce ruido no nulo ----------------
emb2 = nn.Embedding(5, 4)
nn.init.normal_(emb2.weight, 0, 0.5)
new_emb2 = widen_embedding(emb2, new_dim=8, noise_scale=1e-3, generator=torch.Generator().manual_seed(1))
new_cols2 = new_emb2.weight.data[:, 4:]
assert new_cols2.abs().sum().item() > 0, "control negativo roto: se esperaba ruido no nulo"

print("PER-API-003: OK (si ves esto sin AssertionError arriba, el defecto existe)")
