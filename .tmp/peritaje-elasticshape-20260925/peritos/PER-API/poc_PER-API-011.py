"""PER-API-011: pad_adam_entry no restringe cuales claves trata como
"momento de Adam" (candidatas a zero-pad). La condicion es puramente
estructural: cualquier valor que sea un tensor Y tenga la MISMA forma que
`exp_avg` se trata como si fuera un momento de Adam y se le aplica
pad_state_tensor -- incluida una clave completamente ajena al optimizador
(p.ej. un contador o mascara por-parametro de un adaptador externo) que
coincida en forma por casualidad. docs/ELASTICSHAPE-SAFETY.md solo
describe 'exp_avg', 'exp_avg_sq' y 'max_exp_avg_sq'; no dice nada sobre
claves desconocidas.

Ubicacion: src/mztrain/shape_ops.py:343-347
    for key, value in entry.items():
        if key != "step" and isinstance(value, torch.Tensor) and value.shape == old_shape:
            out[key] = pad_state_tensor(value, new_shape, row_map, col_map)
        else:
            out[key] = copy.deepcopy(value)

Esto puede ser una decision de diseno deliberada (generalidad), pero no
esta documentada, y significa que un estado de optimizer con una clave
extra del mismo shape se transforma exactamente igual que un momento real,
sin ningun aviso de que se esta extrapolando fuera de las claves
conocidas de AdamW/AMSGrad.

Control negativo: con 'exp_avg'/'exp_avg_sq'/'step' unicamente (el caso
documentado), pad_adam_entry se comporta como se espera.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from mztrain.shape_ops import pad_adam_entry

old_shape = (4,)
entry = {
    "exp_avg": torch.zeros(old_shape),
    "exp_avg_sq": torch.zeros(old_shape),
    "step": torch.tensor(3.0),
    # clave ajena a AdamW/AMSGrad que por casualidad comparte forma:
    "custom_adapter_counter": torch.arange(4, dtype=torch.float32),
}

out = pad_adam_entry(entry, (6,))

assert out["custom_adapter_counter"].shape == entry["custom_adapter_counter"].shape, (
    "DEFECTO PER-API-011: pad_adam_entry trato 'custom_adapter_counter' (una clave "
    "ajena a AdamW/AMSGrad, no documentada) como si fuera un momento de Adam y la "
    f"expandio a shape {tuple(out['custom_adapter_counter'].shape)} en vez de "
    f"dejarla intacta con shape {tuple(entry['custom_adapter_counter'].shape)} via "
    "copy.deepcopy, unicamente porque coincidia en forma con 'exp_avg'."
)

# --- control negativo: solo las claves documentadas -----------------------
entry_ok = {
    "exp_avg": torch.zeros(old_shape),
    "exp_avg_sq": torch.zeros(old_shape),
    "step": torch.tensor(3.0),
}
out_ok = pad_adam_entry(entry_ok, (6,))
assert out_ok["exp_avg"].shape == (6,)
assert float(out_ok["step"]) == 3.0

print("PER-API-011: OK (si ves esto sin AssertionError arriba, el defecto existe)")
