import sys, importlib.util
sys.path.insert(0, "D:/mztrain/src")
import torch
from mztrain import layers as NEW
spec = importlib.util.spec_from_file_location("layers_head", "D:/mztrain/.tmp/maestranza-20260925/revision2/revisor/layers_head.py")
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
import logging; logging.disable(logging.CRITICAL)
import warnings; warnings.simplefilter("ignore")
for mod in (OLD, NEW):
    for gate in ([1., 1., 1.], [1., .5, 0.]):
        torch.manual_seed(2)
        S = mod.ZSparseFactorizedLinear(16, 12, rank=3, bias=False, sparse_density=0.2)
        with torch.no_grad(): S.sparse_values.normal_(); S.wake_gate.copy_(torch.tensor(gate))
        xx = torch.randn(6, 16)
        with torch.no_grad(): yb = S(xx)
        S.grow_rank(5)
        with torch.no_grad(): ya = S(xx)
        print(mod.__name__, gate, float((ya - yb).norm() / yb.norm()), S.wake_gate.tolist())
    for gate in ([1., 1., 1.], [1., .5, 0.]):
        torch.manual_seed(2)
        L = mod.ZFactorizedLinear(16, 12, rank=3, bias=False)
        with torch.no_grad(): L.wake_gate.copy_(torch.tensor(gate))
        xx = torch.randn(6, 16)
        with torch.no_grad(): yb = L(xx)
        L.grow_rank(5, preserve_weights=False)
        with torch.no_grad(): ya = L(xx)
        print("dense", mod.__name__, gate, float((ya - yb).norm() / yb.norm()), L.wake_gate.tolist())
