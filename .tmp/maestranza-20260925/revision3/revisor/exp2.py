import os, warnings, logging, collections, torch
logging.disable(logging.CRITICAL)
from mztrain.checkpoint import ZActivationCheckpoint as AC
dev = "cuda" if torch.cuda.is_available() else "cpu"
AC.reset(); AC.prefer_zspace = True
t = torch.randn(16, 256, 512, device=dev)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    k, sh, dt = AC.compress_activation(t); AC.decompress_activation(k, sh, dt)
    k, sh, dt = AC.compress_activation(t); AC.decompress_activation(k, sh, dt)
z = AC._zspace
print("list_tensors:", z.list_tensors())
print("name_to_desc:", len(z.name_to_desc))
c = collections.Counter((x.category.__name__, os.path.basename(x.filename), str(x.message)[:90]) for x in w)
for kk,v in c.most_common(8): print(v, kk)
