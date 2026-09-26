import os, warnings, logging, collections, torch, mneme
logging.disable(logging.CRITICAL)
warnings.simplefilter("ignore")
from mztrain.checkpoint import ZActivationCheckpoint as AC
dev = "cuda" if torch.cuda.is_available() else "cpu"
AC.reset(); AC.prefer_zspace = True
t = torch.randn(16, 256, 512, device=dev)
k, sh, dt = AC.compress_activation(t); AC.decompress_activation(k, sh, dt)
z = AC._zspace
print("has name_to_desc:", hasattr(z, "name_to_desc"), "list_tensors:", len(z.list_tensors()) if hasattr(z,'list_tensors') else '?')
# 1) probe blind spot: plain register small vs large
for shape in [(64,64),(16,256,512)]:
    x = torch.randn(*shape, device=dev)
    z.register("pb", x); y = z.load("pb").to(x.device).float(); z.delete("pb")
    print("plain register", shape, "rel", float((y-x).norm()/x.norm()))
# 2) dtypes via adapter and raw
for dtp in [torch.float16, torch.bfloat16]:
    x = torch.randn(16,256,512, device=dev, dtype=dtp)
    try:
        z.register("d", x, quantization_type="int8"); print(dtp, "int8 kwarg ok"); z.delete("d")
    except Exception as e:
        print(dtp, "int8 kwarg ->", type(e).__name__, str(e)[:100])
        try:
            z.register("d", x); y=z.load("d").to(x.device).float(); z.delete("d")
            print(dtp, "  plain ok rel", float((y-x.float()).norm()/x.float().norm()))
        except Exception as e2:
            print(dtp, "  plain ->", type(e2).__name__, str(e2)[:100])
    try:
        k, sh, d2 = AC.compress_activation(x); back = AC.decompress_activation(k, sh, d2)
        print(dtp, "adapter rel", float((back.float()-x.float()).norm()/x.float().norm()), back.dtype)
    except Exception as e:
        print(dtp, "adapter ->", type(e).__name__, str(e)[:120])
# 3) disk usage
sp = z.config.storage_path
def du(p):
    tot=0; n=0
    for r,_,fs in os.walk(p):
        for f in fs: tot+=os.path.getsize(os.path.join(r,f)); n+=1
    return tot,n
print("dir after rounds:", du(sp))
for _ in range(20):
    k, sh, d2 = AC.compress_activation(t); AC.decompress_activation(k, sh, d2)
print("dir after 20 more:", du(sp))
# compress without decompress (simulated crash mid-step)
keys=[AC.compress_activation(t) for _ in range(3)]
print("dir with 3 pending:", du(sp))
AC.reset()
print("after reset dir exists:", os.path.isdir(sp), du(sp))
import tempfile, glob
print("mztrain temp dirs in tempdir:", len(glob.glob(os.path.join(tempfile.gettempdir(), "mztrain_act_ckpt_*"))))
