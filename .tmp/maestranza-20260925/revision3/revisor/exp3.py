import os, glob, tempfile, torch, warnings
import mztrain.checkpoint as ck
AC = ck.ZActivationCheckpoint
class Cfg:
    def __init__(self): self.compression_level=None; self.storage_path="x"
class Lvl: ULTRA_FAST=1
class Boom:
    def __init__(self, c): raise OSError("disco lleno")
ck.HAS_MNEME=True; ck.MnemeConfig=Cfg; ck.CompressionLevel=Lvl; ck.ZSpace=Boom
AC.reset(); AC.prefer_zspace=True
before=len(glob.glob(os.path.join(tempfile.gettempdir(),"mztrain_act_ckpt_*")))
for i in range(3):
    try: AC.compress_activation(torch.randn(4,64)); print("no raise")
    except Exception as e: print("call",i,"->",type(e).__name__, e, "reason:", AC._zspace_disabled_reason)
after=len(glob.glob(os.path.join(tempfile.gettempdir(),"mztrain_act_ckpt_*")))
print("temp dirs created by 3 failed calls:", after-before)
# global last-writer: prefer toggled after construction
class Good:
    def __init__(self,c): self.s={}
    def register(self,k,t,**kw): self.s[k]=t.clone()
    def load(self,k): return self.s[k]
    def delete(self,k): self.s.pop(k,None)
ck.ZSpace=Good; AC.reset(); AC.prefer_zspace=True
AC.compress_activation(torch.randn(4,64))
AC.prefer_zspace=False  # otro engine con mneme_activation_store=False
k,s,d=AC.compress_activation(torch.randn(4,64))
print("prefer False pero sigue usando ZSpace:", AC._zspace is not None and k in AC._zspace.s)
# mixed routing: key in fallback, then prefer flips True before decompress
AC.reset(); AC.prefer_zspace=False
k,s,d=AC.compress_activation(torch.randn(4,64))
AC.prefer_zspace=True
try: AC.decompress_activation(k,s,d); print("ok")
except Exception as e: print("fallback key tras flip ->", type(e).__name__, e)
