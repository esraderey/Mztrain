"""Smoke con el backend REAL mnemosys (PyPI): correr con PYTHONPATH apuntando al site donde esta instalado."""
import sys, time, warnings, logging, torch, mneme
logging.disable(logging.CRITICAL)
from mztrain.checkpoint import ZActivationCheckpoint as AC
print("mneme:", mneme.__version__, mneme.__file__)
dev = "cuda" if torch.cuda.is_available() else "cpu"
# 1) por defecto (opt-in apagado): no se construye ZSpace
AC.reset(); AC.prefer_zspace = False
k, sh, dt = AC.compress_activation(torch.randn(16, 256, 512, device=dev)); AC.decompress_activation(k, sh, dt)
print("1) default -> _zspace is None:", AC._zspace is None)
# 2) opt-in: backend real, fidelidad, dispositivo, sin warnings, registro acotado
AC.reset(); AC.prefer_zspace = True
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t = torch.randn(16, 256, 512, device=dev)
    k, sh, dt = AC.compress_activation(t); back = AC.decompress_activation(k, sh, dt)
    rel = float((back - t).norm() / t.norm())
    t0 = time.perf_counter(); n = 30
    for _ in range(n):
        k, sh, dt = AC.compress_activation(t); AC.decompress_activation(k, sh, dt)
    per = (time.perf_counter() - t0) / n
z = AC._zspace
print(f"2) opt-in -> zspace={type(z).__name__ if z else None} rel_err={rel:.3g} device_ok={back.device == t.device} dtype_ok={back.dtype == t.dtype} "
      f"warnings_mztrain={[str(x.message)[:70] for x in w if 'ZActivationCheckpoint' in str(x.message)]} otros_warnings_backend={len(w)} registro={len(getattr(z, 'name_to_desc', {}))} coste/activacion={per*1000:.0f} ms")
print("   storage_path temporal:", getattr(getattr(z, 'config', None), 'storage_path', '?'))
ok = AC._zspace is None if False else (z is not None and rel < 0.05 and back.device == t.device and not [x for x in w if 'ZActivationCheckpoint' in str(x.message)] and len(getattr(z, 'name_to_desc', {})) == 0)
print("SMOKE", "OK" if ok else "FALLA")
sys.exit(0 if ok else 1)
