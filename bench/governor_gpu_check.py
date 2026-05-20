"""
Validacion REAL del VRAM Governor en GPU (no inyectado, no mock).

Hay GPU (RTX 4060, 8.59GB). Esto ejercita la ruta CUDA de verdad:
  1. Sensor real (mem_get_info / memory_reserved / max_memory_allocated).
  2. Lazo cerrado real: allocar memoria de verdad -> presion sube ->
     el governor BLOQUEA el crecimiento de rango; liberar + reset ->
     vuelve a permitir (y documenta el caveat del caching allocator:
     reserved/peak son pegajosos sin empty_cache + reset_peak).
  3. OOM real: forzar torch.cuda.OutOfMemoryError y verificar que
     oom_guarded lo captura, vacia cache y reintenta una vez.
  4. Entrenamiento end-to-end en device=cuda con el governor activo.

Uso: python -m bench.governor_gpu_check
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mztrain import ZTrainConfig, RankSchedule, ZTrainEngine  # noqa: E402
from mztrain.vram_governor import ZVRAMGovernor  # noqa: E402

assert torch.cuda.is_available(), "este check requiere GPU"
DEV = torch.device("cuda")
_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          f"{'  -- ' + detail if detail else ''}")


def cfg(**kw):
    base = dict(use_vram_governor=True, vram_governor_interval=1,
                vram_governor_ema_beta=0.0, vram_preventive_threshold=0.90,
                vram_emergency_threshold=0.98, vram_hysteresis_checks=1,
                vram_oom_retry=True)
    base.update(kw)
    return ZTrainConfig(**base)


# ---------------------------------------------------------------------------
print("\n== 1. Sensor real ==")
g = ZVRAMGovernor(cfg())
snap = g.observe(device=DEV)                       # SIN reader -> ruta CUDA
free0, total0 = torch.cuda.mem_get_info(DEV)
check("observe() devuelve snapshot real", snap is not None)
if snap:
    check("total ~ mem_get_info", abs(snap.total - total0) < 1e6,
          f"snap.total={snap.total/1e9:.2f}GB")
    check("presion baja en arranque", snap.pressure < 0.5,
          f"pressure={snap.pressure:.4f} mode={snap.mode}")
    check("no bloquea growth en arranque", g.block_growth is False)

# ---------------------------------------------------------------------------
print("\n== 2. Lazo cerrado real: allocar -> bloquear -> liberar ==")
free, total = torch.cuda.mem_get_info(DEV)
# Allocar ~40% del total (seguro: < free), float32.
n = int(0.40 * total / 4)
g2 = ZVRAMGovernor(cfg(vram_preventive_threshold=0.25,
                       vram_emergency_threshold=0.95,
                       vram_governor_ema_beta=0.0))
g2.observe(device=DEV)
before_block = g2.block_growth
hog = torch.empty(n, dtype=torch.float32, device=DEV)  # presion real
torch.cuda.synchronize()
s2 = g2.observe(device=DEV)
check("presion sube tras allocar real",
      s2 is not None and s2.pressure > 0.25,
      f"pressure={s2.pressure:.3f} (alloc~{n*4/1e9:.2f}GB) mode={s2.mode}")
check("growth BLOQUEADO bajo presion real (no estaba antes)",
      (not before_block) and g2.block_growth is True)
check("approve_rank_growth veta el crecimiento",
      g2.approve_rank_growth(128, 64) == 64)

del hog
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEV)            # caveat: peak es monotono
for _ in range(3):
    s3 = g2.observe(device=DEV)
check("tras liberar+empty_cache+reset_peak: presion baja",
      s3 is not None and s3.pressure < 0.25,
      f"pressure={s3.pressure:.4f} mode={s3.mode}")
check("growth RE-PERMITIDO tras liberar",
      g2.approve_rank_growth(128, 64) == 128)

# ---------------------------------------------------------------------------
print("\n== 3. OOM real capturado y recuperado ==")
g3 = ZVRAMGovernor(cfg(vram_oom_retry=True))
_st = {"n": 0}


def oom_then_ok():
    # 1a llamada: pedir mucho mas que el total -> OOM real.
    # reintento: allocacion pequeña -> ok.
    if _st["n"] == 0:
        _st["n"] += 1
        _f, _t = torch.cuda.mem_get_info(DEV)
        big = torch.empty(int(_t / 2), dtype=torch.float32, device=DEV)
        big2 = torch.empty(int(_t / 2), dtype=torch.float32, device=DEV)
        return (big, big2)
    return torch.empty(16, device=DEV)


try:
    r = g3.oom_guarded(oom_then_ok)
    check("oom_guarded recupera tras OOM real (retry)",
          r is not None and g3._counters["oom_events"] == 1
          and g3._counters["oom_recoveries"] == 1,
          f"events={g3._counters['oom_events']} "
          f"recov={g3._counters['oom_recoveries']}")
    del r
except torch.cuda.OutOfMemoryError:
    check("oom_guarded recupera tras OOM real (retry)", False,
          "relanzo OOM (el retry tambien OOMeo)")
torch.cuda.empty_cache()

g4 = ZVRAMGovernor(cfg(vram_oom_retry=True))


def always_oom():
    _f, _t = torch.cuda.mem_get_info(DEV)
    return torch.empty(int(_t), dtype=torch.float32, device=DEV)  # >100% -> OOM


try:
    g4.oom_guarded(always_oom)
    check("OOM persistente -> re-lanza", False, "no relanzo")
except torch.cuda.OutOfMemoryError:
    check("OOM persistente -> re-lanza",
          g4._counters["oom_events"] == 1
          and g4._counters["oom_recoveries"] == 0)
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
print("\n== 4. Entrenamiento end-to-end en GPU con governor activo ==")
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
                      nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
x = torch.randn(128, 128)
y = torch.randint(0, 10, (128,))
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


def loss_fn(m, b):
    xb, yb = b
    return F.cross_entropy(m(xb), yb)


engine = ZTrainEngine(
    model,
    cfg(initial_rank=8, max_rank=32, use_amp=False,
        rank_schedule=RankSchedule.EXPONENTIAL,
        rank_growth_interval=1, rank_growth_factor=2.0,
        vram_preventive_threshold=0.90),
    device=DEV,
)
summary = engine.train(loader, None, loss_fn, epochs=3)
gs = summary.get("governor_stats")
check("entrena en GPU sin crashear", len(summary["train_losses"]) == 3)
check("governor_stats con presion real",
      gs is not None and gs["pressure_ema"] is not None
      and len(gs["pressure_history"]) > 0,
      f"mode={gs['mode']} pressure_ema={gs['pressure_ema']} "
      f"hist_len={len(gs['pressure_history'])}")
check("rango crecio (presion baja, modelo pequeño)",
      summary["ranks"][-1] > 8, f"ranks={summary['ranks']}")
check("0 OOM en run normal",
      gs["actions"]["oom_events"] == 0)

# ---------------------------------------------------------------------------
ok = sum(1 for _, p in _results if p)
tot = len(_results)
print(f"\n{'='*56}\n  VRAM Governor GPU validation: {ok}/{tot} PASS\n{'='*56}")
sys.exit(0 if ok == tot else 1)
