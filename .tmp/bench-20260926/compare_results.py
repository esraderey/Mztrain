"""Compara los JSON nuevos de bench/results con los respaldados (mayo 2026) hoja a hoja."""
import json, sys, pathlib
ROOT = pathlib.Path("D:/mztrain"); NEW = ROOT / "bench/results"; OLD = ROOT / ".tmp/bench-20260926/results-backup"

def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:6]): yield from flat(v, f"{p}[{i}]")
    else: yield p, o

def compare(name, old_name=None, skip=("timestamp", "wall_time", "torch_version", "elapsed", "time_s", "device", "label", "epoch_times")):
    a = json.load(open(OLD / (old_name or name), encoding="utf-8")); b = json.load(open(NEW / name, encoding="utf-8"))
    fa, fb = dict(flat(a)), dict(flat(b))
    print(f"\n===== {name}  (viejo: {old_name or name})")
    rows = []
    for k in sorted(set(fa) | set(fb)):
        if any(s in k for s in skip): continue
        va, vb = fa.get(k, "—"), fb.get(k, "—")
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            if va == vb: continue
            rel = abs(vb - va) / (abs(va) if va else 1.0)
            rows.append((k, va, vb, f"{rel:+.2%}" if va else "n/a"))
        elif va != vb:
            rows.append((k, va, vb, "≠"))
    if not rows: print("  sin diferencias en hojas numericas/textuales (fuera de tiempos/timestamps)")
    for k, va, vb, d in rows[:80]:
        print(f"  {k:60s} {str(va)[:18]:>18s} -> {str(vb)[:18]:>18s}  {d}")
    if len(rows) > 80: print(f"  ... {len(rows)-80} diferencias mas")

for n, o in (("elastic_bench_from_scratch.json", None), ("elastic_bench_guard.json", None), ("elastic_bench_real.json", None),
             ("post_peritaje_20260926.json", "fixed_v2.json")):
    try: compare(n, o)
    except FileNotFoundError as e: print(f"\n===== {n}: falta ({e.filename})")
