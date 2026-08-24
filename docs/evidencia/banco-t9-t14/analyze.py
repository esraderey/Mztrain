"""Aplica las reglas PREREGISTRADAS de T9 a t9_results.json. Solo lee JSON (sin
torch, sin GPU): puede correrse mientras los brazos siguen ejecutandose."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STALL = 3.20
SIGMA2 = 0.136  # 2*sigma_1 del arco (T4/T6)


def load():
    with open(os.path.join(HERE, "t9_results.json"), encoding="utf-8") as f:
        return json.load(f)


def group(runs, prefix):
    return [(k, v) for k, v in sorted(runs.items()) if k.startswith(prefix)]


def ref(runs, tag):
    rs = [v for k, v in group(runs, f"R/{tag}/")]
    ok = [r for r in rs if r["final_bpc"] <= STALL]
    stalled = [r for r in rs if r["final_bpc"] > STALL]
    if not ok:
        return None
    return {
        "Q": sum(r["final_bpc"] for r in ok) / len(ok),
        "T_R": sum(r["train_s"] for r in ok) / len(ok),
        "seeds": [r["seed"] for r in ok],
        "stalled": [(r["seed"], r["final_bpc"]) for r in stalled],
        "bpcs": [r["final_bpc"] for r in ok],
        "T_Rs": [r["train_s"] for r in ok],
    }


def morphs(runs, prefix):
    return [v for _, v in group(runs, prefix)]


def summarize(ms):
    hit = [m for m in ms if m["T_M"] is not None]
    if not hit:
        return None
    return {
        "n": len(ms), "n_hit": len(hit),
        "T_M": [m["T_M"] for m in hit],
        "mean_T_M": sum(m["T_M"] for m in hit) / len(hit),
        "ratio": sum(m["T_M"] for m in hit) / len(hit) / hit[0]["T_R"],
        "bpc_at_T_R": [m["bpc_at_T_R"] for m in hit],
        "final_bpc": [m["final_bpc"] for m in ms],
        "drifts": [[round(e["drift_rel"], 5) for e in m["events"]] for m in ms],
        "bpc_rel_delta": [[round(e["bpc_rel_delta"], 5) for e in m["events"]] for m in ms],
    }


def verdict_ratio(ratio, all_hit, lo=0.7, hi=0.85):
    if ratio is None or not all_hit:
        return "MUERTE (algun seed no alcanza Q)"
    if ratio <= lo:
        return f"CONFIRMADO (ratio {ratio:.3f} <= {lo})"
    if ratio > hi:
        return f"MUERTE (ratio {ratio:.3f} > {hi})"
    return f"INCONCLUSO ({lo} < ratio {ratio:.3f} <= {hi})"


def show(title, r, s):
    print(f"\n=== {title} ===")
    if r is None:
        print("  sin referencia valida todavia"); return
    print(f"  Q={r['Q']:.4f}  T_R={r['T_R']:.1f}s  seeds validos={r['seeds']}"
          f"  BPCs={[round(b, 4) for b in r['bpcs']]}")
    if r["stalled"]:
        print(f"  EXCLUIDOS por estancamiento (>{STALL}): {r['stalled']}")
    if s is None:
        print("  sin morphs completados todavia"); return
    print(f"  morphs: {s['n_hit']}/{s['n']} alcanzan Q; T_M={s['T_M']}"
          f"  media={s['mean_T_M']:.1f}s")
    print(f"  ratio = {s['ratio']:.3f}   ->  {verdict_ratio(s['ratio'], s['n_hit'] == s['n'])}")
    print(f"  BPC a reloj = T_R: {s['bpc_at_T_R']}  (Q={r['Q']:.4f})")
    print(f"  BPC final: {[round(b, 4) for b in s['final_bpc']]}")
    print(f"  deriva de logits por evento: {s['drifts']}")
    print(f"  delta BPC relativo por evento: {s['bpc_rel_delta']}")


def main():
    d = load()
    runs = d["runs"]
    print(f"runs completados: {len(runs)}  ({d.get('updated')})")

    r1 = ref(runs, "S1")
    m1 = summarize(morphs(runs, "M1/S1/"))
    m2 = summarize(morphs(runs, "M2/S1/"))
    show("T9-B  S1 fp32 - morph SIMPLE (replica T8)", r1, m1)
    show("T9-B  S1 fp32 - morph ENCADENADO (2 cirugias)", r1, m2)
    if m1 and m2:
        q = m2["ratio"] / m1["ratio"]
        v = ("H-CHAIN SOBREVIVE" if q <= 1.15 else
             "MUERTE de H-CHAIN" if q > 1.30 else "INCONCLUSO")
        print(f"\n  >>> ratio_M2/ratio_M1 = {q:.3f}  ->  {v}"
              f"   (umbrales preregistrados: <=1.15 / >1.30)")

    rb = ref(runs, "S1_bf16")
    mb = summarize(morphs(runs, "M1/S1_bf16/"))
    show("T9-C  S1 bf16", rb, mb)
    if rb and r1:
        dq = abs(rb["Q"] - r1["Q"])
        print(f"  paridad de calidad |Q_bf16 - Q_fp32| = {dq:.4f}  ->  "
              f"{'PARIDAD' if dq <= SIGMA2 else 'DIVERGENCIA'} (umbral {SIGMA2})")
        print(f"  speedup de reloj de train (R): {r1['T_R'] / rb['T_R']:.3f}x")

    r2 = ref(runs, "S2")
    m2s = summarize(morphs(runs, "M1/S2/"))
    show("T9-A  S2 fp32 - ESCALA (endpoint 38.8M)", r2, m2s)


if __name__ == "__main__":
    sys.exit(main())
