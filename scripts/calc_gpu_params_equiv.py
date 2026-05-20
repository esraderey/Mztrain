"""Calculo: params densos-equivalentes vs reales (factorizados), VRAM de
entrenamiento estandar vs mztrain, y equivalentes comerciales.

NO entrena nada: aritmetica pura sobre la estructura de ZCodeBERT
(ZFactorizedTransformerBlock: atencion QKVO 4h^2 + MLP 2*(h*4h) = 12h^2
por bloque; embeddings vocab*h NO se factorizan).

Anclaje empirico (validacion de la formula): la prueba real
scripts/test_zcoder_410m_full.py midio, para h=1024 L=28 vocab=50k r=48:
  - reales = 76,673,840   (formula predice ~76.0M  -> OK)
  - VRAM pico = 3.83 GB con TODO prendido, batch 2, seq 32

Modelo de bytes/param (documentado, no medido salvo el ancla):
  Estandar (Adam, mixed precision): 16 B/param  (fp16 w 2 + fp16 grad 2 +
    fp32 master 4 + Adam m 4 + Adam v 4). Es un PISO (ignora activaciones);
    el real es ~18-20x. Usar 16x es ser GENEROSO con el baseline.
  mztrain: solo entrena los factores reales. ~10 B/param-real
    (factor fp32 4 + grad transitorio 4 + Adam m,v INT8 ~2) + overhead fijo
    (contexto CUDA + runtime MNEME + deepcopy de _factorize + activaciones
    checkpointeadas/comprimidas a batch/seq chico). El overhead fijo se
    CALIBRA con el ancla: 3.83 GB medidos - 10B*76.7M ~= 3.0 GB fijo.
"""

from __future__ import annotations

VOCAB = 50_000
MLP_RATIO = 4          # intermediate = 4*h (ZCodeBERT)
GPU_GB = 8.59          # RTX 4060

# Modelo de VRAM
STD_BYTES_PER_PARAM = 16          # piso Adam mixed-precision
MZ_BYTES_PER_REAL = 10            # factor fp32 + grad + Adam INT8
MZ_FIXED_GB = 3.0                 # calibrado con el ancla empirica


def dense_equiv_params(h: int, L: int, vocab: int = VOCAB) -> int:
    """Params si las lineales fueran densas (= get_model_stats equivalent).

    Por bloque: 4h^2 (QKVO) + 2*(h*4h) (MLP) = 12h^2.  + pooler+mlm_dense 2h^2
    + embeddings (word vocab*h + pos + type ~ vocab*h).
    """
    per_block = 12 * h * h
    transformer = per_block * L
    extra = 2 * h * h                 # pooler.dense + mlm_dense
    embeddings = vocab * h            # decoder atado a word_embeddings
    return transformer + extra + embeddings


def real_params(h: int, L: int, r: int, vocab: int = VOCAB) -> int:
    """Params reales entrenables (factores low-rank) a rango r.

    ZFactorizedLinear(in,out,r): U(out*r)+V(r*in)+S(r)+bias(out).
    Por bloque: 4 atencion (h->h)=4*(2hr+r+h); fc1 (h->4h)=5hr+r+4h;
    fc2 (4h->h)=5hr+r+h.  Embeddings NO se factorizan: + vocab*h.
    """
    attn = 4 * (2 * h * r + r + h)
    fc1 = MLP_RATIO * h * r + h * r + r + MLP_RATIO * h     # in h, out 4h
    fc2 = h * r + MLP_RATIO * h * r + r + h                 # in 4h, out h
    per_block = attn + fc1 + fc2
    transformer = per_block * L
    extra = 2 * (2 * h * r + r + h)                          # pooler+mlm_dense
    embeddings = vocab * h
    return transformer + extra + embeddings


def std_vram_gb(p_dense: int) -> float:
    return STD_BYTES_PER_PARAM * p_dense / 1024**3


def mz_vram_gb(p_real: int) -> float:
    return MZ_FIXED_GB + MZ_BYTES_PER_REAL * p_real / 1024**3


# (etiqueta comercial aproximada, h, L)  -- arquitecturas reales difieren
# (GQA, SwiGLU, sin pooler); es "clase de", no exacto.
TIERS = [
    ("GPT-2 small (~124M)",        768, 12),
    ("GPT-2 medium / BERT-large",  1024, 24),
    ("Pythia-410M (lo testeado)",  1024, 28),
    ("GPT-2 large (~774M)",        1280, 36),
    ("GPT-2 XL (~1.5B)",           1600, 48),
    ("GPT-Neo-2.7B / Pythia-2.8B", 2560, 32),
    ("Llama-2-7B / Mistral-7B",    4096, 32),
    ("Llama-2-13B",                5120, 40),
    ("Llama-2-70B",                8192, 80),
]

RANK = 64    # rango factorizado fijo para la tabla (el test uso 48->84)


def fmt_p(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    return f"{n/1e6:.0f}M"


def main() -> None:
    print(f"  Rango factorizado r={RANK}  |  vocab={VOCAB:,}  |  "
          f"GPU={GPU_GB} GB (RTX 4060)")
    print(f"  Bytes/param: estandar={STD_BYTES_PER_PARAM} (piso Adam mixed) "
          f"| mztrain={MZ_BYTES_PER_REAL}/real + {MZ_FIXED_GB}GB fijo\n")
    hdr = (f"  {'Equivalente comercial':<28}{'denso-eq':>9}{'reales':>9}"
           f"{'%real':>7}{'VRAM std':>10}{'VRAM mz':>9}{'  veredicto'}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for label, h, L in TIERS:
        pd = dense_equiv_params(h, L)
        pr = real_params(h, L, RANK)
        vs = std_vram_gb(pd)
        vm = mz_vram_gb(pr)
        std_ok = vs <= GPU_GB
        mz_ok = vm <= GPU_GB
        if std_ok and mz_ok:
            verd = "ambos OK"
        elif mz_ok and not std_ok:
            verd = f"SOLO mztrain (std x{vs/GPU_GB:.0f} sobre)"
        elif not mz_ok:
            verd = "ni mztrain (bajar r / governor)"
        else:
            verd = "?"
        print(f"  {label:<28}{fmt_p(pd):>9}{fmt_p(pr):>9}"
              f"{100*pr/pd:>6.1f}%{vs:>9.1f}G{vm:>8.1f}G   {verd}")

    print("\n  -- Validacion contra la prueba real (ancla empirica) --")
    h, L, r = 1024, 28, 48
    pr_pred = real_params(h, L, r)
    print(f"  h{h} L{L} r{r} vocab50k:  formula reales = {pr_pred:,}")
    print(f"                            medido en test  = 76,673,840")
    print(f"                            error = "
          f"{100*abs(pr_pred-76_673_840)/76_673_840:.2f}%")
    print(f"  VRAM mztrain predicha = {mz_vram_gb(pr_pred):.2f} GB  "
          f"(medido con todo prendido = 3.83 GB)")

    print("\n  -- Que hace mztrain para lograrlo --")
    print("  1. Factorizacion low-rank: entrena ~1.5*r/h del cuerpo "
          "transformer (a h grande, <5%).")
    print("  2. Adam INT8 (ZCompressedAdam): estados m,v 8B -> ~2B/param.")
    print("  3. Grad TOP_K 90% sparse + error-feedback.")
    print("  4. Activation checkpointing + MNEME INT8 (test: -528 MB).")
    print("  5. VRAM Governor: si la presion sube, BLOQUEA el crecimiento "
          "de rango y reintenta ante OOM (red de seguridad en 13B+).")
    print("  6. ElasticRank: INERTE en from-scratch real (0 ahorro; "
          "documentado) -> NO se cuenta como memoria ganada aqui.")


if __name__ == "__main__":
    main()
