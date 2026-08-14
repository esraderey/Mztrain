# PREREGISTRO — T6-core: cuatro ataques al gap + el eje bf16

**Fecha:** 2026-08-11, ANTES de ejecutar ningún run. **Precondición cumplida:** criba+maestranza
pre-T6 (GO con 4 condiciones, `CRIBA-MAESTRANZA-preT6.md`); sello re-firmado y verificado.
**Contexto:** T4 estableció gap iso-parámetro Δ̄₁=+0.295 BPC (σ₁=0.068, 3 seeds) estable con la
escala. T6-core ataca el gap con 3 mecanismos ortogonales (fp32, comparables con T4) y valida el
eje bf16 (tiempo/acantilado). fp8 queda fuera (T6b, con 2 prerrequisitos de forja).

**Protocolo base (idéntico a T4 salvo lo declarado):** char-WT2, GPT mínimo, batch 16, seq 256,
4000 pasos, AdamW, S1 (d=384, L=6, h=6) y confirmaciones en S2 (d=768, L=8, h=8). Métrica
primaria `val_bpc_full` (todo el val set, fp32). **Líneas base REUTILIZADAS de t4_results.json**
(mismo protocolo, mismos seeds — declarado): S1 F_64 {3.3817, 3.2470, 3.3375}, D_small {3.0316,
3.0284, 3.0219}, D_full {2.7362, 2.7320, 2.7374}; S2 F 2.9003, D_small 2.5964, D_full 2.5946.

## C1 — Barrido de LR (¿el gap era un artefacto de LR?)

F_64 y D_small en S1, LR ∈ {1.5e-4, 6e-4, 1.2e-3} (×0.5/×2/×4 del 3e-4 de T4; el ×1 se reutiliza
de T4), seeds {0,1,2} → 18 runs. Gap*(mejor-LR) = mean_seeds[F en su mejor LR] − mean_seeds[D_small
en su mejor LR].
- **H-LR:** el gap con LR tuneado se reduce ≥⅓: Gap* ≤ ⅔·Δ̄₁ = 0.197.
- **Muerte de H-LR:** Gap* > 0.8·Δ̄₁ = 0.236 (el LR no era la explicación). Zona 0.197–0.236: inconcluso.
- Predicción del autor: F mejora algo con LR mayor (su ruido entre seeds lo sugiere), pero
  Gap* ∈ [0.7, 0.9]·Δ̄₁ — muerte o inconcluso.

## C2 — Annealing de rango (nacer alto, aprender a ser bajo)

Condición A: r=192 (d/2) → 128@1000 → 96@2000 → 64@3000 (cirugía en boundaries: top-|S|,
`elastic_replace_factors`, optimizer AdamW RECONSTRUIDO con estado migrado por rebanadas —
exp_avg/exp_avg_sq/step de U/S/V rebanados por idx; params no operados trasplantan su estado).
Control: F_192 fijo (descarta el confound "cualquier rango alto ayuda"). Seeds {0,1,2} en S1.
Declarado: A gasta más FLOPs tempranos que F_64 (rango medio ~120); la comparación es a
PASOS iguales con params FINALES iso (64) — se reporta también wall_s.
- **H-ANN:** BPC(A) < mean(F_64 T4) − 2σ_F (σ_F≈0.068) = 3.3221 − 0.136 ≈ 3.186, y BPC(A) < BPC(F_192).
- **Muerte:** BPC(A) ≥ mean(F_64 T4) = 3.322 (annealing no aporta nada sobre nacer bajo).
- Predicción: annealing mejora sobre F_64 fijo (cruza la muerte) pero NO alcanza a D_small (3.027).

## C3 — Factorización selectiva (¿el gap vive en la atención?)

Condición FF_64: fc1/fc2 factorizadas r=64, qkv/proj DENSAS. Control iso-param: denso d elegido
por grid (múltiplos de heads) contra la fórmula exacta de params de FF_64. Seeds {0,1,2} en S1.
- Gap_rel(X) = (BPC_X − BPC_iso(X)) / BPC_iso(X). De T4: gap_rel(F_64) ≈ 9.7%.
- **H-FFN:** gap_rel(FF_64) ≤ ½·gap_rel(F_64) — la mayor parte del gap viene de factorizar atención.
- **Muerte:** gap_rel(FF_64) ≥ 0.8·gap_rel(F_64) (la ubicación no importa; el gap es del mecanismo).
- Predicción: intermedio; me inclino por muerte (el cuello es el rango, no dónde está).

## C4 — bf16 (tiempo y acantilado; la calidad no debe moverse)

Trío (D_full, F_64, D_small) × seeds {0,1,2} en S1 + trío×seed0 en S2, ENTRENANDO bajo
`torch.autocast(cuda, bf16)` con params fp32 (condición de maestranza; sin GradScaler — bf16 no lo
necesita). **EVAL SIEMPRE EN FP32** (sin autocast) para comparabilidad exacta con T4. Además,
re-sonda del acantilado en bf16 (escalera reducida de T5: dense (1280,14),(1408,16); fact
(1792,22),(1920,24),(2048,26), 10 pasos, batch 8).
- **H-BF16-calidad:** |BPC_bf16 − BPC_fp32(T4)| ≤ 2σ₁ = 0.136 por condición (paridad).
- **H-BF16-tiempo:** speedup wall ≥ 1.3× en S2 (en S1 el modelo es chico y el overhead puede comerlo).
- **H-BF16-acantilado:** fact (1920,24) [equiv 1.06B] sale de la zona muerta: ≥ 1 000 tok/s
  (fp32 medido en T5: 159 tok/s).
- Muerte respectiva: divergencia de calidad > 2σ₁; speedup < 1.15×; equiv-1B sigue < 500 tok/s.

## Reglas transversales

- Orden de ejecución: C1 → C2 → C3 → C4 (fp32 primero, comparables entre sí y con T4).
- Harness `t6_core.py` con las condiciones vinculantes de la criba: dump atómico (tmp+os.replace),
  catch OOM amplio (OutOfMemoryError + RuntimeError con "out of memory"), errores por-run sin
  abortar la batería. Artefacto: `t6_results.json` incremental. Smoke real (20 pasos) ANTES del run.
- Análisis con σ₁ de T4 como vara; cada H con su muerte preregistrada arriba; comparaciones
  primarias: 4 (una por componente). Todo lo demás es descriptivo.
- Límites declarados: S1/S2 solamente (11M/57M); 3 seeds solo en S1; schedule de anneal único
  (sin barrido de curvas de annealing); bf16 de la capa vía autocast (no kernels custom).
