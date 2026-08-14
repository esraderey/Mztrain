# T4 — Veredicto (regla preregistrada aplicada a t4_results.json)

**Fecha:** 2026-08-11. **Preregistro:** PREREGISTRO-T4-barrido-escala.md (escrito antes de correr).
**Artefacto crudo:** t4_results.json + stdout tasks/b6f141e1r.output. 15 runs, exit 0, 0 NaN, 0 OOM.

## Datos (métrica primaria: BPC sobre todo el val set)

| Escala | D_full | F | D_small | Δ = F − D_small | VRAM F/D_full |
|---|---|---|---|---|---|
| S1 11M s0 | 2.7362 | 3.3817 | 3.0316 | +0.3501 | 0.939 |
| S1 11M s1 | 2.7320 | 3.2470 | 3.0284 | +0.2186 | 0.937 |
| S1 11M s2 | 2.7374 | 3.3375 | 3.0219 | +0.3156 | 0.937 |
| S2 57M | 2.5946 | 2.9003 | 2.5964 | +0.3039 | 0.834 |
| S3 152M | 2.6621 | 2.7543 | 2.4969 | +0.2574 | 0.777 |

Δ̄₁ = +0.2948 (σ₁ = 0.0682, n=3) · Δ₂ = +0.3039 · Δ₃ = +0.2574

## Regla preregistrada → veredicto

- **H1 confirmada** exigía: Δ₃ ≤ 0.6·Δ̄₁ (=0.177) ∧ Δ₃ < Δ̄₁−2σ₁ (=0.158) ∧ Δ̄₁ > Δ₂ > Δ₃.
  → Falla todo: Δ₃ = 0.257 y Δ₂ > Δ̄₁ (sin monotonía).
- **H1 falsada** si Δ₃ ≥ 0.8·Δ̄₁ = 0.2358. → Δ₃ = 0.2574 ≥ 0.2358. **H1 FALSADA.**
  Δ₃/Δ̄₁ = 0.873; la caída S1→S3 es 0.55·σ₁ (indistinguible de ruido). En relativo el gap es
  ~10% de BPC en las tres escalas (9.7% / 11.7% / 10.3%): **estable a través de 14× params**.
- **H_mem CONFIRMADA:** ratio VRAM F/D_full decrece monótono 0.94 → 0.83 → 0.78 y cumple el
  umbral preregistrado (≤0.8 en S3). El ahorro de memoria emerge con la escala, como predicho.
- **Estabilidad:** 0 NaN en 15 runs hasta 152M params (predicción cumplida; el código reparado
  aguanta el estrés de escala).
- **Predicción del autor** (Δ₃ ∈ [0.8, 1.2]·Δ̄₁): cumplida (0.873).

## Conclusión

**No hay base empírica para proyectar que el gap se cierre a 1B por la vía de la escala.** El
denso iso-parámetro gana con un gap estable (~0.26–0.35 BPC, ~10%) en 11M→152M, y en S3 domina
también en VRAM (2.91 vs 4.04 GB). La única propuesta de valor que queda en pie es la de
habilitación: entrenar modelos que no caben densos en la GPU (el ratio de memoria confirma el
mecanismo), cuya calidad final sigue SIN PROBAR aquí (requiere hardware mayor).

## Secundarios (descriptivos, no preregistrados)

- El mejor modelo del barrido fue **D_small de S3 (33.8M denso): 2.4969** — mejor que los D_full
  de 57M (2.5946) y de 152M (2.6621). A presupuesto fijo de 16.4M chars, 152M está muy
  sub-entrenado (caveat preregistrado nº1 en acción: esto es gap-a-presupuesto-igual, no a
  convergencia).
- F casi alcanza a D_full en S3 (2.754 vs 2.662, gap 0.09 desde 0.65 en S1) con 23% de los params
  y 78% de la VRAM — descriptivo del régimen sub-entrenado, no de la comparación de control.
- La condición factorizada es la más ruidosa entre seeds (rango 0.13 BPC vs ±0.005 de los densos)
  y entre reruns del mismo seed (no-determinismo GPU ±0.04): σ₁ viene casi todo de F.

## Validez

Lo dicho aplica a: char-level WT2, GPT mínimo, AdamW LR 3e-4, 4000 pasos, r/d=1/6, ≤152M,
1 seed en S2/S3. Extrapolar fuera (incluido 1B) es especulación y así debe etiquetarse.
