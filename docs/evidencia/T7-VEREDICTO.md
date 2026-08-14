# T7 — Veredicto (dinámica de la parametrización; reglas preregistradas)

**Fecha:** 2026-08-11. **Preregistro:** PREREGISTRO-T7-dinamica.md. **Datos:** t7_results.json
(13 runs, 0 NaN). Líneas base T4 (wd=0.01): F_64 3.3221±0.068, D_small 3.0273±0.005.

## Resultados (val_bpc_full, 3 seeds)

| Condición | Media | σ | Rango efectivo final | Desbalance gauge final |
|---|---|---|---|---|
| F64 wd=0 | 3.4271 | **0.193** | 63.0 | 0.21–0.33 |
| D_small wd=0 | 3.0231 | 0.006 | — | — |
| F64 sin decay en S | 3.3281 | 0.084 | 63.0 | 0.22–0.33 |
| F64 + rebalanceo gauge c/250 | 3.2882 | 0.078 | 62.4 | **0.03–0.04** |
| F64 wd=0.01 (ref. instrumentada) | 3.3753 (1s) | — | 63.0 | 0.24 |

## Veredictos

- **M-A (triple weight decay causa parte del gap): MUERTO.** gap_wd0 = +0.404 ≥ umbral de muerte
  0.265. El álgebra es cierta — AdamW decoupled decae W=U·S·V como (1−ηλ)³, 3× el denso
  (optimizer.py:221, decay uniforme sin exclusiones) — pero su efecto es inocuo a estas escalas:
  quitar el decay EMPEORÓ al factorizado y triplicó su ruido entre seeds (σ 0.193 vs 0.068),
  mientras el denso resultó insensible (3.023 vs 3.027).
- **M-A vía muerte de direcciones: FALSADA limpiamente.** El rango efectivo (participation ratio
  de S) terminó ≈63/64 en TODAS las condiciones, idéntico con y sin decay — ninguna dirección
  murió; la instrumentación paso a paso lo fotografía.
- **M-B (deriva de gauge causa el ruido/inestabilidad): MUERTO.** La deriva EXISTE (desbalance
  log de normas hasta 0.33 medido) y la cirugía de rebalanceo la controla (0.04, con W
  exactamente preservada y estado Adam re-escalado), pero σ no colapsó (0.078 vs regla ≤0.034)
  y la media mejoró solo 0.5σ. La deriva es real y benigna.

## Lo aprendido

1. El factorizado NECESITA regularización: wd la estabiliza (contra-hipótesis confirmada).
2. El caos entre seeds del factorizado (~×10-40 vs denso) queda SIN causa identificada tras
   descartar con evidencia: LR (T6), gauge, weight decay, muerte de direcciones. Candidato
   restante: paisaje de pérdida intrínsecamente más afilado/caótico de la parametrización
   (verificable con sondas de curvatura; fuera del alcance de T7).
3. Higiene práctica sin costo: nunca wd=0 con capas factorizadas; rebalanceo periódico opcional
   (mejor media observada, dentro de ruido); ≥3 seeds en toda comparación con factorizado.
