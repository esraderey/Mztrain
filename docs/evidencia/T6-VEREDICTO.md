# T6-core — Veredicto (reglas preregistradas aplicadas a t6_results.json)

**Fecha:** 2026-08-11. **Preregistro:** PREREGISTRO-T6-core.md (previo a los datos).
**Ejecución:** 41 runs, 0 NaN, 0 OOM, ~2.1 h GPU. Líneas base T4 reutilizadas como se declaró.
Vara de ruido: σ₁ = 0.068 (3 seeds, T4).

## C1 — Barrido de LR: **H-LR MUERTA, con hallazgo inverso**

| LR | F_64 (media 3s) | D_small (media 3s) |
|---|---|---|
| 1.5e-4 | 3.572 | 3.461 |
| 3e-4 (T4) | 3.322 | 3.027 |
| 6e-4 | **2.926** | 2.624 |
| 1.2e-3 | 2.959 (seed1 inestable: 3.568 vs 2.654/2.654) | **2.413** (estable) |

Gap*(mejor LR de cada uno) = 2.926 − 2.413 = **+0.513** ≫ umbral de muerte (0.236).
**El LR tuneado no cierra el gap: lo ENSANCHA** (0.295 → 0.513). El denso aprovecha LRs altos
mejor y de forma estable; el factorizado se vuelve inestable en 1.2e-3 (1/3 seeds degenera —
coherente con su firma de ruido entre seeds). Bonus: D_small@1.2e-3 (2.413) supera al D_full de
T4 (2.735) — todo el barrido T4 corrió con LR subóptimo para todos (caveat declarado entonces,
confirmado ahora; no afecta las comparaciones intra-escala de T4, que compartían LR).

## C2 — Annealing de rango: **INCONCLUSO, tendiendo a muerto**

| Condición | BPC (media 3s) | Params fin | Wall |
|---|---|---|---|
| A_192→64 (cirugía top-\|S\| + migración Adam) | 3.311 | 2.86M | ~162 s |
| F_192 fijo (control) | 2.900 | 7.58M | ~196 s |
| F_64 fijo (T4) | 3.322 | 2.86M | ~130 s |

A vs F_64: −0.011 BPC (≪ σ₁) → **beneficio indetectable**, pagando ~25% más de cómputo.
No cruza la muerte formal (A < 3.322 por un pelo) pero tampoco la hipótesis (exigía < 3.186).
El dato duro adyacente: comprimir 192→64 con top-|S| + estado migrado pierde 0.41 BPC vs
quedarse en 192 — **las direcciones aprendidas a rango alto no caben en 64** (eco directo de
T2b: las redes entrenadas no son low-rank). La cirugía funcionó mecánicamente perfecta
(24 capas × 3 boundaries × 3 seeds, assert de migración verde, 0 NaN) — el mecanismo es sólido;
el beneficio, inexistente a esta escala/schedule.

## C3 — FFN-only: **H-FFN MUERTA**

| Condición | BPC (media 3s) | Params |
|---|---|---|
| FF_64 (solo fc1/fc2 factorizadas) | 3.070 | 5.51M |
| D_iso(FF) d=270 (control, +1.6%) | 2.836 | 5.60M |

gap_rel(FF) = 8.3% vs gap_rel(F completo) = 9.7% → conserva el **85% del gap relativo**
renunciando a la mayor parte del ahorro (5.51M vs 2.86M params). Umbral de muerte era ≥7.8%.
**El gap es intrínseco al cuello de rango, no a factorizar la atención.** (Predicción cumplida.)

## C4 — bf16: **tiempo y acantilado CONFIRMADOS; paridad de calidad con una bandera**

**Calidad (eval siempre fp32; parity = |Δ| ≤ 2σ₁ = 0.136):**
| Condición | bf16 | fp32 (T4) | Δ | Parity |
|---|---|---|---|---|
| S1 D_full | 2.739 | 2.735 | +0.003 | ✅ |
| S1 F_64 | 3.360 | 3.322 | +0.038 | ✅ |
| S1 D_small | 3.030 | 3.027 | +0.003 | ✅ |
| S2 D_full | 2.602 | 2.595 | +0.007 | ✅ |
| S2 D_small | 2.634 | 2.596 | +0.038 | ✅ |
| **S2 F_128** | **3.343** | **2.900** | **+0.443** | ❌ (1 seed) |

**BANDERA:** el factorizado a S2 bajo bf16 degrada +0.44 BPC (1 seed; excede todo rango de ruido
conocido de F). Presunto real, pendiente de seeds. Hipótesis mecánica a probar: el backward por
la cadena de 3 matrices acumula redondeo bf16 que el denso no sufre; candidatos de mitigación:
gradiente de S en fp32, o acumulación fp32 en los GEMM de factores. **Hasta resolverlo: bf16 es
seguro para denso a ambas escalas y para factorizado pequeño; NO certificado para factorizado
a escala.** (Ironía instructiva: el argumento "U,V ortonormales ⇒ aptos para baja precisión" era
la tesis fp8 — este dato la tensa y hace el test T6b más interesante, no menos.)

**Velocidad (H ≥1.3× en S2): CONFIRMADA** — D_full 2.44×, F_128 1.98×, D_small 1.54×
(S1: 2.32×/1.12×/1.47× — el F chico es el que menos gana: matmuls flacos).

**Acantilado (H: fact 1.06B-equiv ≥1000 tok/s): CONFIRMADA con 6×** —
| Config (bf16, batch 8) | tok/s | VRAM | vs fp32 (T5) |
|---|---|---|---|
| dense 277M-eq | 7 780 | 5.71 GB | 2.3× |
| dense 383M-eq | 770 | 7.68 GB | aún acantilado |
| fact 850M-eq | 6 303 | 5.99 GB | **14.9×, fuera del acantilado** |
| **fact 1.06B-eq** | **6 361** | **7.20 GB** | **40×, fuera del acantilado** |
| fact 1.31B-eq | 690 | 8.42 GB | acantilado |

**El 1B-equivalente pasó de zona muerta (159 tok/s) a entrenable (6 361 tok/s): 1B de tokens ≈
1.8 días en la 4060.** El mecanismo es el predicho: bf16 reduce activaciones → el pico cae bajo
el acantilado WDDM (~7.2-7.6 GB) → sin paginación.

## Síntesis

1. **El gap de calidad es INTRÍNSECO al cuello de rango.** Tres mecanismos ortogonales fallaron
   en cerrarlo: LR tuneado lo ensancha (0.51), el annealing no aporta nada detectable, y
   reubicar la factorización conserva el 85% del gap. Ya no es "quizá mal configurado": es la
   naturaleza del método a estas escalas.
2. **La historia de tiempo/habilitación mejoró dramáticamente:** con bf16, el 1B-equivalente es
   entrenable de verdad en esta GPU (6.4k tok/s), y denso hasta ~280M gana 2.3×.
3. **Nueva incógnita crítica (bandera):** calidad del factorizado bajo bf16 a escala (+0.44,
   1 seed) — seeds + mitigación de precisión del backward antes de confiar en el combo
   bf16+factorizado grande, que es justamente el combo del titular.
4. Predicciones del autor: C1 ✓ (muerte), C2 ✗ (predije beneficio; no lo hubo — registrado),
   C3 ✓ (muerte), C4 ✓ parcial (parity dense + acantilado sí; la bandera F@S2 no la anticipé).

## Siguiente experimento que más información compra

**T6b-preludio (barato, ~30 min):** 2 seeds más de F_128@S2 bf16 (¿la bandera es real o el seed
más ruidoso del arco?) + 1 variante con gradiente/acumulación fp32 para S. Si la bandera es real
y la mitigación la cierra, el camino al 1B-equiv queda limpio; si no, el titular de velocidad
lleva asterisco de calidad hasta fp8/T6b.
