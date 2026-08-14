# PREREGISTRO — T4: Barrido de escala 11M→152M (factorizado vs denso)

**Fecha de preregistro:** 2026-08-11 (ANTES de ejecutar ningún run del barrido).
**Continuación de:** arco T0→T3, sesión "Inicia peritaje DE mztrain" (T2A falsó H1 a 11M;
T3 confirmó con el engine real; queda abierta la tendencia con la escala).

## Pregunta

¿El gap de calidad factorizado-vs-denso **iso-parámetro** se encoge al crecer la escala del
modelo? Es la única vía honesta de proyectar a 1B: si el gap no se encoge en 11M→152M, no hay
base empírica para esperar que se cierre a 1B.

## Hipótesis

- **H1 (la esperanza del framework):** el gap Δ(S) = BPC_F(S) − BPC_D_small(S) decrece
  monótonamente con la escala, y a S3 es sustancialmente menor que a S1.
- **H0:** el gap es estable o crece con la escala (el sesgo low-rank no paga más a mayor escala,
  a este presupuesto).

**Predicción del autor (honesta, antes de correr):** H1 se falsa — espero Δ₃ ∈ [0.8, 1.2]·Δ̄₁
(gap estable). Predicción secundaria mecanística (H_mem): el ahorro de VRAM de F vs D_full
**emerge** con la escala — ratio_mem(S3) = peakVRAM_F/peakVRAM_D_full ≤ 0.8 (vs ≈0.94 en S1),
porque pesos+Adam crecen ∝ params y las activaciones ∝ L·d.

## Métrica (fijada antes)

- **Primaria:** `val_bpc_full` — BPC sobre **todo** el set de validación (ventanas secuenciales
  no solapadas de 256 chars, determinista, sin ruido de muestreo). Menor = mejor.
- **Ancla de reproducibilidad:** `val_bpc_quick20` (protocolo T2A: 20 batches muestreados,
  generator 1234) — S1/seed0 debe reproducir t2a_results.json (D_full 2.7645, F_64 3.4397,
  D_small 3.0557) ± no-determinismo GPU. Solo para validar continuidad, NO es la métrica primaria.
- **Gap:** Δ(S) = BPC_F(S) − BPC_D_small(S) con la métrica primaria, mismo seed.

## Diseño

3 escalas × 3 condiciones, protocolo POR CONDICIÓN idéntico a T2A (una variable: el tipo de capa):

| Escala | d | L | heads | rank F (r=d//6) | F params | D_small (grid) |
|---|---|---|---|---|---|---|
| S1_11M | 384 | 6 | 6 | 64 | 2.86M | d=192 (regla T2A) |
| S2_57M | 768 | 8 | 8 | 128 | 13.59M | ≈368 (−0.8%) |
| S3_152M | 1024 | 12 | 16 | 170 | 34.78M | ≈480 (−2.8%) |

- **Condiciones por escala:** D_full (denso, ancho d) · F (ZFactorizedLinear rank d//6, init svd
  default = régimen correcto de EPSI) · D_small (denso iso-parámetro a F).
- **D_small por regla, no a ojo:** grid de anchos múltiplos de `heads`, argmin |params_dense −
  params_F| con fórmulas exactas verificadas contra T2A al parámetro; tolerancia ≤3% (registrada).
- **Constantes en TODO el barrido:** char-level WikiText-2 (mismo encode/sampling/eval de T2A),
  batch 16, seq 256, **4000 pasos** (= mismos tokens por condición), AdamW LR 3e-4 sin schedule,
  bias=False, embedding tied. Heads fijos por escala (head_dim escala con d, convención T2A).
- **Seeds:** S1 = {0,1,2} (σ₁ = vara de ruido); S2, S3 = {0} (presupuesto). Asunción declarada:
  σ no crece con la escala.
- **Init:** S1 en CPU (fidelidad al ancla T2A); S2/S3 en GPU (SVD-init en CPU a d=1024 costaría
  minutos; el cambio se aplica a las 3 condiciones de la escala por igual — comparabilidad
  intra-escala intacta).
- **Fallback OOM (preregistrado):** si cualquier condición de una escala da CUDA OOM, se re-corre
  la escala COMPLETA con batch 8 / 8000 pasos (mismos tokens), etiquetada `fallback_b8`.

## Regla de decisión (fijada antes)

Con Δ̄₁ = media de Δ sobre los 3 seeds de S1 y σ₁ su desviación:

- **H1 confirmada (gap se encoge):** Δ₃ ≤ 0.6·Δ̄₁ **y** Δ₃ < Δ̄₁ − 2σ₁ **y** monotonía
  Δ̄₁ > Δ₂ > Δ₃. → "hay razón empírica para esperar cierre a 1B".
- **H1 falsada:** Δ₃ ≥ 0.8·Δ̄₁. → "sin base para proyectar cierre a 1B por esta vía".
- **Inconcluso:** zona intermedia [0.6, 0.8)·Δ̄₁, o no-monotonía, o σ₁ tan grande que
  Δ̄₁ − 2σ₁ < 0.6·Δ̄₁ (ruido domina).
- **H_mem:** confirmada si ratio_mem decrece S1→S3 y ratio_mem(S3) ≤ 0.8; falsada si no.
- **Estabilidad (secundaria):** cualquier NaN = hallazgo contra el código reparado (estrés a
  escala nueva). Predicción: 0 NaN.

**La comparación primaria es UNA** (tendencia de Δ vs escala); todo lo demás es secundario o
descriptivo — sin jardín de senderos.

## Límites declarados (antes de ver datos)

1. **Presupuesto fijo (4000 pasos):** a S3 el modelo queda muy sub-entrenado (152M params vs
   16.4M chars vistos). El gap medido es "a presupuesto igual", NO "a convergencia". Un cruce
   tardío post-convergencia queda fuera del alcance de T4.
2. **LR fijo 3e-4** sin tuning por condición/escala (idéntico a T2A/T3; afecta a las 3
   condiciones de cada escala por igual, pero el óptimo puede diferir por condición).
3. 1 seed en S2/S3; σ₁ de S1 como vara.
4. Un dataset/arquitectura (char-level WT2, GPT mínimo); r/d fijo = 1/6.
5. Extrapolar el resultado (en cualquier dirección) más allá de ~152M/este régimen es
   especulación y se etiquetará como tal.

## Artefactos

- Harness: `t4_scale_sweep.py` (este directorio). Resultados crudos: `t4_results.json`
  (dump incremental tras cada condición; curva, VRAM pico, wall, NaN, params, seed, config).
- Smoke previo: `t4_smoke.json` (20 pasos/condición, valida shapes+OOM antes del run real).
- Entorno: torch 2.11.0+cu128, RTX 4060 8.6GB, mztrain 1.1.0 (post-reparaciones), venv del repo.
