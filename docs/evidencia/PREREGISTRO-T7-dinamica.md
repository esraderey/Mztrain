# PREREGISTRO — T7: dinámica de la parametrización (los mecanismos "raros")

**Fecha:** 2026-08-11, ANTES de correr. **Origen:** revisión dirigida del núcleo tras 4 anomalías
sin explicar (ruido de seeds de F ×10-40, inestabilidad a LR alto, bandera bf16@S2, gap ~10%).

## Mecanismos bajo prueba (derivados y anclados en código)

- **M-A (triple decay):** AdamW decoupled multiplica cada factor por (1−ηλ) → W=U·S·V decae
  (1−ηλ)³ ≈ 3× el denso; y decay sobre factores ≈ presión de norma nuclear (muerte de
  direcciones → rango efectivo < nominal). Anclas: `optimizer.py:221` (`p.data.mul_(1-lr*wd)`
  uniforme), `engine.py:202-222` (un solo param group, sin exclusiones), harness T4/T6
  (torch AdamW, wd default 0.01 uniforme en ambas condiciones).
- **M-B (deriva de gauge):** libertad (cU_i, S_i/cd, dV_i) = direcciones planas; Adam con v≈0
  ahí da pasos ~lr (random walk por ruido de minibatch/redondeo) → desbalance progresivo de
  normas → mal condicionamiento; bf16 lo alimenta más.

## Condiciones (S1, fp32, LR 3e-4, 4000 pasos, batch 16, protocolo T4; 3 seeds salvo indicado)

1. `F64_wd0` — F_64 con weight_decay=0
2. `Dsmall_wd0` — control denso con wd=0
3. `F64_noSdecay` — wd=0.01 en todo SALVO S (param groups; práctica estándar de params de escala)
4. `F64_rebal` — wd=0.01 uniforme + re-balanceo de gauge cada 250 pasos (normaliza columnas de
   U/filas de V, absorbe normas en S — W EXACTAMENTE preservada; estado Adam re-escalado
   consistentemente: m×c, v×c² por el rescale de cada param)
5. `F64_wd001_instr` — 1 seed, réplica del baseline T4 con instrumentación (referencia)

**Instrumentación (cada 100 pasos, todas las capas factorizadas):** rango efectivo de S
(participation ratio (Σs²)²/Σs⁴, media entre capas), |S|min/máx, desbalance de gauge
(máx_i de ||U_i||/||V_i|| y su deriva), norma de W reconstruida por capa (muestra).

**Líneas base reutilizadas (T4, wd=0.01):** F_64 3.3221±0.068, D_small 3.0273±0.005 → Δ̄=0.295.

## Reglas de decisión (fijadas antes)

- **M-A confirmado (contribuye al gap):** gap_wd0 = mean(F64_wd0) − mean(Dsmall_wd0) ≤ 0.8·Δ̄
  (=0.236). **Muerto:** gap_wd0 ≥ 0.9·Δ̄ (=0.265). Intermedio: inconcluso.
- **Predicción instrumental dura de M-A:** rango efectivo final con wd=0.01 < 64 de forma clara
  (≥2 direcciones muertas de media) y ≈64 con wd=0. Si el rango efectivo NO cae con wd=0.01,
  la vía "muerte de direcciones" de M-A queda muerta aunque el gap se mueva.
- **noSdecay:** recupera ≥⅔ de lo que recupere wd0 (si wd0 recupera algo) → el culpable es el
  decay sobre S específicamente.
- **M-B confirmado:** σ_seeds(F64_rebal) ≤ ½·σ_seeds(F64 T4=0.068) y/o mean(F64_rebal) mejora
  ≥2σ. **Muerto:** σ y media indistinguibles del baseline.
- Predicción del autor: M-A confirmado PARCIAL (mueve 0.03-0.08 BPC del gap, no todo), muerte de
  direcciones visible; M-B confirmado en σ (el re-balanceo estabiliza) con mejora de media
  pequeña o nula. El gap NO desaparece (la mayor parte sigue siendo capacidad del cuello de
  rango) — pero el "misterio raro" (ruido/inestabilidad/bf16) quedaría explicado y arreglable.

## Límites

S1 solamente; LR único 3e-4 (la interacción con LR alto se infiere, no se barre); rebalance cada
250 sin barrido de frecuencia; fp32 (la conexión bf16 se prueba en T6b si M-B confirma).
Artefacto: `t7_results.json` incremental atómico; harness `t7_dynamics.py` (base T4 + cirugías).
