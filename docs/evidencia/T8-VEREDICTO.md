# T8 — Veredicto: ElasticShape v1 CONFIRMADO (regla preregistrada)

**Fecha:** 2026-08-12. **Preregistro:** PREREGISTRO-T8-morph.md (previo a los datos).
**Datos:** t8_results.json (6 runs, 0 NaN). fp32, protocolo del arco, reloj de TRAIN puro.

## Resultados

**R (F_192@384 desde cero, 4000 pasos, 3 seeds):** BPC {2.9507, 2.8308, 2.9248} →
Q = 2.9021; T_R = 194.1 s. (Sanity vs T6: banda 2.83-2.94 / ~196 s → reproducido ✓.)

**M (morph: 2000 pasos denso-192 → cirugía M2 → fase ancha):**

| Seed | Fase 1 BPC | Deriva cirugía | Post-cirugía | T_M (alcanza Q) | BPC a reloj=T_R | BPC final (8000 pasos eq.) |
|---|---|---|---|---|---|---|
| 0 | 3.5592 | 0.24% | 3.5601 | 105.4 s | 2.3394 | 2.0867 |
| 1 | 3.5488 | 1.40% | 3.5503 | 104.7 s | 2.3708 | 2.1231 |
| 2 | 3.5384 | 0.79% | 3.5397 | 104.9 s | 2.3526 | 2.1154 |

## Regla aplicada

- **3/3 seeds alcanzan Q** en T_M = {105.4, 104.7, 104.9} s → media 105.0 s.
- **ratio = T_M/T_R = 0.541 ≤ 0.7 → CLAIM v1 CONFIRMADO.**
- Predicción del autor (ratio 0.6-0.9, confianza baja): el resultado superó el extremo
  optimista de la predicción.

## Los tres hallazgos

1. **El morph alcanza la calidad del from-scratch en el 54% del reloj** — consistencia casi
   perfecta entre seeds (±0.4 s).
2. **A reloj IGUAL (194 s), el morph rinde 2.34-2.37 vs 2.90 del from-scratch** — no solo
   llega antes: en todo punto posterior a ~105 s va estrictamente por delante. La premisa
   T4/T6 (el chico-denso compra más calidad por segundo temprano) se convierte en mecanismo
   de entrenamiento explotable.
3. **La cirugía es casi gratis en calidad sobre modelos ENTRENADOS:** deriva 0.24-1.4% de
   logits (muy por debajo de la banda toy 3.9-5.3% medida por G4 sobre modelos random-init)
   y BPC pre/post cirugía prácticamente idéntico (3.559→3.560). La maquinaria certificada de
   M1/M2 (mapas exactos, corrección SDPA + estado, compensación LN, migración Adam) hace
   exactamente lo que la matemática dice, en el caso real.

## Alcance honesto

Válido para: esta escala (endpoint 7.6M params, 11M-equiv), este schedule único (2000 pasos
pequeños, un solo growth 192→384), fp32, char-WT2, 3 seeds. El claim NO está probado a
escala mayor (T9 con endpoint 57M+ sería el siguiente), ni con múltiples growths encadenados
en entrenamiento real, ni en bf16 (bandera F-grande-bf16 sigue abierta). El endpoint es el
MISMO modelo en ambas condiciones — la comparación es limpia por construcción.

## Estado de ElasticShape v1

M1 certificado (21 tests) · M2 certificado (19 tests) · **T8: claim confirmado por regla
preregistrada**. Pendiente M4: integración a `src/mztrain` (v1.3.0), evidencia al repo
sellado, re-sello y anclas de regresión.
