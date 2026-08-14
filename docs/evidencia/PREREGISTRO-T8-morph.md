# PREREGISTRO — T8: el juicio de ElasticShape v1 (morph vs from-scratch)

**Fecha:** 2026-08-12, ANTES de correr. **Precondición:** M1 y M2 certificados (doble G4 c/u).
**Protocolo base:** char-WikiText-2 del arco (batch 16, seq 256, AdamW wd 0.01, fp32 — bf16
queda fuera: la bandera F-grande-bf16 de T6 sigue abierta y no debe contaminar este juicio).

## Pregunta

¿Entrenar chico-denso y CRECER al factorizado-ancho alcanza la calidad del from-scratch en
sustancialmente menos reloj? Es el claim central de ElasticShape; T4/T6 aportan la premisa
(el chico-denso aprende más calidad por segundo temprano).

## Condiciones (3 seeds {0,1,2} cada una)

- **R (referencia):** F_192@d384 (GPT del arco, `fact_lin(192)`, L=6, h=6) desde cero,
  4000 pasos. Q := media de su `val_bpc_full` final; T_R := media de su reloj de TRAIN puro
  (los evals no cuentan el tiempo, ni en R ni en M). Sanity: T6 midió esta condición en
  2.83-2.94 BPC / ~196 s — debe reproducirse aproximadamente.
- **M (morph):** fase 1: denso d=192 (la ganadora de T4) por 2 000 pasos; cirugía M2 en un
  solo evento (`factorize=True, new_d=384`, noise 1e-3; optimizer reset documentado en la
  conversión + LrWarmup 200 pasos, floor 0.1); fase 2: hasta 6 000 pasos como F_192@d384,
  `eval_full` cada 250 pasos con el reloj de train acumulado registrado. El schedule
  (2000/384/…) se fija AQUÍ, sin barrido — es v1, no el óptimo.
- T_M := primer reloj-de-train (fase1 + cirugía + fase2) donde `val_bpc_full` ≤ Q.

## Regla de decisión (fijada antes)

- **CLAIM v1 CONFIRMADO:** media(T_M) ≤ 0.7 · media(T_R) con los 3 seeds alcanzando Q.
- **MUERTE de v1:** media(T_M) > 0.7·media(T_R), o ≥2 seeds no alcanzan Q dentro del
  presupuesto de fase 2. → ElasticShape v1 se RECHAZA como claim (las primitivas certificadas
  quedan como infraestructura sin reivindicación funcional; así se documenta).
- Zona gris (1 seed no alcanza, o 0.7 < ratio ≤ 0.85): INCONCLUSO — se reporta sin claim.
- **Secundarios (descriptivos):** BPC del morph a reloj = T_R (calidad a tiempo igual);
  deriva de la cirugía (probe); BPC de fase 1 al momento de crecer; curvas completas.

## Predicción del autor (honesta)

Ratio media(T_M)/media(T_R) ∈ [0.6, 0.9]: creo que el morph gana algo pero no sé si cruza el
0.7 — la deriva del 4-5% de la cirugía y el arranque frío del optimizer en la conversión
pueden comerse la ventaja. Confianza baja. La banda del annealing (T6-C2, beneficio nulo)
me hace especialmente escéptico de mi propio entusiasmo aquí.

## Límites declarados

Un solo schedule (sin tuning); una escala (11M-equiv endpoint — barato y con líneas base
sólidas; el claim a escala mayor requeriría T9); fp32; 3 seeds; el endpoint F_192@384 no es
iso-parámetro con nada (es el MISMO modelo en ambas condiciones — la comparación es limpia
por construcción). Artefacto: `t8_results.json` (dump atómico incremental) + smoke previo.
