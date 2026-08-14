# SPEC — ElasticShape v1 (crecimiento de forma en espacio factorizado)

**Fecha:** 2026-08-11 · **Riesgo:** R2 · **Proceso por módulo (contrato del usuario):**
implementar → aislar (banco propio, fuera del árbol sellado) → probar (tests del módulo) →
**revisar 2 veces (ciegas, independientes, lentes distintas)** → integrar solo tras certificar.
**Destino final:** `src/mztrain/shape_ops.py` (+ controller) en v1.3.0, off-by-default.
**Workspace:** `scratchpad/elasticshape/` (paquete `mzshape`), importa mztrain del venv.

## Motivación (evidencia del arco)

- Temprano/pequeño gana el DENSO estrecho (T4/T6, 3 escalas, preregistrado).
- Grande solo cabe FACTORIZADO (T5: techo denso ~280M vs fact ~1B-equiv en bf16, T6).
- La cirugía con migración de estado Adam ya existe y está blindada (A2/A3 + T6-anneal:
  216 migraciones sin fallo; T7-rebalance con rescale de estado verificado).
- Literatura (recuerdo, verificable): crecimiento progresivo (stacking/Net2Net/LiGO) reporta
  30-50% de ahorro de cómputo a calidad igual. Nadie lo hace EN espacio factorizado.

**El schedule objetivo v1:** denso-estrecho → (conversión exacta) → factorizado-ancho, con la
mayoría de tokens procesados en la fase barata.

## La matemática (qué se preserva EXACTO y qué no — sin autoengaño)

1. **Ensanchar una capa factorizada es trivialmente exacto A NIVEL DE CAPA:**
   `V' = [V | 0]` (r×in'): las dims nuevas de entrada se ignoran. `U' = [U ; 0]` (out'×r):
   las dims nuevas de salida emiten exactamente 0. S y r intactos. y_viejo idéntico bit a bit
   (misma suma, mismos términos); y_nuevo = 0.
2. **Gradientes muertos si todo es 0:** si las dims nuevas emiten 0 y nadie las lee, no llega
   gradiente. Remedio v1: ruido ε en las FILAS nuevas de U (`noise_scale` relativo, default
   1e-3) → deriva funcional acotada por ε y medible; el gradiente fluye desde el primer paso.
3. **Layouts estructurados (qkv fusionado):** la salida es [q|k|v]; ensanchar d inserta dims
   DENTRO de cada bloque, no al final. Primitiva con `out_map`/`in_map` (índices donde las
   dims viejas viven en el layout nuevo). Sin el mapa, el zero-pad ingenuo corrompe el reshape
   por cabezas — el bug que los tests deben clavar.
4. **Escala de SDPA:** `scaled_dot_product_attention` divide por √(head_dim'). Al crecer
   head_dim (convención del arco: heads fijos), los scores viejos se reescalarían por
   √(d_h/d_h'). Corrección EXACTA: multiplicar las SALIDAS del bloque q — filas de U **y sus
   entradas de bias** (q_i = U_i·h + b_i; solo-U rompe la identidad con bias≠0, hallazgo
   G4-A) — por √(d_h'/d_h) en un solo lado (q O k, nunca ambos: bilinealidad). Primitiva
   `scale_output_rows`. Con k/q nuevos en 0, los productos punto no cambian → scores idénticos.
5. **LayerNorm NO preserva a nivel de red:** LN sobre [x, 0] cambia μ y σ por muestra. La
   dilución de varianza (factor √(d'/d), primer orden) se cancela con `variance_compensation`
   (γ_viejo·√(d/d')); el residuo del shift de media es TAMBIÉN de primer orden en μ (corrección
   G4-A a la versión inicial de esta spec, que decía "segundo orden"). **Deriva MEDIDA con
   noise=0 (bandas de referencia, G4-A/G4-B):** 3.9–5.3% en toy d=48→72 (7 seeds, nunca ≤2%),
   1.3–4.1% en configs mayores al mismo ratio 1.5×, ~10% en growth 4×; decrece con d (el 2%
   original era aspiracional y solo plausible a escala grande — irreconciliado entonces con el
   test, ahora documentado). La deriva se MIDE con probe en cada growth y el warmup de LR
   absorbe el transitorio de entrenamiento (no "borra" la deriva). Umbral de test toy: 7%
   (banda medida + margen de seed).
6. **Profundidad es EXACTA:** un bloque residual nuevo con U=0 **y bias=0** en `proj` y
   `fc2` es la identidad (x + 0; con bias≠0 sería x + b_proj + b_fc2, hallazgo G4-A).
   Primitiva `zero_block_outputs`: el bloque nace muerto-exacto y despierta por gradiente
   (las otras capas del bloque nacen normales).
7. **Denso→factorizado EXACTO:** `dense_to_factorized(linear)` embebe W con SVD completa a
   rango m = min(out,in) — exacto salvo numérico, sin EPSI (construcción directa de U/S/V).
   **Corrección G4-A:** r > m es INEXPRESABLE en la misma forma (el constructor de
   ZFactorizedLinear clampa rank a min(in,out) — verificado ejecutando); el "banco de
   crecimiento" NO vive aquí sino en la COMPOSICIÓN: convertir exacto a r=m → ensanchar la
   forma (min dims sube) → crecer rango con la maquinaria existente (`grow_rank`). La
   primitiva se queda mínima a propósito.
8. **Estado Adam:** pad del estado (m,v → zero-pad en filas/cols nuevas; `step` conservado
   POR VALOR — clonado, no alias: AdamW lo muta in-place; hallazgo G4-B). **Corrección G4-A
   al claim "recién nacido":** con step alto y m=v=0, las correcciones de sesgo dan un primer
   update de hasta lr·(1−β1)/√(1−β2) ≈ 3.16·lr (derivado y simulado) en vez de 1.0·lr —
   acotado, no divergente, transitorio; el warmup post-growth del controller (M2) lo absorbe.
   `step` es escalar por tensor en torch: limitación estructural heredada, no creada.

## Módulos y contratos

### M1 — `mzshape/shape_ops.py` (este turno)
Primitivas puras (sin engine, sin config): `widen_factorized_linear`, `widen_embedding`,
`widen_layernorm`, `scale_output_rows`, `zero_block_outputs` (anula U y bias de proj/fc2),
`dense_to_factorized`, `pad_state_tensor` + `pad_adam_entry`.
- Restricciones: no tocar mztrain; solo leer sus clases. Sin abstracciones especulativas.
- **Criterios ejecutables (tests del módulo):**
  - exactitud del camino viejo en widen con noise=0 a tol 1e-6 en rutas con GEMM/SDPA
    (la reasociación FP impide atol=0 — medido: diffs ~2e-7 incluso en CPU; hallazgo G4-A);
    `torch.equal` estricto donde no media GEMM (salidas nuevas = 0, bloque identidad,
    fila de padding, gates); incluye capa con `out_map` estilo qkv + reshape por cabezas +
    SDPA con corrección de escala, con y sin bias;
  - deriva ≤ noise_scale·c en widen con ruido (cota medida);
  - `dense_to_factorized` reconstruye W con error ≤ 1e-5 relativo (fp32) para r ≥ min-dim,
    y los S sobrantes son 0;
  - bloque identidad: salida del bloque == entrada (atol 0) antes de entrenar;
  - pad de estado Adam: shapes correctas, `step` preservado, zonas nuevas en 0, viejas intactas;
  - un mini-train de 30 pasos post-growth converge sin NaN y con las dims nuevas recibiendo
    gradiente ≠ 0 (con noise>0).
- **Gates:** G0 compile/import · G1 ruff (E9,F,B,PLE) · G3 pytest del módulo · G4 ×2 ciego
  (lente A: corrección matemática de las primitivas contra esta SPEC; lente B: contratos de
  estado/aliasing/dtype/device y casos borde de índices).

### M2 — `mzshape/controller.py` (siguiente)
Schedule declarativo {paso: forma}, growth end-to-end de un GPT del harness (dense→fact y
fact→fact-ancho), probe de deriva, rebuild del optimizer con transplante+pad, warmup hook.
Mismo proceso y doble G4.

### M3 — T8 (validación empírica preregistrada)
Morph 11M-denso→57M-fact vs 57M-fact-desde-cero e iso-reloj vs denso: **muerte preregistrada:**
si el morph no alcanza el BPC del from-scratch en ≤0.7× del reloj, ElasticShape v1 se rechaza
(el mecanismo quedará como primitivas auditadas sin claim).

### M4 — Integración a `src/mztrain` (solo tras M1-M3 verdes + 2×G4 cada uno)
Flags off-by-default; docs; anclas de regresión al árbol sellado; re-sello; v1.3.0.

## Fuera de alcance v1
Crecer heads (solo head_dim), shrink de ancho, DDP, kernels custom, LN-corrección exacta
(documentada como imposible estática), auto-scheduling del growth (schedule manual v1).
