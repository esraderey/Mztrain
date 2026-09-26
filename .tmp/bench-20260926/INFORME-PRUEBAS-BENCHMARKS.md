# Pruebas y benchmarks tras las reparaciones — 2026-09-26

Arbol: `D:\mztrain` con la revision ElasticShape 2026-09-07 + lotes de reparacion 2026-09-25/26 (sin
commitear). Runtime: Python 3.13.2, torch 2.11.0+cu128, RTX 4060. Logs y JSON en esta carpeta.

## Pruebas

| Comando | Resultado |
|---|---|
| `pytest tests` sin gate de cobertura | **413 passed**, 3 warnings, 9.3 s |
| `pytest tests` oficial (addopts del pyproject, con `--cov-fail-under=85`) | 413 passed; **cobertura 82.99 %** → el gate falla (85 %). Deuda preexistente: era 81.71 % antes del peritaje, 79.65 % en agosto. No hay tests rojos. |
| Banco de maestranza (suite completa, ambito con `layers.py`) | verde, hash `080c70e9a97a3885` |

## Benchmarks del repositorio (`bench/`)

Resultados historicos respaldados en `results-backup/` (mayo 2026) antes de correr; los nuevos en
`new-results/` y en `bench/results/`. Los JSON de mayo son de un codigo anterior (sin loss-guard de
ElasticRank, torch 2.6 en `bench.run`), asi que las comparaciones con mayo son orientativas; la comparacion
que atribuye o descarta efectos de las reparaciones es el **A/B pre/post** de abajo.

| Benchmark | Que mide | Resultado 2026-09-26 | Frente a mayo |
|---|---|---|---|
| `bench.run --label post_peritaje_20260926` (MNIST MLP, 3 semillas x 20 epocas, CUDA, rank 16→64) | regresion general del engine + sanity INT8 de GaLore/Adaptive | val acc 92.27 % ± 0.75; val loss 0.387 ± 0.036; rank crece 100 %; INT8 rel_err 0.0075 (407 valores unicos) | `bench.compare fixed_v2 → post`: acc −0.2 %, val loss +9.6 %, train loss −11.6 %, wall −6.4 %. INT8 identico. Dentro del ruido entre semillas (σ 0.75-1.0 % acc); torch distinto (2.6 → 2.11). |
| `bench.elastic_bench` (sintetico rango 6, escenario from_scratch, CPU) | control de memoria de ElasticRank | baseline/v1/v2: 16 944 params activos sin reduccion; R² 0.99977 / 0.9999 / 0.99995; v2: 0 dormidas, **19 rollbacks** del loss-guard | Mayo: v2 dormia 25 direcciones (−52 % params). **A/B con el arbol pre-reparacion: identico bit a bit (0 hojas distintas, mismos 19 rollbacks).** El cambio es anterior a este lote (loss-guard introducido tras mayo), no efecto de las reparaciones. |
| `bench.elastic_bench_guard` (loss-guard OFF/ON, SAFE/RISKY) | que el guard convierta una compactacion destructiva en no-op | SAFE: OFF/ON R² 0.99762/0.99785; RISKY: OFF R² 0.860 (75 % params) vs ON R² 0.987 (32 dormidas / 16 rollbacks) | Metricas iguales a mayo (±0.2 % R²). Wall +65-76 % (5.3→8.9 s): CPU, misma maquina con carga distinta; no comparable. |
| `bench.elastic_bench_real` (ZCodeBERT 4 capas MLM, rank 48 sobre-aprovisionado, CPU) | redundancia EMERGENTE en un transformer real | ningun brazo compacta (como en mayo); val MLM acc 0.224 / 0.208 (v1); loss 3.71 / 3.74 | igual que mayo (±3 % acc, ±0.2 % loss) |
| `bench.governor_gpu_check` (VRAM governor en GPU real) | sensor, lazo cerrado, OOM real, e2e | **15/15 PASS** | igual que mayo (15/15) |

Conclusion de esta seccion: ninguna regresion atribuible a las reparaciones. La unica diferencia
llamativa frente a mayo (v2 de `elastic_bench`) se reproduce exactamente con el codigo previo al lote.

## Benchmark preregistrado de ElasticShape (`.tmp/elasticshape-evidence-20260907/PROTOCOL.md`)

Es la comparacion que SAFETY.md declara pendiente ("para atribuir calidad o ahorro de reloj a esta
revision hay que repetir una comparacion controlada"). Se corre en `elasticshape-rerun-20260926/` (copia
del protocolo y del script; el script se niega a sobrescribir `results.json` y hashea el protocolo y
`src/mztrain/elastic_shape.py` para provenance). Referencia 2026-09-07 (codigo pre-reparacion):
3/3 cirugias aceptadas, 3/3 alcanzan Q, ratio geometrico M/R = 0.6985 (ahorro 30.15 %).

Corrida 2026-09-26 completa (6 entrenamientos, 525 s medidos, sin errores). `provenance_hashes_match: true`
(el script hashea el `elastic_shape.py` reparado y el protocolo intacto). **Regla preregistrada: CUMPLIDA.**

| Semilla | Q (BPC) pre = post | R hasta Q (s) pre → post | M hasta Q (s) pre → post | Ahorro pre → post | Cirugia: Δloss / KL (nats) | Ahorro conservador | M a reloj de R (BPC) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | 3.064610 | 80.14 → 86.03 | 59.67 → 56.82 | 25.5 % → **34.0 %** | +0.0058 / 0.0038 | 30.0 % | 2.716 vs 3.065 |
| 29 | 3.079263 | 79.51 → 73.40 | 54.05 → 49.53 | 32.0 % → **32.5 %** | +0.0024 / 0.0016 | 28.1 % | 2.783 vs 3.079 |
| 43 | 3.119238 | 80.10 → 74.43 | 53.94 → 49.03 | 32.7 % → **34.1 %** | +0.0039 / 0.0028 | 29.4 % | 2.798 vs 3.119 |

- 3/3 cirugias aceptadas bajo las guardas fijadas (max_loss_increase 0.05, max_kl 0.05; observado ≤0.006 /
  ≤0.004), 3/3 M alcanzan Q ya con la arquitectura final (pasos 3750/3500/3500, como el 09-07), 3/3 ratios < 1.
- **Ratio geometrico M/R = 0.6647 (ahorro 33.5 %)** frente a 0.6985 (30.2 %) el 09-07. Medias: R 77.95 s, M 51.80 s.
- **Los objetivos Q son identicos a seis decimales en las tres semillas**: el brazo R (GPT factorizado, fp32,
  TF32 off) es bit a bit reproducible tras cambiar `layers.py`, como exigia el contrato de forja (con gate=1 y
  dtype unico la reparacion es un no-op numerico). Los BPC de M en el cruce difieren en la 3a cifra
  (3.00443 → 3.00559, etc.): esperable, el ruido de las filas q ya no se escala por c y `LrWarmup`/`eps`
  cambiaron; la deriva de cirugia queda igual o menor.
- Los tiempos absolutos por semilla se mueven ±8 % entre sesiones (deriva termica documentada en T12); solo
  el ratio dentro de la misma sesion es comparable, y es lo que el protocolo mide.
- Alcance identico al del 09-07: tres semillas, una escala (1.4M params), sin significacion poblacional. El
  secundario descriptivo (M a igual reloj que R: −0.30 a −0.35 BPC) no es criterio de decision.

## Veredicto
Pruebas: 413/413 verdes; cobertura 82.99 % bajo el gate de 85 % (deuda preexistente, subiendo). Benchmarks del
repo: sin regresion atribuible a las reparaciones (A/B pre/post bit a bit en ElasticRank; governor 15/15).
ElasticShape: la revalidacion empirica que SAFETY.md dejaba pendiente se ha hecho y la regla preregistrada se
cumple con el codigo reparado (ahorro geometrico 33.5 %). Sigue pendiente del usuario: commit + re-sello.
