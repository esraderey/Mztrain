# Changelog

Todos los cambios notables en MZTrain se documentan en este archivo.

El formato esta basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-10-09

### Linea base sana por defecto (evidencia: `docs/evidencia/T20-VEREDICTO.md`; T21 y T22 la usan)
- `GPT(..., emb_std=0.02)`: `tok` y `pos` arrancan en N(0, 0,02²). El N(0, 1) de `nn.Embedding` se
  escala (no se vuelve a sortear): no consume RNG, así que a igual semilla los bloques y el estado
  del RNG global son los de 1.4.0 y `tok`/`pos` son exactamente 0,02 × los de 1.4.0.
  `emb_std=None` recupera el init anterior. Motivo: con la cabeza atada, N(0, 1) arranca
  prediciendo el token actual con margen ≈ 0,645·d (BPC inicial ≈ 720 a d=768 frente a ≈ 10,4).
- `ZTrainConfig`: `weight_decay` 0,01 (antes 1e-4); `betas` (0,9, 0,95) y `eps` 1e-8, nuevos y
  pasados por el engine a los tres optimizadores (también al reconstruirlos tras un crecimiento);
  `warmup_steps` 200 (antes 100 y sin uso); `lr_schedule` `"warmup_cosine"` | `"none"`,
  `lr_warmup_floor` 0,1, `lr_final_factor` 0,0 y `lr_warmup_max_fraction` 0,1 (la rampa no ocupa
  más del 10 % del presupuesto; en el régimen medido, 200 de 6000 o de 18 000, no actúa).
  Validación de todos ellos (reales finitos; admite listas y escalares de numpy).
- Calendario de LR dentro del engine, por paso: rampa lineal desde `lr_warmup_floor`·pico y coseno
  hasta `lr_final_factor`·pico en el último paso de `epochs × len(train_loader)`. Solo actúa en
  `train()` sin `scheduler` y con `lr_schedule="warmup_cosine"`. El warmup post-crecimiento de
  rango se compone como factor sobre el LR programado, con la misma secuencia que en 1.4.0.

### Cambios de comportamiento que conviene conocer antes de actualizar
- **El calendario cubre UNA llamada a `train()`.** Otra llamada, también tras `load_checkpoint`,
  empieza rampa y coseno de nuevo sobre su presupuesto: no reanuda el calendario (el paso global
  no viaja en el checkpoint). Al salir de `train()` por cualquier vía (fin, early stopping o
  excepción) el LR vuelve a `learning_rate`, como quedaba en 1.4.0. Para continuar un calendario
  a medias, pasa un scheduler propio.
- **Para reproducir 1.4.0:** `GPT(..., emb_std=None)` y
  `ZTrainConfig(weight_decay=1e-4, betas=(0.9, 0.999), lr_schedule="none")`. Un checkpoint de
  1.4.0 cargado con la configuración por defecto de 1.5.0 conserva sus β y su weight decay solo
  hasta el primer crecimiento de rango, que reconstruye el optimizador con los de la
  configuración: usa la de arriba para continuar esas corridas.
- **Las clases de optimizador usadas sueltas no cambian** (`ZCompressedAdam`,
  `ZAdaptiveOptimizer` y `ZGaLoreOptimizer` conservan (0,9, 0,999) y 1e-4): la línea base sana es
  el valor por defecto del engine y de `GPT`.
- **Bancos, scripts y ejemplos** que construyen `ZTrainConfig` sin fijar estos campos corren
  ahora con β₂ 0,95 y rampa + coseno (`bench/run.py`, `bench/elastic_bench_real.py`,
  `bench/elastic_bench_guard.py`, `examples/example_basic.py`; en
  `scripts/test_zcoder_410m_full.py`, `warmup_steps=5` pasa a actuar). Los resultados guardados en
  `bench/results` y la evidencia T4–T14 se generaron con el régimen anterior.
- **ElasticShape:** las bandas de deriva de `widen_gpt` de la SPEC (≤ 7 % en el toy 48→72) se
  midieron con embeddings N(0, 1). Con `emb_std=0.02` la deriva relativa medida en el init (ruido
  0, 20 semillas) es 0,04–0,13 y supera el 7 % en 12 de 20 (0,04–0,09 a 96→144); no está medida
  en modelos entrenados. Los tests de cota de deriva fijan `emb_std=None`; el resto de la suite
  de ElasticShape corre con el valor por defecto, y un test nuevo vigila que bajo la línea base
  sana la deriva no pase de 0,2 en 10 semillas.

### Qué dice y qué no dice la evidencia
- T20 (13,5M, char-WT103, 18 000 pasos, 3 semillas) mide **bajo** la línea base sana: F★ no gana
  a Ds★ en ninguna semilla (hueco +0,051 BPC) y el sumidero de norma del arco T4–T19 no aparece
  (rms del flujo final 9–10 frente a 273–356; parte constante 13 % frente a 87–98 %). El +0,086
  del montaje anterior se midió en otro corpus (WT-2) y con otra programación: **la diferencia no
  es atribuible a la línea base sana sola**, y qué pieza quita el sumidero queda para la ablación
  E6, pendiente. T21 (forma) y T22 (optimizador espectral) no cierran el hueco. Si tu modelo cabe
  denso, entrena denso.
- Tests: `tests/test_linea_base_sana.py`, con los valores esperados del calendario escritos a
  mano; el test M7 de refactorización aísla su ruta con `lr_schedule="none"`.

### Corregido
- `optimizer_type="adaptive"` y `"galore"`: el primer paso tras reconstruir el optimizador
  (crecimiento de rango, cirugía de ElasticRank o rollback del loss-guard) lanzaba
  `KeyError: 'is_projected'`, también en 1.4.0. El engine migra estados Adam planos y estos
  optimizadores solo creaban sus claves sobre un estado vacío; ahora completan el estado migrado
  (conservan el momentum si el parámetro no se proyecta y lo reinician en el subespacio si se
  proyecta). `compressed_adam` no cambia. Regresión en `tests/test_bug_regressions.py`.
- `optimizer_type="adaptive"` y `"galore"` con `compress_optimizer_states=True` (el valor por
  defecto) y factores U/V proyectados (rango ≥ 128): si el estado estaba comprimido al cambiar la
  topología (contador de pasos del optimizador múltiplo de `compression_interval`, 10),
  `_grow_model_rank` y la compactación de ElasticRank lanzaban `RuntimeError` por tamaños
  incompatibles (la compactación, después de guardar en el sleep bank momentos del tamaño del
  subespacio), y `observe()` de ElasticRank también cuando `rank == in_features`; también en
  1.4.0. `decompress_opt_state` devolvía el momento comprimido del subespacio (p. ej. (8, 128)
  para un factor (256, 128)) sin comprobar su forma; ahora devuelve `None`, como ya hacía con ese
  momento sin comprimir, y el optimizador lo reinicia en el subespacio. De paso, la señal de
  update de ElasticRank deja de depender de si la observación cae en un paso comprimido (U y V
  proyectados aportan 0, como sin comprimir). `compressed_adam` no cambia. Regresión en
  `tests/test_bug_regressions.py`.

## [1.4.0] - 2026-09-26

### MNEMOSYS desde PyPI
- MNEME ya se instala desde PyPI como `mnemosys` (import `mneme`): nuevo extra `pip install "mztrain[mneme]"`
  (`mnemosys>=1.0.1`). `ZActivationCheckpoint` gana un adaptador de backend: fuerza `quantization_type="int8"`
  al registrar (el enrutado automatico de mnemosys descompone tensorialmente las activaciones grandes: error
  relativo ~0.9 medido; con INT8 ~6e-3), tolera `load()` sin kwarg `device` y restaura el device original,
  exige evicción por clave (`delete`/`remove`) y hace una sonda de fidelidad en el primer uso; si el backend no
  cumple, avisa con `RuntimeWarning` y usa el fallback INT8 interno en vez de fugar memoria.
- El store ZSpace pasa a ser **opt-in** (`ZTrainConfig.mneme_activation_store=False` por defecto): con
  mnemosys 1.0.1 cuesta ~0.2-0.6 s por activacion porque persiste cada tensor en disco (MZTrain usa un directorio
  temporal propio, limpiado al salir, y sin cifrado en reposo para ese store efimero; extra acotado a
  `mnemosys<2`). Cierra la fuga de activaciones
  observada con backends sin API de evicción (`example_zcodebert` llegaba a OOM tras 57 min).

### ElasticShape: cirugia reversible y migracion AdamW
- `apply_event` restaura topologia, identidades de parametros, modos y RNG al fallar.
  Las guardas opcionales `max_loss_increase` y `max_kl` usan sondas de perdida/KL en
  nats por token; un rechazo devuelve el optimizer original con `accepted=False`.
  Los candidatos con parametros/estado/logits no finitos se rechazan.
- Ruido funcional por capa: norma Frobenius de las filas nuevas del mapa efectivo
  calibrada contra la norma del mapa original, mediante QR de los factores sin
  reconstruir una matriz densa. Corrige la dependencia de magnitud respecto a `std(U)`.
- Migracion de grupos, opciones, metadatos y estados AdamW/AMSGrad sin alias; fuentes
  por parametro para bloques nuevos, conservacion de parametros excluidos y base de
  warmup. La conversion densa sigue reiniciando momentos por politica explicita.
- Preflight completo; migracion de bias; profundizacion conserva dtype, rangos,
  bias, modos y trainability. Se conserva S/gates fp32 cuando U/V estan en bf16.
- Nuevas regresiones CPU/CUDA y documentacion en `docs/ELASTICSHAPE-SAFETY.md`.

### Peritaje ElasticShape (2026-09-25) y reparaciones
- Auditoria A2 de `shape_ops`/`elastic_shape` (3 peritos + re-disparo del director): 1 alto, 2 medio,
  15 bajo confirmados; rollback, migracion AdamW, correccion SDPA y calibracion QR verificados exactos.
  Expediente en `.tmp/peritaje-elasticshape-20260925/`.
- Reparado (maestranza, cada arreglo con ancla roja->verde): `LrWarmup` libera la base de LR al terminar
  (un warmup posterior ya no vuelve al pico viejo tras un cambio externo del LR); `apply_event` aplica
  los overrides `lr`/`weight_decay` aunque el evento no migre; `migrate_optimizer` rechaza un ledger ya
  consumido; ruido no representable -> rechazo recuperable (`NoiseError`); `probe_targets` validado
  (dtype y rango) antes de la transaccion; `LrWarmup.floor` en [0, 1]; indices, dimensiones y `new_shape`
  deben ser enteros; `pad_adam_entry` usa la lista explicita de momentos; la correccion SDPA escala solo
  las filas q viejas (el presupuesto de ruido de qkv vuelve a ser exacto); `widen_layernorm` compensa
  tambien `eps` (`eps*d/d'`); `dense_to_factorized` avisa en fp16/bf16 (S queda en ese dtype).
- Forja: `GPT.forward` emite `RuntimeWarning` si hay migracion de optimizer pendiente y el gradiente esta
  habilitado (entrenar con el optimizer anterior ya no es silencioso); inferencia y sondas no avisan.
- Residuos cerrados (2026-09-26, forja + revisor ciego): `ZFactorizedLinear.reconstruct_weight` y la
  variante sparse devuelven la W EFECTIVA (incluye `wake_gate`): la exportacion a denso y `refactorize`
  reproducen el forward tambien durante un mini-warmup de ElasticRank; `grow_rank(preserve_weights=False)`
  resetea gates/mask porque la re-SVD cambia la base; el forward tolera U/V bf16 con S fp32 sin autocast y
  `dense_to_factorized` conserva S en fp32 tambien en bf16/fp16; `LrWarmup` aplica la regla 'la escritura
  externa gana' (un warmup abandonado o reanudado de checkpoint ya no devuelve al pico viejo si el LR
  cambio fuera); `GPT(..., ln_eps=)`/`Block(..., eps=)` y `model.ln_eps` hacen reconstruible el `eps`
  compensado de LayerNorm. Import muerto `Tuple` eliminado en `layers.py`.

### Compatibilidad
- Cambia el significado de `noise_scale` positivo; no se heredan los claims de reloj
  de T8-T14 sin repetir experimentos. LayerNorm y weight decay conservan la politica
  previa: sus alternativas matematicas requieren una comparacion separada.
- Esta revision de fuentes requiere un nuevo sello firmado antes de publicarse;
  `SEAL.json` y `MANIFEST.sha256` siguen describiendo el snapshot anterior.

### Evidencia
- **T9-T14: seis experimentos preregistrados sobre ElasticShape** (85 corridas). T9 cierra las
  tres banderas que T8 dejo abiertas: escala (endpoint 38.8M, ratio 0.476), cirugias encadenadas
  (0.496, mejor que el salto unico) y bf16 (0.689 con paridad de calidad). T10-T12 atacan el
  claim por su lado mas debil (tasa de aprendizaje afinada) y lo dejan en **0.542**. T13 mide el
  promediado de pesos. T14 descarta que el corpus fuera el factor limitante.
- Preregistros, veredictos, datos crudos y **banco reproducible completo** en `docs/evidencia/`.
- `docs/RFC-ELASTICSHAPE-1.md`: RFC tecnico con el fundamento matematico, los diez defectos que
  encontraron las auditorias ciegas, los resultados de T8 a T14 y las amenazas a la validez.

### Correcciones metodologicas
- La varianza a semilla **fija** (~0.4 BPC) supera a la sigma_1=0.068 entre semillas que el arco
  asumia: la tarea es bimodal. Documentado con la evidencia que ya estaba en T6.
- No se deben comparar BPC finales de corridas sin converger. El banco incorpora la guarda
  (`docs/evidencia/banco-t9-t14/convergencia.py`).
- Metrica de evaluacion unificada en todo el arco (habia dos implementaciones divergentes);
  ancla de no-regresion bit a bit y re-ejecucion de la unica celda afectada.

### Notas
- La ampliacion de evidencia T9-T14 no modifico `src/mztrain`; la revision de
  cirugia reversible descrita arriba si introduce cambios de implementacion.

## [1.3.2] - 2026-08-14

### Corregido
- Metadata: correo del proyecto (msc.framework@gmail.com) para ambos autores en pyproject
  (el segundo autor tenia un placeholder @example.com). Barrido de privacidad del arbol
  publicado: sin correos personales, rutas locales ni credenciales.

## [1.3.1] - 2026-08-14

### Corregido
- Sello: `mztrain.egg-info` y `.claude` excluidos del manifiesto (artefactos de build y
  config local por-maquina rompian `verify` en clones git y sdists en falso).
- README: la verificacion canonica es el repositorio git; el sdist de PyPI es un
  subconjunto de instalacion (documentado); URL real del repo; conteo de tests.
- Higiene de publicacion: `.gitattributes` (bytes exactos en clones — el sello sobrevive
  cualquier plataforma), checkpoints `.pt` y `data/` fuera del control de versiones.

## [1.3.0] - 2026-08-14

### Agregado
- **ElasticShape v1** (`mztrain.shape_ops` + `mztrain.elastic_shape`): crecimiento de forma
  en espacio factorizado — conversion densa→factorizada exacta (SVD), ensanchado/profundizado
  con cirugia que preserva la funcion, migracion completa del estado Adam, correccion de
  escala SDPA, compensacion de varianza de LayerNorm, schedule y warmup. Cada modulo con
  doble revision adversarial independiente y banco aislado (40 tests).
  Claim validado con preregistro (T8, `docs/evidencia/`): el morph alcanza la calidad del
  from-scratch en ~54% del reloj (escala 11M-equiv, 3 seeds).
- `docs/evidencia/`: serie empirica preregistrada T0-T8 completa (preregistros, veredictos
  formales y JSONs crudos) dentro del arbol sellado.
- Firma Ed25519 obligatoria con lista de claves de confianza en `scripts/seal.py`.

### Cambiado
- README reescrito con los resultados medidos (regimen de uso, peaje de calidad iso-parametro
  ~10% estable con la escala, acantilado WDDM, velocidades bf16, limitaciones conocidas y
  configuraciones a evitar). Version unificada en pyproject/setup/__init__.
- Extra `mneme` retirado del empaquetado PyPI (instalacion manual desde su repositorio;
  evita resolver un paquete homonimo ajeno).

### Corregido
- Los 5 hallazgos ALTO del peritaje 2026-08-10 y todo lo escalado (18+ anclas de regresion;
  suite 273 → 347 tests). Detalle en `docs/evidencia/` y PRIOR_ART.

## [1.0.0] - 2025-01-15

### Agregado

#### Capas Factorizadas
- `ZFactorizedLinear`: Capa linear que entrena factores SVD (U, S, V) directamente
  - Inicializacion SVD, Kaiming, random, y desde pesos existentes
  - Crecimiento progresivo de rango con preservacion de pesos
  - Reconstruccion de pesos completos para exportacion
  - Metricas de compresion y error de reconstruccion
- `ZFactorizedAttention`: Multi-Head Attention con proyecciones Q,K,V,O factorizadas
- `ZFactorizedTransformerBlock`: Bloque Transformer completo factorizado

#### Motor de Entrenamiento
- `ZTrainEngine`: Orquestador principal del entrenamiento
  - Conversion automatica de nn.Linear a ZFactorizedLinear
  - Early stopping con restauracion del mejor modelo
  - Sistema de callbacks para monitoreo
  - Soporte AMP (mixed precision) en GPU
  - Guardado y carga de checkpoints

#### Optimizer
- `ZCompressedAdam`: Adam con estados comprimidos INT8
  - Compresion periodica de momentos (m, v)
  - 75% de ahorro en memoria del optimizer
  - Estadisticas de uso de memoria

#### Compresion de Gradientes
- `ZGradientCompressor`: Multiples estrategias
  - TOP_K: Mantener top-K gradientes por magnitud
  - 1-BIT: SignSGD con escalado por media
  - INT8: Cuantizacion INT8 con reconstruccion
  - SVD: Compresion de bajo rango para gradientes matriciales
  - Error feedback para convergencia garantizada

#### Entrenamiento Progresivo
- `ZRankScheduler`: 5 estrategias de crecimiento de rango
  - CONSTANT, LINEAR, EXPONENTIAL, COSINE, ADAPTIVE
  - Crecimiento adaptativo basado en convergencia del loss

#### Activation Checkpointing
- `ZActivationCheckpoint`: Checkpointing comprimido via MNEME/INT8
  - Compresion durante forward, descompresion durante backward
  - Fallback INT8 cuando MNEME no esta disponible
- `z_checkpoint()`: Wrapper conveniente para checkpointing

#### Utilidades
- `factorize_existing_model()`: Conversion de modelos existentes
- `estimate_memory_savings()`: Estimacion sin modificar el modelo

#### Infraestructura
- Estructura modular (7 modulos especializados)
- pyproject.toml con configuracion completa de herramientas
- Suite de tests con cobertura >85%
- CI/CD con GitHub Actions (quality, tests, security, build)
- Documentacion completa (README, CHANGELOG, CONTRIBUTING)
- Ejemplos de uso (basico, transformer, estimacion)

### Integracion con MNEME
- Compresion avanzada de activaciones via ZSpace
- Fallback automatico a INT8 sin MNEME
- Compatibilidad con MNEME v2.0+
