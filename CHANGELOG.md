# Changelog

Todos los cambios notables en MZTrain se documentan en este archivo.

El formato esta basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
