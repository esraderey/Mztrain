# Changelog

Todos los cambios notables en MZTrain se documentan en este archivo.

El formato esta basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
