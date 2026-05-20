# Arquitectura de MZTrain

## Principio Fundamental

MZTrain entrena modelos en espacio comprimido usando factorizacion SVD.

## Modulos

### config.py
Configuraciones y enumeraciones (ZTrainConfig, RankSchedule, GradientCompression).

### layers.py
Capas factorizadas que son el corazon del sistema:
- ZFactorizedLinear: W = U @ diag(S) @ V
- ZFactorizedAttention: Atencion multi-cabeza factorizada
- ZFactorizedTransformerBlock: Bloque transformer completo

### engine.py
Motor principal que orquesta todo el entrenamiento.

### optimizer.py
ZCompressedAdam: Adam con estados comprimidos INT8 (75% ahorro).

### gradient.py
ZGradientCompressor: TOP_K, 1-bit, INT8, SVD con error feedback.

### scheduler.py
ZRankScheduler: Crecimiento progresivo de rango (5 estrategias).

### checkpoint.py
ZActivationCheckpoint: Compresion de activaciones durante forward.

### utils.py
Utilidades: factorize_existing_model(), estimate_memory_savings().

## Flujo de Datos

```
Modelo Original
    |
    v
factorize_model() --> ZFactorizedLinear (U, S, V)
    |
    v
ZTrainEngine.train()
    |-- ZCompressedAdam (optimizer INT8)
    |-- ZGradientCompressor (gradientes comprimidos)
    |-- ZRankScheduler (crecimiento progresivo)
    |-- ZActivationCheckpoint (activaciones comprimidas)
    |
    v
export_full_model() --> nn.Linear (W reconstruido)
```

## Integracion con MNEME

MZTrain usa MNEME opcionalmente para:
- Compresion avanzada de activaciones via ZSpace
- Sin MNEME: fallback a cuantizacion INT8 manual
