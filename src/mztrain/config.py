"""
MZTrain - Configuracion y enumeraciones.

Define las configuraciones centrales y tipos enumerados para el sistema
de entrenamiento en espacio comprimido.
"""

import math
import logging
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("mztrain")

# ---------------------------------------------------------------------------
# Importacion condicional de MNEME
# ---------------------------------------------------------------------------

try:
    from mneme import CompressionLevel
    HAS_MNEME = True
except ImportError:
    HAS_MNEME = False

    class CompressionLevel(Enum):
        """Fallback cuando MNEME no esta disponible."""
        FAST = "fast"
        BALANCED = "balanced"
        ULTRA_FAST = "ultra_fast"
        MAXIMUM = "maximum"


# ============================================================================
# ENUMERACIONES
# ============================================================================

class RankSchedule(Enum):
    """Estrategia de crecimiento de rango durante entrenamiento progresivo.

    Attributes:
        CONSTANT: Rango fijo durante todo el entrenamiento.
        LINEAR: Crecimiento lineal del rango con respecto a los epochs.
        EXPONENTIAL: Crecimiento exponencial (duplicar cada N epochs).
        COSINE: Crecimiento coseno (lento-rapido-lento).
        ADAPTIVE: Crecimiento basado en convergencia del loss.
        SPECTRAL: Crecimiento basado en analisis espectral de pesos (T4).
    """
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"
    SPECTRAL = "spectral"


class GradientCompression(Enum):
    """Tipo de compresion de gradientes.

    Attributes:
        NONE: Sin compresion de gradientes.
        TOP_K: Solo mantener los top-K gradientes mas grandes.
        RANDOM_K: Muestra aleatoria de gradientes.
        QUANTIZE_1BIT: 1-bit SGD (solo signo del gradiente).
        QUANTIZE_INT8: Cuantizacion INT8 de gradientes.
        SVD: Compresion SVD de bajo rango para gradientes matriciales.
    """
    NONE = "none"
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    QUANTIZE_1BIT = "1bit"
    QUANTIZE_INT8 = "int8"
    SVD = "svd"


# ============================================================================
# CONFIGURACION PRINCIPAL
# ============================================================================

@dataclass
class ZTrainConfig:
    """Configuracion completa para entrenamiento MZTrain.

    Agrupa todos los hiperparametros del sistema de entrenamiento
    factorizado: rango, optimizer, gradientes, activaciones, y mas.

    Example:
        >>> config = ZTrainConfig(initial_rank=32, max_rank=256)
        >>> config.rank_schedule = RankSchedule.COSINE
        >>> config.gradient_compression = GradientCompression.TOP_K
    """

    # --- Rango factorizado ---
    initial_rank: int = 32
    """Rango inicial de factorizacion SVD."""

    max_rank: int = 256
    """Rango maximo permitido durante entrenamiento progresivo."""

    min_params_to_factorize: int = 4096
    """Minimo de parametros para que una capa sea factorizada."""

    energy_retention: float = 0.995
    """Retencion de energia SVD (0.995 = 99.5%)."""

    # --- Entrenamiento progresivo ---
    rank_schedule: RankSchedule = RankSchedule.EXPONENTIAL
    """Estrategia de crecimiento de rango."""

    rank_growth_interval: int = 10
    """Cada N epochs, evaluar crecimiento de rango."""

    rank_growth_factor: float = 2.0
    """Factor multiplicativo de crecimiento (para EXPONENTIAL)."""

    rank_growth_warmup_steps: int = 50
    """Steps de warmup lineal de LR despues de cada crecimiento de rango.
    Empieza al 10% del LR actual y sube linealmente. (ReLoRA jagged warmup)"""

    refactorize_interval: int = 0
    """Cada N steps de entrenamiento, refactorizar capas con SVD fresca.
    Previene rank collapse y mejora convergencia (~3.85 perplexity en 1B).
    0 = deshabilitado. Valor recomendado: 200."""

    # --- Optimizer ---
    use_galore: bool = False
    """Usar GaLore optimizer (proyeccion low-rank de gradientes).
    Reduce estados del optimizer de O(m*n) a O(r*n). Incompatible con
    gradient_compression (GaLore ya comprime gradientes internamente)."""

    galore_rank: int = 128
    """Rango del subespacio de proyeccion para GaLore."""

    galore_update_freq: int = 200
    """Cada cuantos steps actualizar el subespacio de proyeccion (SVD)."""

    compress_optimizer_states: bool = True
    """Comprimir estados (m, v) del optimizer con INT8."""

    optimizer_compression_ratio: float = 0.3
    """Objetivo de compresion para estados del optimizer (0.3 = 30%)."""

    # --- Gradient compression ---
    gradient_compression: GradientCompression = GradientCompression.NONE
    """Metodo de compresion de gradientes."""

    gradient_top_k_ratio: float = 0.1
    """Para TOP_K/RANDOM_K: ratio de gradientes a mantener (0.1 = top 10%)."""

    min_size_to_compress_grad: int = 1024
    """Minimo de elementos en un tensor para aplicar compresion de gradientes.
    Tensores mas pequenos (biases, layer norms) pasan sin comprimir."""

    compress_error_buffer: bool = False
    """Comprimir buffers de error feedback a INT8 (ConEF). Reduce overhead
    de memoria del error feedback de ~100% a ~25%. Puede causar inestabilidad
    con compresores de baja distorsion (INT8). Recomendado solo con TOP_K."""

    # --- Activation checkpointing ---
    checkpoint_activations: bool = True
    """Habilitar checkpointing de activaciones."""

    activation_compression: bool = True
    """Comprimir activaciones guardadas con MNEME/INT8."""

    checkpoint_every_n_layers: int = 2
    """Checkpoint cada N capas del modelo."""

    # --- Training ---
    learning_rate: float = 3e-4
    """Tasa de aprendizaje inicial."""

    weight_decay: float = 1e-4
    """Decaimiento de pesos (AdamW)."""

    warmup_steps: int = 100
    """Pasos de warmup para learning rate."""

    max_grad_norm: float = 1.0
    """Norma maxima de gradientes (gradient clipping)."""

    use_amp: bool = True
    """Usar mixed precision (AMP) cuando hay GPU CUDA."""

    # --- MNEME ---
    mneme_compression_level: CompressionLevel = CompressionLevel.BALANCED
    """Nivel de compresion MNEME para activaciones."""

    lazy_weight_loading: bool = True
    """Cargar pesos solo cuando se necesitan (lazy loading)."""

    # --- Logging ---
    log_interval: int = 10
    """Log cada N steps de entrenamiento."""

    memory_log_interval: int = 50
    """Resetear peak memory stats de CUDA cada N epochs (para medir consumo por ventana)."""

    # --- Sparse + Low-Rank (T6) ---
    use_sparse_component: bool = False
    """Usar capas ZSparseFactorizedLinear (W = U@S@V + sparse) en vez de
    ZFactorizedLinear puro. Cierra brecha de ~4.25 a ~0.5 puntos de perplexity."""

    sparse_density: float = 0.02
    """Fraccion de elementos no-cero en el componente sparse (0.02 = 2%).
    Mas density = mejor aproximacion pero mas parametros."""

    # --- Adaptive Optimizer (T7, Fase 4) ---
    optimizer_type: str = "compressed_adam"
    """Tipo de optimizer a usar.
    Opciones: "compressed_adam" (ZCompressedAdam), "galore" (ZGaLoreOptimizer),
    "adaptive" (ZAdaptiveOptimizer, APOLLO-style: m low-rank + v escalar)."""

    adaptive_block_size: int = 1024
    """Tamano de bloque para v escalar en ZAdaptiveOptimizer.
    Un valor v cada N elementos. Adam-mini usa ~64, APOLLO usa ~1024."""

    # --- Multi-Precision Manager (T8, Fase 4) ---
    precision_level: str = "standard"
    """Nivel de precision mixta.
    Opciones: "standard" (FP16 AMP actual), "aggressive" (INT8 activaciones+optimizer),
    "fp8" (FP8 E4M3/E5M2, solo Hopper+, fallback a aggressive en Ada Lovelace)."""

    # --- Spectral Rank Scheduler (T4) ---
    rank_energy_threshold: float = 0.90
    """Umbral de energia espectral para decidir crecer rango.
    Si energy_ratio < threshold, el rango actual es insuficiente."""

    rank_energy_ceiling: float = 0.99
    """Techo de energia espectral. Si energy_ratio > ceiling y loss estancado,
    ajustar LR en su lugar, no crecer rango."""

    rank_sample_layers: int = 4
    """Numero de capas a muestrear para el analisis espectral.
    Muestrear todas es O(L * m * n * min(m,n)), muestrear K es O(K * ...)."""

    # --- ElasticRank: rango bidireccional (grow / sleep / revive / prune) ---
    use_elastic_rank: bool = False
    """Habilitar ElasticRank: el rango no solo crece, tambien puede dormir
    direcciones singulares que dejan de aportar, revivirlas si vuelven a ser
    utiles, y podarlas si llevan mucho tiempo inactivas. Opt-in; no altera el
    camino de crecimiento existente cuando esta en False."""

    elastic_rank_check_interval: int = 500
    """Cada N steps de entrenamiento: medir señales (espectral + actividad
    Adam), actualizar EMAs y marcar candidatos. Barato; NO reconstruye el
    optimizer (eso se difiere al boundary de epoch)."""

    elastic_rank_ema_beta: float = 0.9
    """Factor de suavizado EMA para los scores espectral y de actividad.
    score_ema = beta*score_ema + (1-beta)*score."""

    elastic_rank_score_quantile: float = 0.95
    """Cuantil usado para normalizar el score espectral dentro de la capa
    (raw / quantile(raw, q)). p95 es mas robusto que max ante una direccion
    dominante gigante. Usar 1.0 equivale a normalizar por el maximo."""

    elastic_rank_sleep_spectral_threshold: float = 1e-3
    """tau_s: una direccion es candidata a dormir si su spectral_ema cae por
    debajo de esto (0.1% de la direccion dominante de la capa)."""

    elastic_rank_sleep_update_threshold: float = 1e-2
    """tau_u: ademas del score espectral bajo, el update_ema (energia del
    paso efectivo de Adam por direccion) debe estar por debajo de esto.
    Se usa AND, no producto: una direccion con score espectral bajo pero
    update alto sigue aprendiendo y NO debe dormir."""

    elastic_rank_sleep_patience_checks: int = 5
    """Numero de chequeos consecutivos cumpliendo el criterio antes de dormir."""

    elastic_rank_min_age_checks: int = 5
    """No dormir direcciones recien creadas/revividas hasta que tengan al
    menos esta edad (en chequeos)."""

    elastic_rank_post_growth_grace_checks: int = 5
    """Tras un crecimiento de rango, suprimir decisiones de sleep durante
    estos chequeos (es fase de estabilizacion, no de inutilidad)."""

    elastic_rank_post_refactor_grace_checks: int = 3
    """Tras una refactorizacion, suprimir decisiones de sleep durante estos
    chequeos (la SVD fresca acaba de reordenar el espectro)."""

    elastic_rank_min_per_layer: int = 4
    """Rango activo minimo por capa. ElasticRank nunca duerme por debajo de
    esto (evita matar capas enteras)."""

    elastic_rank_soft_sleep: bool = True
    """Aplicar 'soft sleep' intra-epoch: congelar (grad=0) las direcciones
    marcadas sin cambiar shapes ni reconstruir el optimizer. La compactacion
    real (que libera memoria) se difiere al boundary de epoch."""

    elastic_rank_compact_min_dirs: int = 4
    """Solo compactar (mover a sleep bank + reducir rango activo) cuando una
    capa tiene al menos esta cantidad de direcciones soft-dormidas acumuladas.
    Evita reconstrucciones de optimizer por una sola direccion."""

    elastic_rank_prune_after_epochs: int = 5
    """Una direccion dormida que lleva esta cantidad de epochs en el sleep
    bank sin ser revivida se poda definitivamente (libera su memoria CPU)."""

    elastic_rank_sleep_store_dtype: str = "float16"
    """Precision del sleep bank en CPU. Opciones: 'float16', 'bfloat16',
    'float32'. float16 minimiza memoria; los factores dormidos no necesitan
    precision alta porque solo se restauran como punto de partida."""

    elastic_rank_wake_momentum_damping: float = 0.1
    """Al revivir, exp_avg se restaura amortiguado por este factor (el
    momentum guardado puede empujar en una direccion obsoleta). exp_avg_sq
    se preserva (protege contra learning rates gigantes al revivir)."""

    elastic_rank_wake_warmup_steps: int = 100
    """Mini-warmup propio de cada direccion revivida: wake_gate sube de 0 a 1
    en estos steps y escala la contribucion (S_efectivo = wake_gate * S)."""

    elastic_rank_revive_cooldown_checks: int = 3
    """Tras revivir una direccion no puede volver a dormir durante estos
    chequeos (anti-oscilacion sleep<->revive)."""

    # --- ElasticRank v2: señal de redundancia funcional ---
    elastic_rank_use_redundancy_signal: bool = True
    """v2: ademas de la señal espectral+actividad (que solo ve direcciones
    cuyo |S_i|·‖U_i‖·‖V_i‖ decae), dormir tambien direcciones FUNCIONALMENTE
    REDUNDANTES: casi contenidas en el span de las otras (colinealidad /
    dependencia lineal del conjunto de terminos rango-1 T_i = S_i·u_i·v_i^T).
    Se mide con el leverage estadistico del Gram de los T_i. Detecta la
    redundancia que la señal por-triplete NO ve en entrenamiento desde cero
    (el sesgo low-rank de GD actua sobre el producto W, no sobre cada |S_i|;
    con Adam todas las direcciones siguen moviendose). Camino independiente
    con su propio umbral/paciencia: NO usa el gate de actividad (una direccion
    en el span de otras no aporta capacidad aunque su update sea alto)."""

    elastic_rank_redundancy_threshold: float = 0.9
    """tau_coh: coherence_ema por encima de esto = redundante. coherence en
    [0, 1): 0 = ortogonal a las demas direcciones, ->1 = casi en su span."""

    elastic_rank_redundancy_patience_checks: int = 3
    """Chequeos consecutivos como redundante antes de dormir por esta via."""

    elastic_rank_allow_rank_neutral_swap: bool = False
    """[avanzado] Permitir swaps rank-neutral (revivir la mejor dormida y
    dormir la peor activa) aunque el scheduler global no crezca. Desactivado
    por defecto en v1 por seguridad."""

    elastic_rank_allow_independent_growth: bool = False
    """[avanzado] Permitir que una capa crezca por presion local aunque el
    scheduler global no pida crecer. Desactivado por defecto en v1."""

    # --- ElasticRank probe: diagnostico de loss-sensitivity (no muta nada) ---
    elastic_rank_probe_loss_sensitivity: bool = False
    """Modo DIAGNOSTICO puro: no duerme/revive/compacta nada. Mide, por
    ablacion (wake_gate[i]=0 temporal sobre un batch sonda fijo, en eval/
    no_grad), cuanto sube la loss al quitar cada direccion. Sirve para
    decidir si la utilidad por direccion es heterogenea entre capas (-> un
    mercado global de rango se justifica) o plana (-> ElasticRank solo puede
    aspirar a 'no hacer daño'), y si los proxies de peso (spectral/update/
    coherence) correlacionan con la sensibilidad real de loss."""

    elastic_rank_probe_interval_epochs: int = 5
    """Cada cuantos epochs ejecutar el probe (coste: 1 forward por direccion
    sondeada sobre el batch sonda)."""

    elastic_rank_probe_max_dirs_per_layer: int = 16
    """Maximo de direcciones a sondear por capa (muestreo uniforme sobre el
    indice si rank excede esto). Acota el coste del diagnostico."""

    elastic_rank_probe_output: str = "bench/results/loss_sensitivity.json"
    """Ruta del JSON con los resultados acumulados del probe."""

    # --- ElasticRank loss-guard: rollback de compactacion por perdida ---
    # Decisivo: el experimento de loss-sensitivity mostro que los proxies
    # de peso (spectral/update/coherence) NO predicen el impacto real en
    # loss (Spearman ~0.13). Por tanto las señales solo generan CANDIDATOS;
    # la decision final de compactar la toma el loss-delta medido aqui.
    elastic_rank_loss_guard_enabled: bool = True
    """Toda compactacion se aplica como TENTATIVA: se mide la loss en un
    batch sonda fijo antes/despues y se REVIERTE si sube mas del umbral.
    Hace ElasticRank seguro-por-construccion (si no hay redundancia real,
    revierte; si la hay, la aprovecha sin picos de perdida)."""

    elastic_rank_loss_guard_threshold: float = 1e-3
    """Maximo rel_loss_delta tolerado: (loss_after-loss_before)/|loss_before|.
    1e-3 = 0.1%. Si se supera, se revierte la compactacion."""

    elastic_rank_loss_guard_cooldown_epochs: int = 3
    """Tras un rollback, no volver a proponer compactacion en esa capa
    durante estos epochs (anti-oscilacion)."""

    elastic_rank_loss_guard_mode: str = "probe_batch"
    """Como medir el efecto. "probe_batch": inmediato y determinista sobre
    el batch sonda fijo (eval/no_grad). Unico modo implementado en v1."""

    elastic_rank_loss_guard_batches: int = 1
    """[reservado] Nº de minibatches a entrenar antes de medir, para modos
    distintos de "probe_batch" (no implementados aun)."""

    # --- VRAM Governor (MVP): gobierna el CRECIMIENTO de rango bajo presion
    # de memoria + recuperacion de OOM. Alcance deliberadamente acotado: la
    # capa de sensores CUDA no es validable sin GPU, asi que el MVP solo
    # hace lo incondicionalmente correcto y testeable (gating de growth +
    # OOM-retry). Precision autopilot / checkpoint controller / rank-budget
    # en bytes / acoplamiento con ElasticRank quedan como roadmap. ---
    use_vram_governor: bool = False
    """Opt-in (default OFF: no validable sin GPU en este entorno; convencion
    del codebase para subsistemas nuevos). Si no hay CUDA, es inerte."""

    vram_governor_interval: int = 100
    """Cada cuantos steps medir memoria y actualizar la presion (barato)."""

    vram_governor_ema_beta: float = 0.9
    """Suavizado EMA de la presion (evita reaccionar a un spike aislado)."""

    vram_preventive_threshold: float = 0.92
    """Presion (EMA) por encima de la cual se BLOQUEA el crecimiento de rango.
    presion = max(reserved/total, peak_allocated/total)."""

    vram_emergency_threshold: float = 0.985
    """Presion por encima de la cual se entra en modo emergencia (ademas de
    bloquear growth; las acciones estructurales agresivas son roadmap)."""

    vram_hysteresis_checks: int = 3
    """Nº de chequeos consecutivos en una banda antes de cambiar de modo
    (anti-oscilacion)."""

    vram_oom_retry: bool = True
    """Ante torch.cuda.OutOfMemoryError: vaciar cache y reintentar UNA vez."""

    def validate(self) -> None:
        """Validar la configuracion.

        Raises:
            ValueError: Si algun parametro tiene un valor invalido.
        """
        if self.initial_rank < 1:
            raise ValueError(f"initial_rank debe ser >= 1, recibido: {self.initial_rank}")
        if self.max_rank < self.initial_rank:
            raise ValueError(
                f"max_rank ({self.max_rank}) debe ser >= initial_rank ({self.initial_rank})"
            )
        if self.min_params_to_factorize < 1:
            raise ValueError(f"min_params_to_factorize debe ser >= 1")
        if not 0.0 < self.energy_retention <= 1.0:
            raise ValueError(f"energy_retention debe estar en (0, 1], recibido: {self.energy_retention}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate debe ser > 0, recibido: {self.learning_rate}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm debe ser > 0, recibido: {self.max_grad_norm}")
        if not 0.0 < self.gradient_top_k_ratio <= 1.0:
            raise ValueError(f"gradient_top_k_ratio debe estar en (0, 1]")
        if self.use_elastic_rank:
            if self.elastic_rank_check_interval < 1:
                raise ValueError("elastic_rank_check_interval debe ser >= 1")
            if not 0.0 <= self.elastic_rank_ema_beta < 1.0:
                raise ValueError("elastic_rank_ema_beta debe estar en [0, 1)")
            if not 0.0 < self.elastic_rank_score_quantile <= 1.0:
                raise ValueError("elastic_rank_score_quantile debe estar en (0, 1]")
            if self.elastic_rank_min_per_layer < 1:
                raise ValueError("elastic_rank_min_per_layer debe ser >= 1")
            if self.elastic_rank_min_per_layer > self.initial_rank:
                raise ValueError(
                    f"elastic_rank_min_per_layer ({self.elastic_rank_min_per_layer}) "
                    f"no puede exceder initial_rank ({self.initial_rank})"
                )
            if not 0.0 <= self.elastic_rank_wake_momentum_damping <= 1.0:
                raise ValueError(
                    "elastic_rank_wake_momentum_damping debe estar en [0, 1]"
                )
            if self.elastic_rank_wake_warmup_steps < 0:
                raise ValueError("elastic_rank_wake_warmup_steps debe ser >= 0")
            if self.elastic_rank_prune_after_epochs < 1:
                raise ValueError("elastic_rank_prune_after_epochs debe ser >= 1")
            if self.elastic_rank_sleep_store_dtype not in (
                "float16", "bfloat16", "float32"
            ):
                raise ValueError(
                    "elastic_rank_sleep_store_dtype debe ser 'float16', "
                    "'bfloat16' o 'float32'"
                )
            if not 0.0 < self.elastic_rank_redundancy_threshold <= 1.0:
                raise ValueError(
                    "elastic_rank_redundancy_threshold debe estar en (0, 1]"
                )
            if self.elastic_rank_redundancy_patience_checks < 1:
                raise ValueError(
                    "elastic_rank_redundancy_patience_checks debe ser >= 1"
                )
            # --- umbrales/paciencias de sleep (faltaban) ---
            if self.elastic_rank_sleep_spectral_threshold <= 0.0:
                raise ValueError(
                    "elastic_rank_sleep_spectral_threshold debe ser > 0"
                )
            if self.elastic_rank_sleep_update_threshold <= 0.0:
                raise ValueError(
                    "elastic_rank_sleep_update_threshold debe ser > 0"
                )
            if self.elastic_rank_sleep_patience_checks < 1:
                raise ValueError(
                    "elastic_rank_sleep_patience_checks debe ser >= 1"
                )
            if self.elastic_rank_min_age_checks < 0:
                raise ValueError("elastic_rank_min_age_checks debe ser >= 0")
            if self.elastic_rank_post_growth_grace_checks < 0:
                raise ValueError(
                    "elastic_rank_post_growth_grace_checks debe ser >= 0"
                )
            if self.elastic_rank_post_refactor_grace_checks < 0:
                raise ValueError(
                    "elastic_rank_post_refactor_grace_checks debe ser >= 0"
                )
            if self.elastic_rank_compact_min_dirs < 1:
                raise ValueError(
                    "elastic_rank_compact_min_dirs debe ser >= 1"
                )
            if self.elastic_rank_revive_cooldown_checks < 0:
                raise ValueError(
                    "elastic_rank_revive_cooldown_checks debe ser >= 0"
                )
            # --- flags avanzadas: declaradas para la roadmap pero NO
            # implementadas correctamente (rank_neutral_swap no se usa;
            # independent_growth solo rellena hasta requested_rank cada epoch
            # y NEUTRALIZA la reduccion). Fallar fuerte en vez de mentir. ---
            if self.elastic_rank_allow_rank_neutral_swap:
                raise ValueError(
                    "elastic_rank_allow_rank_neutral_swap no esta implementado "
                    "en esta version (roadmap v3); mantener en False"
                )
            if self.elastic_rank_allow_independent_growth:
                raise ValueError(
                    "elastic_rank_allow_independent_growth no esta implementado "
                    "correctamente (rellena hasta requested_rank cada epoch y "
                    "anula la reduccion de memoria); mantener en False"
                )
            if self.elastic_rank_probe_loss_sensitivity:
                if self.elastic_rank_probe_interval_epochs < 1:
                    raise ValueError(
                        "elastic_rank_probe_interval_epochs debe ser >= 1"
                    )
                if self.elastic_rank_probe_max_dirs_per_layer < 1:
                    raise ValueError(
                        "elastic_rank_probe_max_dirs_per_layer debe ser >= 1"
                    )
            if self.elastic_rank_loss_guard_enabled:
                if self.elastic_rank_loss_guard_threshold < 0.0:
                    raise ValueError(
                        "elastic_rank_loss_guard_threshold debe ser >= 0"
                    )
                if self.elastic_rank_loss_guard_cooldown_epochs < 0:
                    raise ValueError(
                        "elastic_rank_loss_guard_cooldown_epochs debe ser >= 0"
                    )
                if self.elastic_rank_loss_guard_batches < 1:
                    raise ValueError(
                        "elastic_rank_loss_guard_batches debe ser >= 1"
                    )
                if self.elastic_rank_loss_guard_mode != "probe_batch":
                    raise ValueError(
                        "elastic_rank_loss_guard_mode: solo 'probe_batch' "
                        "esta implementado en v1"
                    )
        if self.use_vram_governor:
            if self.vram_governor_interval < 1:
                raise ValueError("vram_governor_interval debe ser >= 1")
            if not 0.0 <= self.vram_governor_ema_beta < 1.0:
                raise ValueError("vram_governor_ema_beta debe estar en [0, 1)")
            if not 0.0 < self.vram_preventive_threshold <= 1.0:
                raise ValueError(
                    "vram_preventive_threshold debe estar en (0, 1]")
            if not 0.0 < self.vram_emergency_threshold <= 1.0:
                raise ValueError(
                    "vram_emergency_threshold debe estar en (0, 1]")
            if self.vram_emergency_threshold < self.vram_preventive_threshold:
                raise ValueError(
                    "vram_emergency_threshold debe ser >= "
                    "vram_preventive_threshold"
                )
            if self.vram_hysteresis_checks < 1:
                raise ValueError("vram_hysteresis_checks debe ser >= 1")
