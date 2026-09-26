"""
MZTrain - Motor de entrenamiento principal.

Orquesta todas las componentes del sistema de entrenamiento factorizado:
conversion de modelo, optimizer comprimido, compresion de gradientes,
activation checkpointing y entrenamiento progresivo por rango.
"""

import copy
import time
import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger("mztrain")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.engine. "
        "Instalar con: pip install torch>=2.0.0"
    )

from .config import ZTrainConfig, GradientCompression, RankSchedule
from .layers import ZFactorizedLinear, ZFactorizedAttention, ZFactorizedTransformerBlock, ZSparseFactorizedLinear
from .optimizer import ZCompressedAdam
from .projector import ZGaLoreOptimizer
from .adaptive_optimizer import ZAdaptiveOptimizer
from .precision import ZMultiPrecisionManager
from .gradient import ZGradientCompressor
from .scheduler import ZRankScheduler, ZSpectralRankScheduler
from .checkpoint import ZActivationCheckpoint, z_checkpoint
from .refactorize import refactorize_model
from .elastic_rank import (
    ElasticRankController,
    SleepingDirection,
    decompress_opt_state,
    _LayerState,
)
from .vram_governor import ZVRAMGovernor


class ZTrainEngine:
    """Motor de entrenamiento MZTrain.

    Orquesta:

    1. Conversion de modelo a factorizado (nn.Linear -> ZFactorizedLinear)
    2. Optimizer con estados comprimidos (ZCompressedAdam)
    3. Compresion de gradientes (ZGradientCompressor)
    4. Activation checkpointing comprimido (ZActivationCheckpoint)
    5. Entrenamiento progresivo por rango (ZRankScheduler)

    Args:
        model: Modelo PyTorch a entrenar.
        config: Configuracion de entrenamiento.
        device: Dispositivo de computo (auto-detectado si None).

    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 512), nn.ReLU(),
        ...     nn.Linear(512, 256), nn.ReLU(),
        ...     nn.Linear(256, 10),
        ... )
        >>> config = ZTrainConfig(initial_rank=32, max_rank=256)
        >>> engine = ZTrainEngine(model, config)
        >>> summary = engine.train(
        ...     train_loader, val_loader,
        ...     loss_fn=lambda m, b: F.cross_entropy(m(b[0]), b[1]),
        ...     epochs=50,
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ZTrainConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or ZTrainConfig()
        self.config.validate()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Convertir modelo a factorizado
        self.model = self._factorize_model(model)
        self.model.to(self.device)

        # Aplicar activation checkpointing si configurado
        if self.config.checkpoint_activations:
            self._apply_activation_checkpointing()

        # Optimizer (GaLore o CompressedAdam segun config)
        self.optimizer = self._create_optimizer()

        # Gradient compressor
        self.grad_compressor = ZGradientCompressor(
            method=self.config.gradient_compression,
            top_k_ratio=self.config.gradient_top_k_ratio,
            min_size_to_compress=self.config.min_size_to_compress_grad,
            compress_error_buffer=self.config.compress_error_buffer,
        )

        # Rank scheduler (se configura en train())
        self.rank_scheduler: Optional[ZRankScheduler] = None

        # ElasticRank: rango bidireccional (opt-in). Si esta deshabilitado,
        # el camino de crecimiento existente queda intacto.
        self.elastic_rank: Optional[ElasticRankController] = (
            ElasticRankController(self.config)
            if getattr(self.config, "use_elastic_rank", False) else None
        )

        # VRAM Governor (opt-in; inerte sin CUDA). MVP: solo gobierna el
        # crecimiento de rango bajo presion + recuperacion de OOM.
        self.vram_governor: Optional[ZVRAMGovernor] = (
            ZVRAMGovernor(self.config)
            if getattr(self.config, "use_vram_governor", False) else None
        )

        # Multi-Precision Manager (T8, Fase 4)
        precision_level = getattr(self.config, 'precision_level', 'standard')
        self.precision_manager = ZMultiPrecisionManager(
            precision_level=precision_level,
            device=self.device,
        )

        # FP8 path: si el manager esta efectivamente en "fp8" (no fallback),
        # habilitamos GEMM en FP8 e4m3 dentro de cada ZFactorizedLinear.
        # Maestros de pesos (U, S, V) siguen en BF16/FP32; solo el matmul
        # baja a FP8 via torch._scaled_mm.
        if self.precision_manager.level == "fp8" and self.precision_manager.has_fp8:
            n_enabled = 0
            for module in self.model.modules():
                if isinstance(module, ZFactorizedLinear):
                    module._use_fp8 = True
                    n_enabled += 1
            logger.info(
                f"[MZTrain Precision] FP8 GEMM habilitado en {n_enabled} "
                f"capas factorizadas (torch._scaled_mm, e4m3 -> bf16)."
            )

        # AMP scaler
        self.scaler = None
        if self.config.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")

        # Metricas
        self._metrics: Dict[str, Any] = {
            "train_losses": [],
            "val_losses": [],
            "ranks": [],
            "memory_mb": [],
            "time_per_epoch": [],
            "total_train_time": 0,
        }

        # Best state para early stopping
        self._best_state: Optional[Dict[str, torch.Tensor]] = None

        # Rank growth warmup state
        self._rank_growth_warmup_remaining: int = 0
        self._rank_growth_warmup_base_lr: float = self.config.learning_rate

        # Global step counter para refactorizacion periodica
        self._global_step: int = 0

        # Loss-guard / probe: batch sonda y loss_fn (se fijan en train()).
        self._probe_batch: Any = None
        self._loss_fn: Optional[Callable] = None

        # Log estado inicial
        self._log_model_stats()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Crear optimizer segun configuracion.

        Seleccion por optimizer_type (Fase 4) con fallback a use_galore:
        - "adaptive": ZAdaptiveOptimizer (APOLLO-style, T7)
        - "galore" o use_galore=True: ZGaLoreOptimizer
        - "compressed_adam" (default): ZCompressedAdam
        """
        opt_type = getattr(self.config, 'optimizer_type', 'compressed_adam')

        if opt_type == "adaptive":
            logger.info(
                "[MZTrain] Usando ZAdaptiveOptimizer "
                f"(APOLLO-style, rank={self.config.galore_rank}, "
                f"block_size={self.config.adaptive_block_size})"
            )
            return ZAdaptiveOptimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                rank=self.config.galore_rank,
                block_size=self.config.adaptive_block_size,
                projection_update_freq=self.config.galore_update_freq,
                compress_states=self.config.compress_optimizer_states,
            )
        elif opt_type == "galore" or self.config.use_galore:
            logger.info("[MZTrain] Usando ZGaLoreOptimizer (proyeccion low-rank de gradientes)")
            return ZGaLoreOptimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                rank=self.config.galore_rank,
                projection_update_freq=self.config.galore_update_freq,
                compress_states=self.config.compress_optimizer_states,
            )
        else:
            return ZCompressedAdam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                compress_states=self.config.compress_optimizer_states,
            )

    def _apply_activation_checkpointing(self) -> None:
        """Envolver capas del modelo con activation checkpointing comprimido.

        Selecciona capas elegibles cada checkpoint_every_n_layers y reemplaza
        su forward con una version que comprime activaciones durante forward
        y las descomprime durante backward (recomputation + decompression).
        """
        checkpointed = 0
        layer_idx = 0

        # Buscar bloques transformer o modulos Sequential significativos
        for name, module in self.model.named_modules():
            is_checkpointable = isinstance(
                module, (ZFactorizedTransformerBlock, nn.TransformerEncoderLayer)
            )
            if not is_checkpointable:
                continue

            layer_idx += 1
            if layer_idx % self.config.checkpoint_every_n_layers != 0:
                continue

            # Envolver forward con z_checkpoint
            original_forward = module.forward

            def _make_wrapped(fn):
                """Closure para capturar fn correctamente."""
                def wrapped_forward(*args, **kwargs):
                    # z_checkpoint solo acepta args posicionales
                    if kwargs:
                        return fn(*args, **kwargs)
                    return z_checkpoint(fn, *args)
                return wrapped_forward

            module.forward = _make_wrapped(original_forward)
            checkpointed += 1

        if checkpointed > 0:
            logger.info(
                f"[MZTrain] Activation checkpointing aplicado a {checkpointed} capas "
                f"(cada {self.config.checkpoint_every_n_layers})"
            )
        else:
            logger.debug(
                "[MZTrain] No se encontraron capas elegibles para activation checkpointing"
            )

    def _factorize_model(self, model: nn.Module) -> nn.Module:
        """Convertir capas nn.Linear elegibles a ZFactorizedLinear.

        Criterios de elegibilidad:
        - Capa debe tener >= min_params_to_factorize parametros.
        - Capas de embedding, normalization, y cabezas pequenas se dejan intactas.

        Args:
            model: Modelo original.

        Returns:
            Copia del modelo con capas factorizadas.
        """
        model = copy.deepcopy(model)
        replaced = 0
        skipped = 0

        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                num_params = module.weight.numel()

                if num_params < self.config.min_params_to_factorize:
                    skipped += 1
                    continue

                if self.config.use_sparse_component:
                    z_linear = ZSparseFactorizedLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.config.initial_rank,
                        bias=module.bias is not None,
                        existing_weight=module.weight.data,
                        sparse_density=self.config.sparse_density,
                    )
                else:
                    z_linear = ZFactorizedLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.config.initial_rank,
                        bias=module.bias is not None,
                        existing_weight=module.weight.data,
                    )

                if module.bias is not None:
                    z_linear.bias.data.copy_(module.bias.data)

                if name == "":
                    model = z_linear
                else:
                    self._replace_module(model, name, z_linear)
                replaced += 1

        logger.info(
            f"[MZTrain] Modelo factorizado: {replaced} capas reemplazadas, "
            f"{skipped} omitidas (< {self.config.min_params_to_factorize} params)"
        )
        return model

    @staticmethod
    def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
        """Reemplazar un modulo por nombre (soporta nombres con punto)."""
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        if parts[-1].isdigit():
            parent[int(parts[-1])] = new_module
        else:
            setattr(parent, parts[-1], new_module)

    def _grow_model_rank(self, new_rank: int) -> None:
        """Crecer el rango de todas las capas factorizadas preservando momentum.

        En vez de destruir y recrear el optimizer (que pierde todos los estados
        acumulados de Adam: m y v), este metodo:
        1. Captura los estados del optimizer para los parametros que van a cambiar.
        2. Crece el rango de las capas.
        3. Reconstruye el optimizer con los nuevos parametros.
        4. Migra los estados antiguos a los nuevos (padding con zeros/eps).
        5. Activa un mini-warmup de LR post-crecimiento.
        """
        # Paso 1: Capturar el estado DESCOMPRIMIDO del optimizer de TODOS los
        # parametros. Recrear el optimizer (paso 3) resetea el estado de Adam;
        # para no perder el momentum se captura antes y se restaura despues.
        # - factores U/S/V: seran tensores nuevos (grow_rank los reemplaza) ->
        #   se migran con reshape.
        # - resto (bias, LayerNorm, embeddings, ...): su tensor sobrevive ->
        #   se restauran por identidad. (Antes solo se migraban U/S/V, y el
        #   formato comprimido se saltaba, reseteando m,v en cada crecimiento.)
        factor_param_ids = set()
        captured_factor = {}   # (id(module), pname) -> {'shape','exp_avg','exp_avg_sq','step'}
        for module in self.model.modules():
            if isinstance(module, ZFactorizedLinear):
                param_names = ['U', 'S', 'V']
                if isinstance(module, ZSparseFactorizedLinear):
                    param_names.append('sparse_values')
                for pname in param_names:
                    param = getattr(module, pname, None)
                    if param is None:
                        continue
                    factor_param_ids.add(id(param))
                    st = self.optimizer.state.get(param)
                    captured_factor[(id(module), pname)] = {
                        'shape': param.shape,
                        'exp_avg': decompress_opt_state(
                            self.optimizer, st, 'exp_avg', param) if st else None,
                        'exp_avg_sq': decompress_opt_state(
                            self.optimizer, st, 'exp_avg_sq', param) if st else None,
                        'step': int(st.get('step', 0) or 0) if st else 0,
                    }

        captured_other = {}    # id(param) -> (param, {'exp_avg','exp_avg_sq','step'})
        for p in self.model.parameters():
            if id(p) in factor_param_ids:
                continue
            st = self.optimizer.state.get(p)
            if not st:
                continue
            captured_other[id(p)] = (p, {
                'exp_avg': decompress_opt_state(self.optimizer, st, 'exp_avg', p),
                'exp_avg_sq': decompress_opt_state(self.optimizer, st, 'exp_avg_sq', p),
                'step': int(st.get('step', 0) or 0),
            })

        # Paso 2: Crecer rango en capas
        grown = 0
        for module in self.model.modules():
            if isinstance(module, ZFactorizedLinear):
                if new_rank > module.rank:
                    module.grow_rank(new_rank)
                    grown += 1
            elif isinstance(module, (ZFactorizedAttention, ZFactorizedTransformerBlock)):
                module.grow_rank(new_rank)
                grown += 1

        if grown == 0:
            return

        # Paso 3: Recrear optimizer con nuevos parametros (preserva tipo)
        old_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = self._create_optimizer()
        # Restaurar LR (puede haber sido modificado por scheduler)
        for pg in self.optimizer.param_groups:
            pg['lr'] = old_lr

        migrated = 0

        # Paso 4: Restaurar el estado de los parametros NO factorizados por
        # identidad (su tensor no cambio con el crecimiento).
        for _pid, (p, c) in captured_other.items():
            ea, es = c['exp_avg'], c['exp_avg_sq']
            self.optimizer.state[p] = {
                'step': c['step'],
                'exp_avg': (ea.to(p.device, p.dtype) if ea is not None
                            else torch.zeros_like(p.data)),
                'exp_avg_sq': (es.to(p.device, p.dtype) if es is not None
                               else torch.zeros_like(p.data)),
                'compressed': False,
            }
            migrated += 1

        # Paso 5: Migrar el estado de los factores U/S/V (tensores nuevos):
        # copiar la region que ya existia y rellenar lo nuevo.
        for module in self.model.modules():
            if not isinstance(module, ZFactorizedLinear):
                continue
            # ZSparseFactorizedLinear.grow_rank hace re-SVD (base completamente
            # nueva): el momentum viejo esta en la base antigua y ya no aplica,
            # asi que no se migra y el estado queda fresco (Adam se re-estabiliza
            # en pocos pasos). En ZFactorizedLinear puro la base de las primeras
            # old_rank direcciones se preserva (preserve_weights=True) y por eso
            # alli si se migra.
            if isinstance(module, ZSparseFactorizedLinear):
                continue
            param_names = ['U', 'S', 'V']
            for pname in param_names:
                key = (id(module), pname)
                if key not in captured_factor:
                    continue
                param = getattr(module, pname, None)
                if param is None:
                    continue
                cap = captured_factor[key]
                old_shape = cap['shape']
                new_exp_avg = torch.zeros_like(param.data)
                new_exp_avg_sq = torch.zeros_like(param.data)

                for moment_key, new_moment in (('exp_avg', new_exp_avg),
                                               ('exp_avg_sq', new_exp_avg_sq)):
                    old_moment = cap[moment_key]
                    if old_moment is None:
                        continue
                    old_moment = old_moment.to(new_moment.device, new_moment.dtype)
                    # Copiar la region que existia antes
                    if old_moment.dim() == 2 and new_moment.dim() == 2:
                        h = min(old_shape[0], new_moment.shape[0])
                        w = min(old_shape[1], new_moment.shape[1])
                        new_moment[:h, :w] = old_moment[:h, :w]
                    elif old_moment.dim() == 1 and new_moment.dim() == 1:
                        n = min(old_shape[0], new_moment.shape[0])
                        new_moment[:n] = old_moment[:n]

                # Para exp_avg_sq (varianza), las regiones nuevas necesitan un
                # valor pequeno para evitar learning rates gigantes.
                if cap['exp_avg_sq'] is not None:
                    mask = new_exp_avg_sq == 0
                    if mask.any():
                        existing_mean = cap['exp_avg_sq'].abs().mean()
                        fill_val = max(float(existing_mean) * 0.1, 1e-8)
                        new_exp_avg_sq[mask] = fill_val

                self.optimizer.state[param] = {
                    'step': cap['step'],
                    'exp_avg': new_exp_avg,
                    'exp_avg_sq': new_exp_avg_sq,
                    'compressed': False,
                }
                migrated += 1

        # Paso 6: Warmup post-crecimiento — empezar al 10% del LR. Solo si
        # esta activado: con warmup_steps=0 el LR quedaba clavado en 0.1x
        # porque la restauracion vive en el step de warmup, que no corre.
        if self.config.rank_growth_warmup_steps > 0:
            self._rank_growth_warmup_remaining = self.config.rank_growth_warmup_steps
            self._rank_growth_warmup_base_lr = old_lr
            warmup_start_lr = old_lr * 0.1
            for pg in self.optimizer.param_groups:
                pg['lr'] = warmup_start_lr

        logger.info(
            f"[MZTrain] Rango crecido a {new_rank} en {grown} capas, "
            f"{migrated} estados migrados, warmup {self.config.rank_growth_warmup_steps} steps"
        )

    def _restore_best_state(self) -> None:
        """Restaurar el mejor snapshot, tolerante a cambios de topologia.

        El mejor snapshot se toma en su epoch; si despues la topologia cambia
        (crecimiento de rango o ElasticRank: sleep/revive/prune) y luego salta
        el early stopping, un load_state_dict estricto fallaria por shapes.
        En ese caso se restaura tensor a tensor solo donde las shapes
        coinciden y se conservan los pesos actuales para el resto (las capas
        crecidas mantienen el mejor subespacio en su prefijo de todos modos).
        """
        if self._best_state is None:
            return
        try:
            self.model.load_state_dict(self._best_state)
            return
        except RuntimeError:
            pass

        own = dict(self.model.state_dict())
        matched = {}
        skipped = 0
        for k, v in self._best_state.items():
            cur = own.get(k)
            if cur is not None and tuple(cur.shape) == tuple(v.shape):
                matched[k] = v
            else:
                skipped += 1
        self.model.load_state_dict(matched, strict=False)
        logger.warning(
            f"[MZTrain] best-state restaurado parcialmente: la topologia "
            f"cambio desde el mejor epoch ({skipped} tensores omitidos por "
            f"cambio de shape; se conservan los pesos actuales para esos)."
        )

    def _apply_elastic_topology(self, plan, epoch: int) -> None:
        """Aplicar el plan de ElasticRank en el boundary de epoch.

        Reutiliza el patron ya probado del camino de crecimiento (capturar
        estados -> mutar factores -> reconstruir optimizer -> migrar estados
        -> warmup) pero generalizado a una permutacion arbitraria de
        direcciones: conservar, dormir (-> sleep bank), revivir (<- sleep
        bank, momentum amortiguado) y crecer (ruido).

        A diferencia del camino legacy, preserva tambien el estado de Adam de
        las capas NO tocadas y de los parametros no factorizados (bias,
        LayerNorm), porque la reconstruccion del optimizer es global.
        """
        ec = self.elastic_rank
        cfg = self.config
        damping = float(cfg.elastic_rank_wake_momentum_damping)
        store_dtype = ec.store_dtype

        modules = dict(self.model.named_modules())

        # Identidad de los parametros factorizados elegibles (pre-mutacion).
        factor_param_ids = set()
        eligible_names = []
        for name, module in ec._iter_layers(self.model):
            eligible_names.append(name)
            for pn in ("U", "S", "V"):
                factor_param_ids.add(id(getattr(module, pn)))

        # 1. Capturar (decomprimido) el estado del optimizer.
        captured_factor = {}   # (name, pn) -> {'exp_avg','exp_avg_sq','step'}
        for name in eligible_names:
            module = modules[name]
            for pn in ("U", "S", "V"):
                p = getattr(module, pn)
                st = self.optimizer.state.get(p)
                step = int(st.get("step", 0) or 0) if st else 0
                captured_factor[(name, pn)] = {
                    "exp_avg": decompress_opt_state(
                        self.optimizer, st, "exp_avg", p
                    ) if st else None,
                    "exp_avg_sq": decompress_opt_state(
                        self.optimizer, st, "exp_avg_sq", p
                    ) if st else None,
                    "step": step,
                }
        captured_other = {}    # id(param) -> (param, {'exp_avg','exp_avg_sq','step'})
        for p in self.model.parameters():
            if id(p) in factor_param_ids:
                continue
            st = self.optimizer.state.get(p)
            if not st:
                continue
            captured_other[id(p)] = (p, {
                "exp_avg": decompress_opt_state(
                    self.optimizer, st, "exp_avg", p
                ),
                "exp_avg_sq": decompress_opt_state(
                    self.optimizer, st, "exp_avg_sq", p
                ),
                "step": int(st.get("step", 0) or 0),
            })

        # 2. Cirugia de factores por capa cambiada.
        target_factor = {}    # (name, pn) -> {'exp_avg','exp_avg_sq','step'}

        def _slice(t, pn, idx):
            if t is None:
                return None
            if pn == "U":
                return t[:, idx]
            if pn == "V":
                return t[idx, :]
            return t[idx]

        def _assign(t, pn, pos, val):
            if pn == "U":
                t[:, pos] = val
            elif pn == "V":
                t[pos, :] = val
            else:
                t[pos] = val

        for name, lp in plan.changed_layers():
            module = modules[name]
            device = module.U.device
            dtype = module.U.dtype
            old_U = module.U.data
            old_S = module.S.data
            old_V = module.V.data
            old_wake = module.wake_gate
            old_mask = module.sleep_mask
            cap = {pn: captured_factor[(name, pn)] for pn in ("U", "S", "V")}
            sc_state = ec._states.get(name)

            # 2a. Mover direcciones dormidas al sleep bank (CPU, baja prec.).
            for i in lp.sleep:
                spec = (float(sc_state.spectral_ema[i])
                        if sc_state is not None
                        and i < sc_state.spectral_ema.numel() else 0.0)
                upd = (float(sc_state.update_ema[i])
                       if sc_state is not None
                       and i < sc_state.update_ema.numel() else 0.0)

                def _to_store(x):
                    return x.detach().to("cpu", store_dtype).clone()

                def _factor_vec(pn):
                    if pn == "U":
                        return old_U[:, i]
                    if pn == "V":
                        return old_V[i, :]
                    return old_S[i]

                def _adam_slice(pn, kind):
                    s = _slice(cap[pn][kind], pn, i)
                    if s is not None:
                        return _to_store(s)
                    return torch.zeros_like(_to_store(_factor_vec(pn)))

                sd = SleepingDirection(
                    layer_name=name,
                    u=_to_store(old_U[:, i]),
                    # S SIEMPRE en fp32 (regla dura de precision.py): cuantizar
                    # el valor singular a fp16 lo redondea y, al revivir, altera
                    # la escala de toda la direccion factorizada.
                    s=old_S[i].detach().to("cpu", torch.float32).clone(),
                    v=_to_store(old_V[i, :]),
                    m_u=_adam_slice("U", "exp_avg"),
                    m_s=_adam_slice("S", "exp_avg"),
                    m_v=_adam_slice("V", "exp_avg"),
                    vsq_u=_adam_slice("U", "exp_avg_sq"),
                    vsq_s=_adam_slice("S", "exp_avg_sq"),
                    vsq_v=_adam_slice("V", "exp_avg_sq"),
                    step=cap["U"]["step"],
                    spectral_at_sleep=spec,
                    update_at_sleep=upd,
                    epoch_slept=epoch,
                )
                ec.add_sleeper(name, sd)

            ec.take_revivals(name, lp.revive)

            # 2b. Construir nuevos factores en el orden keep + revive + grow.
            nr = lp.new_rank
            new_U = torch.zeros(module.out_features, nr, device=device, dtype=dtype)
            new_S = torch.zeros(nr, device=device, dtype=dtype)
            new_V = torch.zeros(nr, module.in_features, device=device, dtype=dtype)
            new_wake = torch.ones(nr, device=device)
            new_mask = torch.zeros(nr, dtype=torch.bool, device=device)
            tgt = {
                "U": {"exp_avg": torch.zeros_like(new_U),
                      "exp_avg_sq": torch.zeros_like(new_U)},
                "S": {"exp_avg": torch.zeros_like(new_S),
                      "exp_avg_sq": torch.zeros_like(new_S)},
                "V": {"exp_avg": torch.zeros_like(new_V),
                      "exp_avg_sq": torch.zeros_like(new_V)},
            }

            pos = 0
            for old_i in lp.keep:
                if pos >= nr:
                    break
                new_U[:, pos] = old_U[:, old_i]
                new_S[pos] = old_S[old_i]
                new_V[pos, :] = old_V[old_i, :]
                if old_i < old_wake.numel():
                    new_wake[pos] = old_wake[old_i]
                    new_mask[pos] = old_mask[old_i]
                for pn in ("U", "S", "V"):
                    for kind in ("exp_avg", "exp_avg_sq"):
                        src = _slice(cap[pn][kind], pn, old_i)
                        if src is not None:
                            _assign(tgt[pn][kind], pn, pos,
                                    src.to(device, dtype))
                pos += 1

            for sd in lp.revive:
                if pos >= nr:
                    break
                new_U[:, pos] = sd.u.to(device, dtype)
                new_S[pos] = sd.s.to(device, dtype)
                new_V[pos, :] = sd.v.to(device, dtype)
                # warmup>0 -> arranca en 0 y tick_wake rampa a 1.
                # warmup<=0 -> sin rampa: gate 1 inmediato (si no, la
                # direccion revivida quedaria apagada para siempre porque
                # tick_wake solo actua con wake_left>0).
                new_wake[pos] = (
                    0.0 if self.config.elastic_rank_wake_warmup_steps > 0
                    else 1.0
                )
                new_mask[pos] = False
                _assign(tgt["U"]["exp_avg"], "U", pos,
                        sd.m_u.to(device, dtype) * damping)
                _assign(tgt["S"]["exp_avg"], "S", pos,
                        sd.m_s.to(device, dtype) * damping)
                _assign(tgt["V"]["exp_avg"], "V", pos,
                        sd.m_v.to(device, dtype) * damping)
                _assign(tgt["U"]["exp_avg_sq"], "U", pos,
                        sd.vsq_u.to(device, dtype))
                _assign(tgt["S"]["exp_avg_sq"], "S", pos,
                        sd.vsq_s.to(device, dtype))
                _assign(tgt["V"]["exp_avg_sq"], "V", pos,
                        sd.vsq_v.to(device, dtype))
                pos += 1

            if pos < nr:
                # Componentes nuevos (ruido), igual que grow_rank legacy.
                noise_scale = float(old_S.abs().min()) * 0.01
                g = slice(pos, nr)
                nn.init.normal_(new_U[:, g], 0, 0.01)
                nn.init.normal_(new_V[g, :], 0, 0.01)
                new_S[g] = noise_scale

            # exp_avg_sq: rellenar ceros (regiones nuevas / revive sin info)
            # con un valor pequeño para evitar learning rates gigantes.
            for pn in ("U", "S", "V"):
                es = tgt[pn]["exp_avg_sq"]
                nz = es[es != 0]
                fill = (max(float(nz.abs().mean()) * 0.1, 1e-8)
                        if nz.numel() > 0 else 1e-8)
                es[es == 0] = fill
                target_factor[(name, pn)] = {
                    "exp_avg": tgt[pn]["exp_avg"],
                    "exp_avg_sq": es,
                    "step": cap[pn]["step"],
                }

            module.elastic_replace_factors(
                new_U, new_S, new_V, new_wake, new_mask
            )
            ec.remap_state(name, lp)

        # 3. Reconstruir el optimizer una sola vez (preserva tipo y LR).
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer = self._create_optimizer()
        for pg in self.optimizer.param_groups:
            pg["lr"] = old_lr

        changed = {n for n, _ in plan.changed_layers()}

        def _set_state(p, ea, es, step):
            self.optimizer.state[p] = {
                "step": int(step),
                "exp_avg": (ea.to(p.device, p.dtype) if ea is not None
                            else torch.zeros_like(p.data)),
                "exp_avg_sq": (es.to(p.device, p.dtype) if es is not None
                               else torch.zeros_like(p.data)),
                "compressed": False,
            }

        # 4. Migrar/restaurar estados.
        for name in eligible_names:
            module = modules[name]
            for pn in ("U", "S", "V"):
                p = getattr(module, pn)
                if name in changed:
                    t = target_factor[(name, pn)]
                    _set_state(p, t["exp_avg"], t["exp_avg_sq"], t["step"])
                else:
                    c = captured_factor[(name, pn)]
                    if c["exp_avg"] is not None or c["exp_avg_sq"] is not None:
                        _set_state(p, c["exp_avg"], c["exp_avg_sq"], c["step"])
        for _pid, (p, c) in captured_other.items():
            if c["exp_avg"] is not None or c["exp_avg_sq"] is not None:
                _set_state(p, c["exp_avg"], c["exp_avg_sq"], c["step"])

        # Invalidar el error-feedback de compresion de gradientes de las capas
        # cuyas direcciones se reordenaron: el residuo de cuantizacion viejo
        # apunta a las direcciones singulares antiguas. El check por shape de
        # ZGradientCompressor NO lo detecta cuando el rango se mantiene (dormir
        # k + revivir/crecer k), asi que se invalida explicitamente por nombre.
        if changed:
            changed_param_ids = set()
            for name in changed:
                for _p in modules[name].parameters(recurse=True):
                    changed_param_ids.add(id(_p))
            for pname, p in self.model.named_parameters():
                if id(p) in changed_param_ids:
                    self.grad_compressor.invalidate(pname)

        # 5. Warmup post-cambio estructural (igual que rank growth). Solo si
        # esta activado: con warmup_steps=0 el LR quedaba clavado en 0.1x.
        if self.config.rank_growth_warmup_steps > 0:
            self._rank_growth_warmup_remaining = self.config.rank_growth_warmup_steps
            self._rank_growth_warmup_base_lr = old_lr
            for pg in self.optimizer.param_groups:
                pg["lr"] = old_lr * 0.1
        ec.note_growth()

    # ------------------------------------------------------------------
    # Loss-guard: rollback de compactacion por perdida (nucleo de decision)
    # ------------------------------------------------------------------

    def _eval_probe_loss(self) -> Optional[float]:
        """Loss en el batch sonda fijo, en eval/no_grad (determinista).
        Restaura el modo de entrenamiento del modelo."""
        if self._probe_batch is None:
            return None
        was = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return float(self._loss_fn(self.model, self._probe_batch))
        finally:
            self.model.train(was)

    def _set_opt_state(self, p, ea, es, step) -> None:
        self.optimizer.state[p] = {
            "step": int(step),
            "exp_avg": (ea.to(p.device, p.dtype) if ea is not None
                        else torch.zeros_like(p.data)),
            "exp_avg_sq": (es.to(p.device, p.dtype) if es is not None
                           else torch.zeros_like(p.data)),
            "compressed": False,
        }

    def _snapshot_elastic_state(self) -> Dict[str, Any]:
        """Snapshot COMPLETO para revertir una compactacion tentativa:
        factores+buffers por capa, estado Adam (decomprimido) de todos los
        params, estado de scoring del controller, sleep bank y warmup.
        """
        ec = self.elastic_rank
        modules = dict(self.model.named_modules())
        factors: Dict[str, Any] = {}
        factor_ids = set()
        opt_factor: Dict[Any, Any] = {}
        for name, module in ec._iter_layers(self.model):
            factors[name] = {
                "U": module.U.detach().clone(),
                "S": module.S.detach().clone(),
                "V": module.V.detach().clone(),
                "wake": module.wake_gate.detach().clone(),
                "mask": module.sleep_mask.detach().clone(),
            }
            for pn in ("U", "S", "V"):
                p = getattr(module, pn)
                factor_ids.add(id(p))
                st = self.optimizer.state.get(p)
                opt_factor[(name, pn)] = {
                    "exp_avg": decompress_opt_state(
                        self.optimizer, st, "exp_avg", p) if st else None,
                    "exp_avg_sq": decompress_opt_state(
                        self.optimizer, st, "exp_avg_sq", p) if st else None,
                    "step": int(st.get("step", 0) or 0) if st else 0,
                }
        opt_other: Dict[int, Any] = {}
        for p in self.model.parameters():
            if id(p) in factor_ids:
                continue
            st = self.optimizer.state.get(p)
            if not st:
                continue
            opt_other[id(p)] = (p, {
                "exp_avg": decompress_opt_state(
                    self.optimizer, st, "exp_avg", p),
                "exp_avg_sq": decompress_opt_state(
                    self.optimizer, st, "exp_avg_sq", p),
                "step": int(st.get("step", 0) or 0),
            })

        def _clone_ls(ls: _LayerState) -> _LayerState:
            return _LayerState(
                spectral_ema=ls.spectral_ema.clone(),
                update_ema=ls.update_ema.clone(),
                low_counter=ls.low_counter.clone(),
                age=ls.age.clone(),
                revive_cd=ls.revive_cd.clone(),
                wake_left=ls.wake_left.clone(),
                redund_ema=ls.redund_ema.clone(),
                redund_counter=ls.redund_counter.clone(),
            )

        return {
            "factors": factors,
            "opt_factor": opt_factor,
            "opt_other": opt_other,
            "ctrl_states": {n: _clone_ls(s) for n, s in ec._states.items()},
            "bank": {n: list(v) for n, v in ec.bank.items()},
            "counters": (ec._total_slept, ec._total_revived, ec._total_pruned),
            "growth_grace": ec._growth_grace,
            "lr": self.optimizer.param_groups[0]["lr"],
            "warmup_remaining": self._rank_growth_warmup_remaining,
            "warmup_base_lr": self._rank_growth_warmup_base_lr,
        }

    def _restore_elastic_state(self, snap: Dict[str, Any]) -> None:
        """Revertir al snapshot pre-compactacion (factores, optimizer,
        estado del controller, warmup)."""
        ec = self.elastic_rank
        modules = dict(self.model.named_modules())
        for name, f in snap["factors"].items():
            modules[name].elastic_replace_factors(
                f["U"], f["S"], f["V"], f["wake"], f["mask"])

        self.optimizer = self._create_optimizer()
        for pg in self.optimizer.param_groups:
            pg["lr"] = snap["lr"]
        for name, module in ec._iter_layers(self.model):
            for pn in ("U", "S", "V"):
                c = snap["opt_factor"].get((name, pn))
                if c is not None:
                    self._set_opt_state(getattr(module, pn),
                                        c["exp_avg"], c["exp_avg_sq"],
                                        c["step"])
        for _pid, (p, c) in snap["opt_other"].items():
            self._set_opt_state(p, c["exp_avg"], c["exp_avg_sq"], c["step"])

        ec._states = dict(snap["ctrl_states"])
        ec.bank = {n: list(v) for n, v in snap["bank"].items()}
        ec._total_slept, ec._total_revived, ec._total_pruned = snap["counters"]
        ec._growth_grace = snap["growth_grace"]
        self._rank_growth_warmup_remaining = snap["warmup_remaining"]
        self._rank_growth_warmup_base_lr = snap["warmup_base_lr"]

    def _apply_elastic_topology_guarded(self, plan, epoch: int) -> None:
        """Aplica la compactacion como TENTATIVA y la revierte si la loss
        (batch sonda fijo) sube mas de elastic_rank_loss_guard_threshold.

        Las señales de peso ya NO deciden muerte definitiva (el experimento
        de loss-sensitivity mostro Spearman~0.13): solo generan candidatos;
        el loss-delta medido es el arbitro final.
        """
        cfg = self.config
        # El guard protege la COMPACTACION (quitar capacidad). Un plan que
        # solo revive/crece añade capacidad: medir justo despues penalizaria
        # injustamente a las direcciones revividas que aun rampan (gate~0).
        compacts = any(
            len(lp.sleep) > 0 for _n, lp in plan.changed_layers()
        )
        guard = (getattr(cfg, "elastic_rank_loss_guard_enabled", False)
                 and self._probe_batch is not None
                 and compacts)
        if not guard:
            self._apply_elastic_topology(plan, epoch)
            return

        loss_before = self._eval_probe_loss()
        if loss_before is None:
            self._apply_elastic_topology(plan, epoch)
            return

        snap = self._snapshot_elastic_state()
        self._apply_elastic_topology(plan, epoch)
        loss_after = self._eval_probe_loss()

        rel = (loss_after - loss_before) / max(abs(loss_before), 1e-8)
        thr = float(cfg.elastic_rank_loss_guard_threshold)
        if rel > thr:
            self._restore_elastic_state(snap)
            self.elastic_rank.note_rollback(plan, epoch)
            logger.info(
                f"[MZTrain] ElasticRank ROLLBACK epoch {epoch}: "
                f"rel_loss_delta={rel:.2e} > {thr:.2e} -> revertido "
                f"({plan.summary()}); cooldown "
                f"{cfg.elastic_rank_loss_guard_cooldown_epochs} epochs"
            )
        else:
            logger.info(
                f"[MZTrain] ElasticRank compactacion CONFIRMADA epoch {epoch}: "
                f"rel_loss_delta={rel:.2e} <= {thr:.2e}"
            )

    def _log_model_stats(self) -> None:
        """Log estadisticas del modelo factorizado."""
        total_params = 0
        total_full_params = 0
        factorized_layers = 0

        for module in self.model.modules():
            if isinstance(module, ZFactorizedLinear):
                total_params += module.num_parameters
                total_full_params += module.full_parameters
                factorized_layers += 1

        all_params = sum(p.numel() for p in self.model.parameters())
        savings = (
            (1 - total_params / max(total_full_params, 1)) * 100
            if total_full_params > 0 else 0
        )

        logger.info(
            f"[MZTrain] ========== MODELO FACTORIZADO ==========\n"
            f"  Capas factorizadas:     {factorized_layers}\n"
            f"  Params totales:         {all_params:,}\n"
            f"  Params (si full):       {total_full_params:,}\n"
            f"  Ahorro en pesos:        {savings:.1f}%\n"
            f"  Rango actual:           {self.config.initial_rank}\n"
            f"  Device:                 {self.device}\n"
            f"  AMP:                    {self.config.use_amp}\n"
            f"  Optimizer compress:     {self.config.compress_optimizer_states}\n"
            f"  Grad compression:       {self.config.gradient_compression.value}\n"
            f"  ============================================="
        )

    def _to_device(self, batch: Any) -> Any:
        """Mover batch al device recursivamente (soporta Tensor, list, tuple, dict)."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(b) for b in batch)
        return batch

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        epoch: int,
    ) -> float:
        """Entrenar un epoch completo.

        Args:
            train_loader: DataLoader de entrenamiento.
            loss_fn: Funcion de perdida fn(model, batch) -> loss.
            epoch: Numero de epoch actual.

        Returns:
            Loss promedio del epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        def _fwd_bwd(batch):
            # SOLO forward + backward. Es idempotente con zero_grad (no toca el
            # error-feedback del compresor ni el estado del scaler), asi que es
            # seguro reintentarlo ante un OOM transitorio; el grueso de la VRAM
            # (activaciones/gradientes) se asigna aqui. El resto del paso
            # (unscale/compress/clip/step) es STATEFUL y va fuera del reintento.
            self.optimizer.zero_grad()
            if self.scaler is not None:
                with self.precision_manager.forward_context():
                    loss = loss_fn(self.model, batch)
                self.scaler.scale(loss).backward()
            else:
                loss = loss_fn(self.model, batch)
                loss.backward()
            return loss

        def _finish_step():
            if self.scaler is not None:
                # unscale_ solo puede llamarse UNA VEZ por step
                self.scaler.unscale_(self.optimizer)
                if self.config.gradient_compression != GradientCompression.NONE:
                    self._compress_gradients()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                # ElasticRank soft sleep: congelar direcciones dormidas (grad=0)
                # justo antes del step. Barato, sin tocar shapes.
                if self.elastic_rank is not None:
                    self.elastic_rank.mask_gradients(self.model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.config.gradient_compression != GradientCompression.NONE:
                    self._compress_gradients()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if self.elastic_rank is not None:
                    self.elastic_rank.mask_gradients(self.model)
                self.optimizer.step()

        for batch_idx, batch in enumerate(train_loader):
            batch = self._to_device(batch)

            # Recuperacion de OOM transitorio: solo el forward/backward (el grueso
            # de la VRAM y re-ejecutable de forma idempotente) se reintenta. Las
            # operaciones stateful (unscale/compress/step) corren una sola vez para
            # no duplicar el error-feedback ni el estado del scaler.
            if self.vram_governor is not None:
                loss = self.vram_governor.oom_guarded(lambda b=batch: _fwd_bwd(b))
            else:
                loss = _fwd_bwd(batch)
            _finish_step()

            total_loss += loss.item()
            num_batches += 1

            self._global_step += 1

            if self.elastic_rank is not None:
                # Avanzar el mini-warmup de direcciones revividas (gate 0->1).
                self.elastic_rank.tick_wake(self.model)
                # Observacion barata cada N steps: EMAs, contadores, marcar
                # soft-sleep. NO reconstruye el optimizer (eso es por epoch).
                if (self._global_step
                        % self.config.elastic_rank_check_interval == 0):
                    self.elastic_rank.observe(
                        self.model,
                        self.optimizer,
                        in_warmup=self._rank_growth_warmup_remaining > 0,
                        epoch=epoch,
                    )

            # VRAM Governor: observacion barata intra-epoch (presion EMA).
            # Inerte sin CUDA. Las decisiones estructurales (gating de
            # growth) se aplican en el boundary de epoch, no aqui.
            if (self.vram_governor is not None
                    and self._global_step
                    % self.config.vram_governor_interval == 0):
                self.vram_governor.observe(self.device)

            # Aplicar warmup post rank-growth (step a step)
            if self._rank_growth_warmup_remaining > 0:
                self._apply_rank_growth_warmup_step()

            # Refactorizacion periodica anti-rank-collapse
            if (self.config.refactorize_interval > 0
                    and self._global_step % self.config.refactorize_interval == 0):
                current_rank = (
                    self.rank_scheduler.current_rank
                    if self.rank_scheduler else self.config.initial_rank
                )
                num_refactorized = refactorize_model(
                    self.model, self.optimizer, current_rank,
                    (ZFactorizedLinear, ZSparseFactorizedLinear),
                )
                # ElasticRank: la SVD fresca reordena el espectro -> el estado
                # por-capa (sleep_mask/wake_gate/_states) queda en la base
                # antigua. Resetear (si no, mask_gradients congela direcciones
                # equivocadas) y ademas aplicar gracia para estabilizar EMAs.
                if num_refactorized > 0 and self.elastic_rank is not None:
                    self.elastic_rank.reset_after_refactorize(self.model)
                    self.elastic_rank.note_refactorize()
                # Activar warmup post-refactorizacion (igual que rank growth).
                # Solo si esta activado: con warmup_steps=0 el LR quedaba clavado
                # en 0.1x y decaia geometricamente en cada refactorizacion.
                if (num_refactorized > 0
                        and self._rank_growth_warmup_remaining <= 0
                        and self.config.rank_growth_warmup_steps > 0):
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self._rank_growth_warmup_remaining = self.config.rank_growth_warmup_steps
                    self._rank_growth_warmup_base_lr = old_lr
                    warmup_start_lr = old_lr * 0.1
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = warmup_start_lr

            if batch_idx % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.debug(
                    f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.6f}"
                )

        return total_loss / max(num_batches, 1)

    def _apply_rank_growth_warmup_step(self) -> None:
        """Aplicar un step del warmup lineal post rank-growth."""
        total = self.config.rank_growth_warmup_steps
        remaining = self._rank_growth_warmup_remaining
        base_lr = self._rank_growth_warmup_base_lr

        progress = 1.0 - (remaining / total)  # 0.0 -> 1.0
        start_factor = 0.1
        lr = base_lr * (start_factor + progress * (1.0 - start_factor))

        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

        self._rank_growth_warmup_remaining -= 1

        # Al terminar, restaurar LR base
        if self._rank_growth_warmup_remaining == 0:
            for pg in self.optimizer.param_groups:
                pg['lr'] = base_lr
            logger.debug(
                f"[MZTrain] Warmup post rank-growth completado, LR restaurado a {base_lr}"
            )

    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
    ) -> float:
        """Validar el modelo.

        Args:
            val_loader: DataLoader de validacion.
            loss_fn: Funcion de perdida fn(model, batch) -> loss.

        Returns:
            Loss promedio de validacion.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = self._to_device(batch)

                loss = loss_fn(self.model, batch)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _compress_gradients(self) -> None:
        """Aplicar compresion de gradientes a todos los parametros."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad.data = self.grad_compressor.compress(
                    name, param.grad.data
                )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: Callable,
        epochs: int = 50,
        early_stopping_patience: int = 7,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """Entrenamiento completo con todas las optimizaciones MZTrain.

        Args:
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validacion (opcional).
            loss_fn: Funcion de perdida fn(model, batch) -> loss.
            epochs: Numero de epochs.
            early_stopping_patience: Paciencia para early stopping.
            scheduler: Learning rate scheduler opcional.
            callbacks: Lista de callbacks fn(engine, epoch, metrics).

        Returns:
            Diccionario con metricas finales del entrenamiento.
        """
        # Limpiar el estado GLOBAL (de clase) del checkpointing de activaciones
        # de cualquier run/instancia previa: al ser estado de clase se arrastra
        # entre entrenamientos y acumula entradas no liberadas.
        ZActivationCheckpoint.reset()

        if self.config.rank_schedule == RankSchedule.SPECTRAL:
            self.rank_scheduler = ZSpectralRankScheduler(
                model=self.model,
                initial_rank=self.config.initial_rank,
                max_rank=self.config.max_rank,
                total_epochs=epochs,
                energy_threshold=self.config.rank_energy_threshold,
                energy_ceiling=self.config.rank_energy_ceiling,
                sample_layers=self.config.rank_sample_layers,
                growth_interval=self.config.rank_growth_interval,
                growth_factor=self.config.rank_growth_factor,
                factorized_cls=ZFactorizedLinear,
            )
        else:
            self.rank_scheduler = ZRankScheduler(
                initial_rank=self.config.initial_rank,
                max_rank=self.config.max_rank,
                total_epochs=epochs,
                schedule=self.config.rank_schedule,
                growth_interval=self.config.rank_growth_interval,
                growth_factor=self.config.rank_growth_factor,
            )

        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        # Batch sonda FIJO capturado una vez. Lo usan tanto el probe
        # diagnostico como el loss-guard (rollback de compactacion).
        self._probe_batch = None
        self._loss_fn = loss_fn
        need_probe = self.elastic_rank is not None and (
            getattr(self.config, "elastic_rank_probe_loss_sensitivity", False)
            or getattr(self.config, "elastic_rank_loss_guard_enabled", False)
        )
        if need_probe:
            try:
                self._probe_batch = self._to_device(next(iter(train_loader)))
            except StopIteration:
                self._probe_batch = None
        probe_batch = self._probe_batch

        def _probe_loss() -> float:
            return float(loss_fn(self.model, probe_batch))

        logger.info(
            f"[MZTrain] Iniciando entrenamiento: {epochs} epochs, "
            f"rank {self.config.initial_rank} -> {self.config.max_rank}"
        )

        for epoch in range(epochs):
            epoch_start = time.time()

            # 1. Verificar crecimiento de rango
            new_rank = self.rank_scheduler.get_rank(
                epoch,
                current_loss=(
                    self._metrics["train_losses"][-1]
                    if self._metrics["train_losses"] else None
                ),
            )
            # VRAM Governor: medir presion y, si la hay, vetar el
            # crecimiento de rango (inerte sin CUDA / si esta deshabilitado).
            if self.vram_governor is not None:
                self.vram_governor.observe(self.device)
                new_rank = self.vram_governor.approve_rank_growth(
                    new_rank, self.rank_scheduler.current_rank)

            grow_requested = new_rank > self.rank_scheduler.current_rank

            if self.elastic_rank is not None:
                # ElasticRank fusiona sleep/revive/prune + grow en UNA sola
                # operacion estructural por epoch: una reconstruccion del
                # optimizer, una migracion de estados, un warmup.
                plan = self.elastic_rank.build_plan(
                    self.model, new_rank, grow_requested, epoch
                )
                if plan.has_changes():
                    self._apply_elastic_topology_guarded(plan, epoch)
                    logger.info(
                        f"[MZTrain] ElasticRank epoch {epoch}: {plan.summary()}"
                    )
                elif plan.pruned > 0:
                    logger.info(
                        f"[MZTrain] ElasticRank epoch {epoch}: "
                        f"podados {plan.pruned} sleepers (sin cambio estructural)"
                    )
                if grow_requested:
                    self.rank_scheduler.current_rank = new_rank
            elif grow_requested:
                self._grow_model_rank(new_rank)
                self.rank_scheduler.current_rank = new_rank

            # 2. Entrenar epoch
            train_loss = self.train_epoch(train_loader, loss_fn, epoch)
            self._metrics["train_losses"].append(train_loss)
            self._metrics["ranks"].append(self.rank_scheduler.current_rank)

            # 2b. Probe de loss-sensitivity (diagnostico; no muta nada).
            # Gated por su PROPIA flag: probe_batch puede existir solo para
            # el loss-guard, sin que el diagnostico deba ejecutarse.
            if (probe_batch is not None
                    and getattr(self.config,
                                "elastic_rank_probe_loss_sensitivity", False)
                    and epoch % self.config.elastic_rank_probe_interval_epochs
                    == 0):
                self.elastic_rank.probe_loss_sensitivity(
                    self.model, self.optimizer, _probe_loss, epoch
                )

            # 3. Validar
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader, loss_fn)
                self._metrics["val_losses"].append(val_loss)

            # 4. Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    scheduler.step()

            # 5. Early stopping
            check_loss = val_loss if val_loss is not None else train_loss
            if check_loss < best_val_loss:
                best_val_loss = check_loss
                patience_counter = 0
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"[MZTrain] Early stopping en epoch {epoch} "
                    f"(paciencia {early_stopping_patience})"
                )
                self._restore_best_state()
                break

            # 6. Metricas de tiempo y memoria
            epoch_time = time.time() - epoch_start
            self._metrics["time_per_epoch"].append(epoch_time)

            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                self._metrics["memory_mb"].append(mem_mb)
                # Resetear peak stats periodicamente para medir consumo por ventana
                if self.config.memory_log_interval > 0 and epoch % self.config.memory_log_interval == 0:
                    torch.cuda.reset_peak_memory_stats()

            # 7. Log
            val_str = f", val_loss={val_loss:.6f}" if val_loss is not None else ""
            mem_str = (
                f", mem={self._metrics['memory_mb'][-1]:.0f}MB"
                if self._metrics['memory_mb'] else ""
            )

            logger.info(
                f"[MZTrain] Epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.6f}{val_str}, "
                f"rank={self.rank_scheduler.current_rank}, "
                f"time={epoch_time:.1f}s{mem_str}"
            )

            # 8. Callbacks
            if callbacks:
                epoch_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "rank": self.rank_scheduler.current_rank,
                    "time": epoch_time,
                }
                for cb in callbacks:
                    cb(self, epoch, epoch_metrics)

        total_time = time.time() - start_time
        self._metrics["total_train_time"] = total_time

        logger.info(
            f"[MZTrain] ========== ENTRENAMIENTO COMPLETO ==========\n"
            f"  Tiempo total:           {total_time:.1f}s ({total_time/60:.1f}min)\n"
            f"  Mejor loss:             {best_val_loss:.6f}\n"
            f"  Rango final:            {self.rank_scheduler.current_rank}\n"
            f"  Epochs completados:     {len(self._metrics['train_losses'])}\n"
            f"  Optimizer memory:       {self.optimizer.get_memory_stats()}\n"
            f"  ================================================="
        )

        return self.get_training_summary()

    def get_training_summary(self) -> Dict[str, Any]:
        """Obtener resumen completo del entrenamiento.

        Returns:
            Diccionario con todas las metricas del entrenamiento.
        """
        layer_stats = []
        for name, module in self.model.named_modules():
            if isinstance(module, ZFactorizedLinear):
                stats = module.get_stats()
                stats["name"] = name
                layer_stats.append(stats)

        return {
            "train_losses": self._metrics["train_losses"],
            "val_losses": self._metrics["val_losses"],
            "ranks": self._metrics["ranks"],
            "memory_mb": self._metrics["memory_mb"],
            "time_per_epoch": self._metrics["time_per_epoch"],
            "total_time_s": self._metrics["total_train_time"],
            "optimizer_stats": self.optimizer.get_memory_stats(),
            "gradient_stats": self.grad_compressor.get_stats(),
            "activation_stats": ZActivationCheckpoint.get_stats(),
            "precision_stats": self.precision_manager.get_stats(),
            "elastic_rank_stats": (
                self.elastic_rank.get_stats()
                if self.elastic_rank is not None else None
            ),
            "governor_stats": (
                self.vram_governor.get_stats()
                if self.vram_governor is not None else None
            ),
            "layer_stats": layer_stats,
            "config": {
                "initial_rank": self.config.initial_rank,
                "max_rank": self.config.max_rank,
                "rank_schedule": self.config.rank_schedule.value,
                "gradient_compression": self.config.gradient_compression.value,
                "compress_optimizer": self.config.compress_optimizer_states,
                "use_amp": self.config.use_amp,
            },
        }

    def export_full_model(self) -> nn.Module:
        """Exportar modelo con pesos completos (reconstruir desde factores).

        Convierte ZFactorizedLinear -> nn.Linear con W = U @ diag(S) @ V.
        Tambien convierte ZFactorizedAttention y ZFactorizedTransformerBlock,
        reconstruyendo recursivamente todas sus capas internas.

        Util para deployment donde no se necesita MZTrain.

        Returns:
            Modelo con capas nn.Linear estandar.
        """
        model = copy.deepcopy(self.model)

        model = self._defactorize_recursive(model)

        return model

    def _defactorize_recursive(self, model: nn.Module) -> nn.Module:
        """Reconstruir todas las capas factorizadas a nn.Linear, recursivamente.

        Maneja ZFactorizedLinear, ZSparseFactorizedLinear, y modulos anidados
        como ZFactorizedAttention y ZFactorizedTransformerBlock.

        Usa multiples pasadas para manejar reemplazos anidados: primero las
        hojas (ZFactorizedLinear), luego re-escanea por si quedan.
        """
        # Multiples pasadas hasta que no queden capas factorizadas
        for _ in range(5):  # max 5 pasadas (suficiente para cualquier profundidad)
            found = False
            for name, module in list(model.named_modules()):
                if isinstance(module, ZFactorizedLinear):
                    linear = nn.Linear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        device=module.U.device,
                    )
                    linear.weight.data = module.reconstruct_weight()
                    if module.bias is not None:
                        linear.bias.data = module.bias.data.clone()
                    if name == "":
                        model = linear
                    else:
                        self._replace_module(model, name, linear)
                    found = True
            if not found:
                break
        return model

    def save_checkpoint(self, path: str) -> None:
        """Guardar checkpoint (modelo factorizado + optimizer + metricas).

        Args:
            path: Ruta del archivo de checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self._metrics,
            "config": {
                "initial_rank": self.config.initial_rank,
                "max_rank": self.config.max_rank,
                "rank_schedule": self.config.rank_schedule.value,
                "learning_rate": self.config.learning_rate,
            },
            "rank_scheduler": {
                "current_rank": (
                    self.rank_scheduler.current_rank
                    if self.rank_scheduler else self.config.initial_rank
                ),
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"[MZTrain] Checkpoint guardado: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Cargar checkpoint.

        Args:
            path: Ruta del archivo de checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Reconstruir la topologia al rango del checkpoint ANTES de cargar: si
        # el entrenamiento habia crecido el rango, el state_dict guardado tiene
        # tensores mas grandes que el modelo recien construido (initial_rank),
        # y load_state_dict(strict=True) fallaria con un size mismatch. El
        # rango objetivo se infiere del propio state_dict (longitud de los
        # valores singulares 'S'), mas robusto que el rank_scheduler guardado,
        # que puede estar ausente o desincronizado con las formas reales.
        sd = checkpoint["model_state_dict"]
        # Reconstruir CADA capa factorizada a SU rango del checkpoint (longitud
        # de su 'S'). ElasticRank guarda rangos por-capa distintos, asi que un
        # unico rango uniforme daria size mismatch en las capas de menor rango.
        grown_any = False
        for name, module in self.model.named_modules():
            if isinstance(module, ZFactorizedLinear):
                s_val = sd.get(f"{name}.S" if name else "S")
                if s_val is not None and hasattr(s_val, "dim") and s_val.dim() == 1:
                    target = int(s_val.shape[0])
                    if target != module.rank:  # crecer O reducir por-capa
                        module.resize_rank(target)
                        grown_any = True
        if grown_any:
            # recrear el optimizer para que referencie los tensores nuevos; su
            # estado se sobrescribe con el del checkpoint justo despues.
            self.optimizer = self._create_optimizer()

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._metrics = checkpoint.get("metrics", self._metrics)

        if self.rank_scheduler and "rank_scheduler" in checkpoint:
            self.rank_scheduler.current_rank = checkpoint["rank_scheduler"]["current_rank"]

        # No arrastrar un warmup espurio activado por la reconstruccion: el LR
        # correcto ya vino en el optimizer_state_dict cargado.
        self._rank_growth_warmup_remaining = 0

        logger.info(f"[MZTrain] Checkpoint cargado: {path}")
