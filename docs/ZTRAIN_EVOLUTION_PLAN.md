# ZTrain Evolution Plan: Teorias, Algoritmos e Implementacion

**Fecha**: 2026-04-07
**Basado en**: Investigacion del estado del arte 2024-2026 (GaLore, APOLLO, SLTrain, CompAct, PRAC, FlashAttention, etc.)

---

## Tabla de Contenidos

1. [Diagnostico del Estado Actual](#1-diagnostico-del-estado-actual)
2. [TEORIA 1: Proyeccion de Gradientes en Subespacios (ZGaLore)](#2-teoria-1-proyeccion-de-gradientes-en-subespacios-zgalore)
3. [TEORIA 2: Error Feedback para Compresion de Gradientes](#3-teoria-2-error-feedback-para-compresion-de-gradientes)
4. [TEORIA 3: Crecimiento de Rango con Preservacion de Momentum](#4-teoria-3-crecimiento-de-rango-con-preservacion-de-momentum)
5. [TEORIA 4: Schedule de Rango Adaptativo por Espectro](#5-teoria-4-schedule-de-rango-adaptativo-por-espectro)
6. [TEORIA 5: Activation Checkpointing Comprimido](#6-teoria-5-activation-checkpointing-comprimido)
7. [TEORIA 6: Sparse + Low-Rank (SLTrain)](#7-teoria-6-sparse--low-rank-sltrain)
8. [TEORIA 7: Optimizer Hibrido Adaptativo (APOLLO-style)](#8-teoria-7-optimizer-hibrido-adaptativo-apollo-style)
9. [TEORIA 8: Mixed Precision Agresivo FP8/MXFP4](#9-teoria-8-mixed-precision-agresivo-fp8mxfp4)
10. [TEORIA 9: Refactorizacion Periodica Anti-Rank-Collapse](#10-teoria-9-refactorizacion-periodica-anti-rank-collapse)
11. [Bugs Criticos a Corregir](#11-bugs-criticos-a-corregir)
12. [Plan de Fases](#12-plan-de-fases)
13. [Tabla de Impacto](#13-tabla-de-impacto)
14. [Referencias Academicas](#14-referencias-academicas)

---

## 1. Diagnostico del Estado Actual

### Que hace ZTrain hoy

ZTrain factoriza capas `nn.Linear` como `W = U @ diag(S) @ V` (SVD truncada), entrena con
un optimizer Adam que comprime estados a INT8, aplica compresion de gradientes (Top-K, random-K,
cuantizacion), y crece el rango progresivamente durante el entrenamiento.

### Donde falla

| Componente | Problema | Severidad |
|---|---|---|
| `engine.py:275-278` | Doble `scaler.unscale_()` en path AMP + gradient compression | **CRITICO** (RuntimeError) |
| `engine.py:197-203` | Optimizer destruido y recreado en rank growth (pierde momentum) | **CRITICO** |
| `gradient.py` | Compresion Top-K/Random-K sin error feedback — divergencia garantizada | **CRITICO** |
| `engine.py:432-434` | `_best_state` clona modelo completo en GPU (duplica VRAM) | **ALTO** |
| `engine.py:590` | `torch.load` con `weights_only=False` — deserializacion insegura | **ALTO** |
| `engine.py:542-556` | `export_full_model` no maneja ZFactorizedAttention/TransformerBlock | **ALTO** |
| `engine.py:137` | `copy.deepcopy(model)` triplica memoria momentaneamente | **MEDIO** |
| `engine.py:346-349` | Comprime gradientes de TODOS los parametros (incluye biases pequenos) | **MEDIO** |
| `checkpoint.py` | Activation checkpointing importado pero NUNCA usado | **MEDIO** |
| `engine.py:259-265` | No soporta batch como dict (comun en HuggingFace) | **MEDIO** |
| `engine.py:428-429` | Early stopping sobre train_loss (antipatron) | **BAJO** |
| `config.py:165` | `memory_log_interval` se usa para reset, no logging | **BAJO** |

### Donde esta el estado del arte (2025-2026)

- **GaLore 2**: Proyeccion low-rank del GRADIENTE (no del peso) + SVD aleatorizada. Validado 7B/500B tokens.
- **APOLLO**: Memoria nivel SGD con convergencia nivel AdamW. LLaMA-13B con DDP naive en A100-80G.
- **SLTrain/LOST**: Sparse + Low-Rank cierra la brecha de 4.25 puntos de perplexity vs full-rank.
- **CompAct/PRAC**: Compresion de activaciones con random projection / SVD. 36% ahorro.
- **Error Feedback (MicroAdam, ConEF)**: 99% sparsity + error compensation = convergencia competitiva.
- **FP8/MXFP4**: Entrenamiento end-to-end en 4 bits con stochastic rounding (Blackwell GPUs).

---

## 2. TEORIA 1: Proyeccion de Gradientes en Subespacios (ZGaLore)

### Motivacion

ZTrain factoriza los PESOS pero el optimizer (Adam) opera sobre los parametros completos U, S, V.
Los estados del optimizer (m, v) tienen la misma dimensionalidad que los parametros.
GaLore (ICML 2024) demostro que la idea correcta es proyectar el GRADIENTE al subespacio de
bajo rango y ejecutar Adam ahi. Los estados del optimizer se reducen de O(m*n) a O(r*n).

### Papers base

- GaLore: Zhao et al., ICML 2024 Oral (arXiv:2403.03507)
- GaLore 2: Su & Gu et al., Abril 2025 (arXiv:2504.20437)
- GUM (Unbiased GaLore): Pan, Luo, Liu et al., Octubre 2025 (arXiv:2510.17802)
- Flora: Hao et al., ICML 2024 (arXiv:2402.03293)

### Algoritmo Completo

```
CLASE ZGaLoreProjector:
    INICIALIZAR(shape_m, shape_n, rank_r, update_freq_T=200):
        P = None  # Proyector izquierdo (m x r)
        Q = None  # Proyector derecho (n x r)
        step_count = 0

    ACTUALIZAR_SUBESPACIO(gradient_G):
        """Cada T steps, recalcular los proyectores via SVD aleatorizada."""
        # SVD aleatorizada (Halko et al. 2011) - 15x mas rapido que SVD exacta
        # Complejidad: O(m*n*r) en vez de O(m*n*min(m,n))
        U, Sigma, Vt = randomized_svd(G, n_components=r, n_oversamples=10)
        P = U[:, :r]       # m x r
        Q = Vt[:r, :].T    # n x r

    PROYECTAR(gradient_G):
        """Proyectar gradiente al subespacio de bajo rango."""
        step_count += 1
        SI step_count % T == 0:
            ACTUALIZAR_SUBESPACIO(G)
        RETORNAR P.T @ G   # r x n (en vez de m x n)

    RECONSTRUIR(update_low_rank):
        """Reconstruir update en espacio completo."""
        RETORNAR P @ update_low_rank  # m x n
```

```
CLASE ZGaLoreAdam(Optimizer):
    INICIALIZAR(params, lr, betas, rank, update_freq):
        PARA CADA parametro p de dimension (m, n):
            state[p].projector = ZGaLoreProjector(m, n, rank, update_freq)
            state[p].m = zeros(rank, n)   # Primer momento LOW-RANK
            state[p].v = zeros(rank, n)   # Segundo momento LOW-RANK
            state[p].step = 0

    STEP():
        PARA CADA parametro p con gradiente g:
            state[p].step += 1
            t = state[p].step

            # 1. Proyectar gradiente
            g_low = state[p].projector.PROYECTAR(g)  # (rank x n)

            # 2. Adam en espacio low-rank
            state[p].m = beta1 * state[p].m + (1-beta1) * g_low
            state[p].v = beta2 * state[p].v + (1-beta2) * g_low**2

            # Bias correction
            m_hat = state[p].m / (1 - beta1**t)
            v_hat = state[p].v / (1 - beta2**t)

            # Update en espacio low-rank
            update_low = m_hat / (sqrt(v_hat) + eps)

            # 3. Reconstruir y aplicar
            update_full = state[p].projector.RECONSTRUIR(update_low)
            p.data -= lr * update_full
```

### SVD Aleatorizada (Implementacion)

```python
def randomized_svd(M, rank, n_oversamples=10, n_power_iterations=2):
    """
    Halko et al. 2011 - Finding structure with randomness.
    Complejidad: O(m * n * (rank + n_oversamples)) vs O(m * n * min(m,n)) de SVD exacta.
    GaLore 2 reporta 15x speedup sin degradacion medible.
    """
    m, n = M.shape
    k = rank + n_oversamples

    # Paso 1: Random projection para capturar espacio columna
    Omega = torch.randn(n, k, device=M.device, dtype=M.dtype)
    Y = M @ Omega  # m x k

    # Paso 2: Power iterations para refinar (crucial para decaimiento lento de sigma)
    for _ in range(n_power_iterations):
        Y = M @ (M.T @ Y)

    # Paso 3: QR para base ortonormal
    Q, _ = torch.linalg.qr(Y)  # m x k

    # Paso 4: Proyectar y SVD pequena
    B = Q.T @ M    # k x n
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Paso 5: Reconstruir U
    U = Q @ U_hat   # m x k

    return U[:, :rank], S[:rank], Vt[:rank, :]
```

### Ahorro de memoria

```
Ejemplo: Capa 4096 x 4096, rank = 128

Adam estandar:
  m: 4096 * 4096 * 4 bytes = 67 MB
  v: 4096 * 4096 * 4 bytes = 67 MB
  Total estados: 134 MB por capa

ZGaLore Adam:
  m: 128 * 4096 * 4 bytes = 2 MB
  v: 128 * 4096 * 4 bytes = 2 MB
  P: 4096 * 128 * 4 bytes = 2 MB (overhead unico)
  Total estados: 6 MB por capa

Reduccion: ~95.5% por capa
Para modelo con 32 capas: 4.3 GB ahorrados en estados del optimizer
```

### Donde implementar en ZTrain

- **Archivo nuevo**: `src/mztrain/projector.py` — Clase `ZGaLoreProjector`
- **Modificar**: `src/mztrain/optimizer.py` — Reemplazar/extender `ZCompressedAdam`
- **Modificar**: `src/mztrain/engine.py` — Pasar rank y update_freq al optimizer

### Consideraciones

- El SVD aleatorizado cada T=200 steps anade ~2% overhead computacional
- Si se usa con gradient compression (Top-K), aplicar Top-K DESPUES de la proyeccion
- Compatible con AMP (FP16/BF16): proyectar en precision del gradiente, Adam en FP32
- GUM (Unbiased GaLore) corrige sesgo inherente: implementar si se observa gap de convergencia

---

## 3. TEORIA 2: Error Feedback para Compresion de Gradientes

### Motivacion

Top-K y cuantizacion de gradientes son compresores SESGADOS. El valor esperado del gradiente
comprimido NO es igual al gradiente real. Sin compensacion de error, la informacion descartada
en cada paso se pierde PERMANENTEMENTE. Esto esta probado formalmente que puede causar
divergencia o convergencia a soluciones suboptimas.

### Papers base

- Error Feedback: Stich et al., ICML 2018
- MicroAdam: NeurIPS 2024 — 99% sparsity + 4-bit error correction
- ConEF: Contractive Error Feedback, arXiv:2312.08538 — comprimir el buffer de error mismo
- ADEF: Marzo 2025 (arXiv:2503.08427) — primer error feedback acelerado (Nesterov)
- PowerSGD+: Septiembre 2025 (arXiv:2509.11254) — fix convergencia para low-rank compression

### Algoritmo Completo

```
CLASE ZErrorFeedbackCompressor:
    INICIALIZAR(method, top_k_ratio, min_size_to_compress=1024):
        self.method = method
        self.top_k_ratio = top_k_ratio
        self.min_size = min_size_to_compress
        self.error_buffers = {}       # {param_name: tensor}
        self.error_compress = True    # Usar ConEF para comprimir buffers

    COMPRESS(param_name, gradient):
        """Comprimir gradiente con error feedback."""

        # Filtro de tamano: NO comprimir tensores pequenos
        SI gradient.numel() < self.min_size:
            RETORNAR gradient  # biases, layer norms, etc.

        # 1. Compensar con error previo
        SI param_name EN self.error_buffers:
            gradient_compensated = gradient + self.error_buffers[param_name]
        SINO:
            gradient_compensated = gradient

        # 2. Comprimir
        SI self.method == TOP_K:
            compressed = self._top_k(gradient_compensated)
        ELIF self.method == QUANTIZE_1BIT:
            compressed = self._sign_compress(gradient_compensated)
        ELIF self.method == QUANTIZE_INT8:
            compressed = self._int8_quantize(gradient_compensated)
        ELIF self.method == SVD:
            compressed = self._low_rank_compress(gradient_compensated)
        ELIF self.method == RANDOM_K:
            compressed = self._random_k(gradient_compensated)

        # 3. Calcular y almacenar residual (error)
        error = gradient_compensated - compressed

        SI self.error_compress:
            # ConEF: comprimir el error mismo a INT4 para ahorrar memoria
            self.error_buffers[param_name] = self._quantize_error_to_int4(error)
        SINO:
            self.error_buffers[param_name] = error

        RETORNAR compressed

    _top_k(gradient):
        """Top-K sparsification con indices."""
        k = max(1, int(gradient.numel() * self.top_k_ratio))
        values, indices = torch.topk(gradient.abs().flatten(), k)
        mask = torch.zeros_like(gradient.flatten())
        mask[indices] = gradient.flatten()[indices]
        RETORNAR mask.reshape(gradient.shape)

    _sign_compress(gradient):
        """1-bit SGD: solo signo + escala por bloque."""
        # Block-wise para mantener magnitud por region
        block_size = 1024
        result = torch.zeros_like(gradient)
        flat = gradient.flatten()
        PARA i EN range(0, flat.numel(), block_size):
            block = flat[i:i+block_size]
            scale = block.abs().mean()
            signs = block.sign()
            result_flat = result.flatten()
            result_flat[i:i+block_size] = signs * scale
        RETORNAR result.reshape(gradient.shape)

    _int8_quantize(gradient):
        """INT8 block-wise quantization (bitsandbytes-style)."""
        block_size = 2048
        flat = gradient.flatten()
        result = torch.zeros_like(flat)
        PARA i EN range(0, flat.numel(), block_size):
            block = flat[i:i+block_size]
            abs_max = block.abs().max()
            SI abs_max > 0:
                scale = abs_max / 127.0
                quantized = torch.round(block / scale).clamp(-127, 127)
                result[i:i+block_size] = quantized * scale
        RETORNAR result.reshape(gradient.shape)

    _low_rank_compress(gradient):
        """SVD low-rank compression para gradientes matriciales."""
        SI gradient.dim() < 2:
            RETORNAR gradient  # No aplicable a vectores
        rank = max(1, int(min(gradient.shape) * self.top_k_ratio))
        U, S, Vt = randomized_svd(gradient, rank)
        RETORNAR U @ torch.diag(S) @ Vt

    _quantize_error_to_int4(error):
        """ConEF: comprimir buffer de error a INT4 para ahorrar ~80-90% memoria."""
        abs_max = error.abs().max()
        SI abs_max == 0:
            RETORNAR torch.zeros_like(error, dtype=torch.int8)  # Pack 2xINT4 en INT8
        scale = abs_max / 7.0  # INT4 rango: [-7, 7]
        quantized = torch.round(error / scale).clamp(-7, 7).to(torch.int8)
        # Almacenar scale junto con tensor cuantizado
        RETORNAR (quantized, scale)

    _dequant_error_int4(packed):
        """Descomprimir error INT4."""
        quantized, scale = packed
        RETORNAR quantized.float() * scale

    get_stats():
        """Estadisticas de compresion."""
        total_error_memory = 0
        PARA name, buf EN self.error_buffers.items():
            SI isinstance(buf, tuple):  # ConEF format
                total_error_memory += buf[0].numel() * 1  # INT8 bytes
            SINO:
                total_error_memory += buf.numel() * 4  # FP32 bytes
        RETORNAR {
            "method": self.method,
            "error_buffer_memory_mb": total_error_memory / (1024*1024),
            "num_compressed_params": len(self.error_buffers),
        }
```

### Ahorro / Impacto

```
Sin error feedback:
  - Top-K al 10% pierde 90% de la informacion por paso
  - Acumulado: divergencia o convergencia a minimo incorrecto

Con error feedback:
  - Top-K al 1% con error feedback CONVERGE al mismo punto que sin compresion
  - MicroAdam demostro: 99% sparsity + 4-bit error = convergencia competitiva en LLaMA-7B

Memoria del error buffer:
  - Sin ConEF: +100% (un buffer FP32 por parametro comprimido)
  - Con ConEF (INT4): +12.5% (buffer INT4 por parametro comprimido)
```

### Donde implementar en ZTrain

- **Reemplazar**: `src/mztrain/gradient.py` completamente
- **Modificar**: `src/mztrain/engine.py` lineas 346-349 — usar nuevo compressor
- **Config**: Anadir `gradient_error_feedback: bool = True` y `compress_error_buffer: bool = True`

### Integracion con engine.py

```python
# ANTES (engine.py actual, ROTO):
def _compress_gradients(self):
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            param.grad.data = self.grad_compressor.compress(name, param.grad.data)

# DESPUES (corregido):
def _compress_gradients(self):
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            # El compressor maneja: error feedback + filtro de tamano + compresion
            param.grad.data = self.grad_compressor.compress(name, param.grad.data)
            # La interfaz es la misma, pero internamente ahora tiene error feedback
```

---

## 4. TEORIA 3: Crecimiento de Rango con Preservacion de Momentum

### Motivacion

Cuando ZTrain crece el rango (ej: de 32 a 64), DESTRUYE el optimizer y crea uno nuevo.
Esto pierde todos los momentos acumulados de Adam (m y v). Adam necesita cientos de steps
para recalibrar sus estimaciones de varianza. ReLoRA (ICLR 2024) demostro que un mini-warmup
de 50-100 steps despues de cada expansion es CRITICO para evitar divergencia.

### Papers base

- ReLoRA: Lialin et al., ICLR 2024 (arXiv:2307.05695) — jagged warmup tras merge
- Spectral Init: Khodak et al., ICLR 2021 (arXiv:2105.01029) — inicializacion SVD
- LDAdam: ICLR 2025 (arXiv:2410.16103) — rotacion suave de momentum entre subespacios

### Algoritmo Completo

```
FUNCION grow_model_rank_safe(model, optimizer, old_rank, new_rank, config):
    """Crecer rango preservando el estado del optimizer."""

    # ======== PASO 1: Expandir capas factorizadas ========
    PARA CADA modulo EN model.modules():
        SI es ZFactorizedLinear Y modulo.rank < new_rank:
            delta_rank = new_rank - modulo.rank

            # --- Expandir U (m x r_old -> m x r_new) ---
            U_old = modulo.U.data                    # (m, r_old)
            # Inicializacion espectral para nuevas columnas
            # Usar SVD del residual del peso actual vs reconstruido
            W_residual = modulo.original_weight - modulo.reconstruct_weight()
            SI W_residual no es None Y norm(W_residual) > eps:
                U_res, S_res, V_res = truncated_svd(W_residual, delta_rank)
                U_new_cols = U_res * sqrt(S_res)     # (m, delta_rank)
                V_new_rows = sqrt(S_res) * V_res     # (delta_rank, n)
            SINO:
                # Kaiming init (NO zeros — zeros causa colapso)
                U_new_cols = kaiming_uniform(m, delta_rank)
                V_new_rows = kaiming_uniform(delta_rank, n)
                # Escalar por factor pequeno para no perturbar demasiado
                U_new_cols *= 0.01
                V_new_rows *= 0.01

            modulo.U.data = cat([U_old, U_new_cols], dim=1)

            # --- Expandir S (r_old -> r_new) ---
            S_old = modulo.S.data
            SI se uso SVD del residual:
                S_new_vals = S_res[:delta_rank]
            SINO:
                S_new_vals = ones(delta_rank) * S_old.min() * 0.01
            modulo.S.data = cat([S_old, S_new_vals])

            # --- Expandir V (r_old x n -> r_new x n) ---
            V_old = modulo.V.data
            modulo.V.data = cat([V_old, V_new_rows], dim=0)

    # ======== PASO 2: Expandir estados del optimizer (NO recrear) ========
    PARA CADA param_group EN optimizer.param_groups:
        PARA CADA param EN param_group['params']:
            SI param EN optimizer.state:
                state = optimizer.state[param]
                old_shape = state['m'].shape  # shape del momento antiguo

                SI param.shape != old_shape:
                    # El parametro cambio de tamano -> expandir estados
                    PARA key EN ['m', 'v']:  # primer y segundo momento
                        old_state = state[key]
                        new_state = zeros(param.shape, device=param.device)

                        # Copiar estado antiguo en la region correspondiente
                        SI old_state.dim() == 2:
                            new_state[:old_shape[0], :old_shape[1]] = old_state
                        ELIF old_state.dim() == 1:
                            new_state[:old_shape[0]] = old_state

                        # Para v (varianza), inicializar nuevas regiones con eps
                        # para evitar division por cero y pasos gigantes
                        SI key == 'v':
                            mask = new_state == 0
                            new_state[mask] = 1e-8

                        state[key] = new_state

                    # Resetear step count parcialmente para bias correction
                    # NO resetear completamente — solo reducir para que nuevas
                    # regiones tengan bias correction mas agresivo
                    state['step'] = max(1, state['step'] // 2)
            SINO:
                # Parametro nuevo sin estado previo -> lazy init normal
                PASAR

    # ======== PASO 3: Mini-warmup post-expansion ========
    # Reducir LR temporalmente y subir gradualmente
    original_lr = optimizer.param_groups[0]['lr']
    warmup_steps = config.rank_growth_warmup_steps  # default: 50
    warmup_start_factor = 0.1  # empezar al 10% del LR

    # Retornar schedule de warmup para que train_epoch lo aplique
    RETORNAR WarmupSchedule(
        optimizer=optimizer,
        start_lr=original_lr * warmup_start_factor,
        end_lr=original_lr,
        steps=warmup_steps,
    )
```

### Clase WarmupSchedule auxiliar

```python
class WarmupSchedule:
    """Mini-warmup lineal post rank growth."""

    def __init__(self, optimizer, start_lr, end_lr, steps):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = steps
        self.current_step = 0
        # Aplicar LR inicial inmediatamente
        for pg in optimizer.param_groups:
            pg['lr'] = start_lr

    def step(self):
        self.current_step += 1
        if self.current_step >= self.total_steps:
            # Warmup completado, restaurar LR original
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.end_lr
            return True  # Senalizar fin
        else:
            # Interpolacion lineal
            progress = self.current_step / self.total_steps
            lr = self.start_lr + progress * (self.end_lr - self.start_lr)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            return False

    @property
    def is_active(self):
        return self.current_step < self.total_steps
```

### Donde implementar en ZTrain

- **Reemplazar**: `engine.py:_grow_model_rank()` completamente
- **Nuevo**: `src/mztrain/warmup.py` para WarmupSchedule
- **Modificar**: `engine.py:train_epoch()` para aplicar warmup steps
- **Config**: Anadir `rank_growth_warmup_steps: int = 50` y `rank_growth_warmup_factor: float = 0.1`

### Integracion con train loop

```python
# En engine.py train() loop:
warmup_schedule = None

for epoch in range(epochs):
    # Verificar crecimiento de rango
    new_rank = self.rank_scheduler.get_rank(epoch, ...)
    if new_rank > self.rank_scheduler.current_rank:
        warmup_schedule = self._grow_model_rank(new_rank)
        self.rank_scheduler.current_rank = new_rank

    # train_epoch ahora recibe warmup opcional
    train_loss = self.train_epoch(train_loader, loss_fn, epoch, warmup_schedule)

    # Limpiar warmup si termino
    if warmup_schedule and not warmup_schedule.is_active:
        warmup_schedule = None
```

```python
# En train_epoch(), aplicar warmup per-step:
for batch_idx, batch in enumerate(train_loader):
    # ... forward, backward, compress, clip ...
    self.optimizer.step()

    # Aplicar warmup step si esta activo
    if warmup_schedule and warmup_schedule.is_active:
        warmup_schedule.step()
```

---

## 5. TEORIA 4: Schedule de Rango Adaptativo por Espectro

### Motivacion

Los schedules fijos (linear, exponential, cosine) no consideran el estado real del entrenamiento.
El paper de Dynamic Rank Adjustment (2025) demostro que el schedule optimo depende de la fase
del learning rate: low-rank es mas fiel cuando LR es bajo (low noise regime), y full-rank es
mas necesario cuando LR es alto (high noise regime).

Ademas, el rango efectivo de los pesos DECAE durante el entrenamiento (weight decay lo acelera).
Crecer rango sin verificar que el rango actual se esta usando efectivamente es desperdiciar
parametros.

### Papers base

- Dynamic Rank Adjustment: Octubre 2025 (arXiv:2508.08625v3)
- AdaLoRA: Zhang et al., ICLR 2023 (arXiv:2303.10512) — rank adaptativo por importancia
- DRIFT: Febrero 2025 (arXiv:2502.03822) — dynamic rank en diffusion
- DR-LoRA: Enero 2026 (arXiv:2601.04823) — growing > pruning para MoE
- SLTrain benchmark: Mayo 2025 — refactorizacion cada 200 iters mejora 3.85 perplexity

### Algoritmo Completo

```
CLASE ZSpectralRankScheduler:
    INICIALIZAR(initial_rank, max_rank, total_epochs, config):
        self.current_rank = initial_rank
        self.max_rank = max_rank
        self.total_epochs = total_epochs
        self.energy_threshold = config.rank_energy_threshold      # default: 0.90
        self.energy_ceiling = config.rank_energy_ceiling          # default: 0.99
        self.sample_layers = config.rank_sample_layers            # default: 4
        self.min_growth_interval = config.rank_growth_interval    # default: 10 epochs
        self.last_growth_epoch = -min_growth_interval
        self.growth_factor = config.rank_growth_factor            # default: 1.5
        self.loss_history = []
        self.rank_history = []

    GET_RANK(epoch, model, current_loss):
        """Decidir si crecer rango basado en analisis espectral."""

        self.loss_history.append(current_loss)
        self.rank_history.append(self.current_rank)

        # Respetar intervalo minimo entre crecimientos
        SI epoch - self.last_growth_epoch < self.min_growth_interval:
            RETORNAR self.current_rank

        # Ya en rango maximo
        SI self.current_rank >= self.max_rank:
            RETORNAR self.current_rank

        # ======== ANALISIS ESPECTRAL ========
        energy_ratios = []
        PARA i, (name, module) EN enumerate(model.named_modules()):
            SI NOT isinstance(module, ZFactorizedLinear):
                CONTINUAR
            SI i >= self.sample_layers:
                BREAK  # Solo muestrear K capas

            # Computar gradiente acumulado o usar el peso actual
            W = module.reconstruct_weight()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

            # Energia capturada por el rango actual
            total_energy = (S ** 2).sum()
            captured_energy = (S[:self.current_rank] ** 2).sum()
            ratio = captured_energy / total_energy

            energy_ratios.append(ratio.item())

        avg_energy = mean(energy_ratios) SI energy_ratios SINO 1.0

        # ======== DECISION ========

        SI avg_energy < self.energy_threshold:
            # El rango actual captura <90% de la energia → CRECER
            new_rank = min(
                int(self.current_rank * self.growth_factor),
                self.max_rank
            )
            self.last_growth_epoch = epoch
            logger.info(
                f"[ZSpectral] Energia {avg_energy:.3f} < {self.energy_threshold} "
                f"-> Creciendo rango {self.current_rank} -> {new_rank}"
            )
            RETORNAR new_rank

        SI avg_energy > self.energy_ceiling:
            # El rango actual captura >99% → NO crecer
            # Verificar si loss esta estancado (podria necesitar otros ajustes)
            SI len(self.loss_history) >= 5:
                recent = self.loss_history[-5:]
                improvement = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-8)
                SI improvement < 0.001:  # <0.1% mejora en 5 epochs
                    logger.info(
                        f"[ZSpectral] Energia {avg_energy:.3f} alta pero loss estancado. "
                        f"Considerar ajustar LR, no rango."
                    )
            RETORNAR self.current_rank

        # Zona intermedia: usar schedule base (cosine/exponential) como fallback
        RETORNAR self._fallback_schedule(epoch)

    _fallback_schedule(epoch):
        """Schedule cosine como fallback cuando espectro es ambiguo."""
        progress = epoch / self.total_epochs
        # Cosine: lento al inicio, rapido en medio, lento al final
        cos_progress = (1 - math.cos(math.pi * progress)) / 2
        target = self.initial_rank + (self.max_rank - self.initial_rank) * cos_progress
        RETORNAR max(self.current_rank, int(target))  # Solo crecer, nunca decrecer
```

### Donde implementar en ZTrain

- **Reemplazar/Extender**: `src/mztrain/scheduler.py` — anadir `ZSpectralRankScheduler`
- **Config**: Anadir `rank_energy_threshold`, `rank_energy_ceiling`, `rank_sample_layers`
- **Modificar**: `engine.py:train()` — pasar modelo al scheduler para analisis espectral

### Consideraciones

- SVD completa es costosa. Solo muestrear 2-4 capas, no todas.
- Alternativamente, usar SVD aleatorizada truncada (mas rapido, suficiente para estimar energia).
- El analisis espectral se hace SOLO cada `rank_growth_interval` epochs, no cada step.

---

## 6. TEORIA 5: Activation Checkpointing Comprimido

### Motivacion

ZTrain importa `ZActivationCheckpoint` pero NUNCA lo usa. La config tiene
`checkpoint_activations=True` pero el engine lo ignora completamente. Esta es una feature
fantasma que deberia ser el pilar de ahorro de memoria en activaciones.

### Papers base

- GACT: Liu et al., ICML 2022 — compression adaptativa por sensibilidad
- CompAct: NAACL 2025 — random projection para activaciones, 25-50% ahorro
- PRAC: Febrero 2026 (arXiv:2602.23111) — SVD principal + random complement, 36% ahorro
- NeuZip: NeurIPS 2024 (arXiv:2410.20650) — entropy-based lossless, 50% ahorro
- ALAM: ICLR 2024 — average quantization, 10-22.5x compression
- Adacc: Agosto 2025 (arXiv:2508.00806) — MILP-based scheduling adaptativo

### Algoritmo Completo

```
CLASE ZCompressedActivationCheckpoint:
    """Activation checkpointing con compresion adaptativa."""

    INICIALIZAR(config):
        self.compression_method = config.activation_compression_method  # "quantize", "low_rank", "hybrid"
        self.checkpoint_every_n = config.checkpoint_every_n_layers
        self.stored = {}
        self.stats = {"original_bytes": 0, "compressed_bytes": 0, "layers_checkpointed": 0}

    @staticmethod
    FUNCTION checkpoint_function(run_fn, *args, compression_method="quantize"):
        """
        Custom checkpointing que comprime activaciones antes de guardar.
        Usa torch.autograd.Function para integracion con autograd.
        """
        # FORWARD: comprimir y guardar
        CLASE CompressedCheckpointFn(torch.autograd.Function):

            @staticmethod
            forward(ctx, *inputs):
                # Ejecutar funcion sin guardar para autograd
                with torch.no_grad():
                    outputs = run_fn(*inputs)

                # Comprimir inputs para reconstruccion en backward
                compressed_inputs = []
                PARA inp EN inputs:
                    SI isinstance(inp, torch.Tensor) Y inp.requires_grad:
                        compressed_inputs.append(
                            ZCompressedActivationCheckpoint._compress(inp, compression_method)
                        )
                    SINO:
                        compressed_inputs.append(inp)

                ctx.compressed_inputs = compressed_inputs
                ctx.run_fn = run_fn
                RETORNAR outputs

            @staticmethod
            backward(ctx, *grad_outputs):
                # Descomprimir inputs
                inputs = []
                PARA item EN ctx.compressed_inputs:
                    SI isinstance(item, dict) Y "type" EN item:
                        inputs.append(
                            ZCompressedActivationCheckpoint._decompress(item)
                        )
                    SINO:
                        inputs.append(item)

                # Re-ejecutar forward para obtener activaciones exactas de backward
                with torch.enable_grad():
                    outputs = ctx.run_fn(*inputs)

                # Computar gradientes
                torch.autograd.backward(outputs, grad_outputs)
                RETORNAR tuple(inp.grad SI isinstance(inp, torch.Tensor) SINO None
                              PARA inp EN inputs)

        RETORNAR CompressedCheckpointFn.apply(*args)

    @staticmethod
    _compress(tensor, method):
        """Comprimir tensor de activacion."""
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        original_bytes = tensor.numel() * tensor.element_size()

        SI method == "quantize":
            # INT8 block-wise quantization (inspirado en ALAM)
            block_size = 256
            flat = tensor.detach().flatten()
            blocks = flat.reshape(-1, block_size) SI flat.numel() % block_size == 0 \
                     SINO flat  # fallback si no es divisible
            scales = blocks.abs().amax(dim=-1, keepdim=True) / 127.0
            quantized = (blocks / scales.clamp(min=1e-8)).round().to(torch.int8)
            RETORNAR {
                "type": "quantize",
                "data": quantized,
                "scales": scales.to(torch.float16),  # escalas en FP16
                "shape": original_shape,
                "dtype": original_dtype,
            }

        ELIF method == "low_rank":
            # Random projection (CompAct-style)
            SI tensor.dim() < 2:
                RETORNAR {"type": "identity", "data": tensor.detach()}

            # Reshapear a 2D para proyeccion
            batch_dims = tensor.shape[:-1]
            d = tensor.shape[-1]
            mat = tensor.detach().reshape(-1, d)  # (B, d)

            rank = max(1, d // 4)  # Comprimir 4x
            # Generar seed para reproducibilidad sin almacenar la matriz
            seed = torch.randint(0, 2**31, (1,)).item()
            gen = torch.Generator(device=tensor.device).manual_seed(seed)
            R = torch.randn(d, rank, generator=gen, device=tensor.device, dtype=tensor.dtype)
            R /= math.sqrt(rank)

            projected = mat @ R  # (B, rank)

            RETORNAR {
                "type": "low_rank",
                "data": projected,
                "seed": seed,
                "rank": rank,
                "d": d,
                "shape": original_shape,
                "dtype": original_dtype,
            }

        ELIF method == "hybrid":
            # PRAC-style: SVD principal + random para cola
            SI tensor.dim() < 2:
                RETORNAR {"type": "identity", "data": tensor.detach()}

            mat = tensor.detach().reshape(-1, tensor.shape[-1])
            d = tensor.shape[-1]

            # Principal subspace via SVD truncada
            r_principal = max(1, d // 8)
            U, S, Vt = randomized_svd(mat, r_principal)
            principal = U @ torch.diag(S) @ Vt  # Aproximacion principal

            # Residual via random projection
            residual = mat - principal
            r_random = max(1, d // 16)
            seed = torch.randint(0, 2**31, (1,)).item()
            gen = torch.Generator(device=tensor.device).manual_seed(seed)
            R = torch.randn(d, r_random, generator=gen, device=tensor.device, dtype=tensor.dtype)
            R /= math.sqrt(r_random)
            proj_residual = residual @ R  # Proyeccion del residual

            RETORNAR {
                "type": "hybrid",
                "U": U, "S": S, "Vt": Vt,  # Componente principal
                "proj_residual": proj_residual,
                "seed": seed,
                "r_random": r_random,
                "d": d,
                "shape": original_shape,
                "dtype": original_dtype,
            }

    @staticmethod
    _decompress(compressed):
        """Descomprimir tensor de activacion."""
        t = compressed["type"]

        SI t == "identity":
            RETORNAR compressed["data"].requires_grad_(True)

        SI t == "quantize":
            blocks = compressed["data"].float() * compressed["scales"].float()
            tensor = blocks.flatten().reshape(compressed["shape"])
            RETORNAR tensor.to(compressed["dtype"]).requires_grad_(True)

        SI t == "low_rank":
            proj = compressed["data"]
            gen = torch.Generator(device=proj.device).manual_seed(compressed["seed"])
            R = torch.randn(compressed["d"], compressed["rank"],
                           generator=gen, device=proj.device, dtype=compressed["dtype"])
            R /= math.sqrt(compressed["rank"])
            # Pseudo-inverse reconstruction: mat_approx = projected @ R^+
            # Usando R^T como aproximacion (valido para random orthogonal)
            mat = proj @ R.T
            RETORNAR mat.reshape(compressed["shape"]).requires_grad_(True)

        SI t == "hybrid":
            # Reconstruir principal
            principal = compressed["U"] @ torch.diag(compressed["S"]) @ compressed["Vt"]
            # Reconstruir residual
            gen = torch.Generator(device=principal.device).manual_seed(compressed["seed"])
            R = torch.randn(compressed["d"], compressed["r_random"],
                           generator=gen, device=principal.device, dtype=compressed["dtype"])
            R /= math.sqrt(compressed["r_random"])
            residual = compressed["proj_residual"] @ R.T
            mat = principal + residual
            RETORNAR mat.reshape(compressed["shape"]).requires_grad_(True)
```

### Integracion con engine.py

```python
# En _factorize_model o en un nuevo metodo _apply_checkpointing:
def _apply_checkpointing(self, model):
    """Envolver capas con activation checkpointing comprimido."""
    if not self.config.checkpoint_activations:
        return model

    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (ZFactorizedTransformerBlock, nn.TransformerEncoderLayer)):
            if layer_idx % self.config.checkpoint_every_n_layers == 0:
                # Envolver forward del modulo con checkpointing comprimido
                original_forward = module.forward

                def make_checkpointed_forward(orig_fn, method):
                    def checkpointed_forward(*args, **kwargs):
                        return ZCompressedActivationCheckpoint.checkpoint_function(
                            lambda *a: orig_fn(*a, **kwargs),
                            *args,
                            compression_method=method,
                        )
                    return checkpointed_forward

                module.forward = make_checkpointed_forward(
                    original_forward,
                    "quantize" if self.config.activation_compression else "identity"
                )
            layer_idx += 1

    return model
```

### Donde implementar en ZTrain

- **Reescribir**: `src/mztrain/checkpoint.py` completamente
- **Modificar**: `engine.py:__init__()` — llamar `_apply_checkpointing` despues de `_factorize_model`
- **Config**: `activation_compression_method: str = "quantize"` (opciones: "quantize", "low_rank", "hybrid")

### Ahorro esperado

```
Metodo "quantize" (INT8):
  Activaciones: 4x compresion (FP32->INT8) o 2x (FP16->INT8)
  Overhead: <2% throughput (cuantizacion block-wise es rapida)

Metodo "low_rank" (CompAct):
  Activaciones: 4x compresion (proyeccion a rank d/4)
  Overhead: ~5% throughput (matmul de proyeccion)

Metodo "hybrid" (PRAC):
  Activaciones: 6-8x compresion (SVD truncada + random tail)
  Overhead: ~8% throughput (SVD + proyeccion)
  Calidad: Mejor que low_rank puro (estimador sin sesgo)

Para modelo 1B con seq_len=2048, batch=32:
  Sin checkpointing: ~12 GB activaciones
  Con quantize: ~3 GB
  Con hybrid: ~1.5-2 GB
```

---

## 7. TEORIA 6: Sparse + Low-Rank (SLTrain)

### Motivacion

La investigacion (Mayo 2025 benchmarking survey) demostro que el low-rank puro tiene una
brecha de 4.25 puntos de perplexity vs full-rank en modelos 1B. La razon: no puede capturar
el espectro de cola (singular values pequenos pero no despreciables). SLTrain (NeurIPS 2024)
demostro que anadir un componente sparse cierra esta brecha.

### Papers base

- SLTrain: Junio 2024, NeurIPS 2024 (arXiv:2406.02214) — sparse + low-rank
- LOST: Agosto 2025 (arXiv:2508.02668) — SVD-guided sparse allocation
- Lottery Ticket en LoRA: Diciembre 2025 (arXiv:2512.22495) — sparse subnetworks in low-rank

### Algoritmo Completo

```
CLASE ZSparseFactorizedLinear(nn.Module):
    """
    W_approx = U @ diag(S) @ V + S_sparse

    Donde:
    - U @ diag(S) @ V captura el espectro dominante (low-rank)
    - S_sparse captura el espectro de cola (sparse)
    """

    INICIALIZAR(in_features, out_features, rank, sparse_density=0.02,
                bias=True, existing_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.sparse_density = sparse_density

        # === Componente Low-Rank: U (m x r), S (r), V (r x n) ===
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.S = nn.Parameter(torch.empty(rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))

        # === Componente Sparse ===
        num_elements = out_features * in_features
        num_nonzero = max(1, int(num_elements * sparse_density))

        # Generar soporte aleatorio fijo (los INDICES no cambian, solo los VALORES)
        indices = torch.randperm(num_elements)[:num_nonzero]
        row_indices = indices // in_features
        col_indices = indices % in_features
        self.register_buffer('sparse_row_idx', row_indices)
        self.register_buffer('sparse_col_idx', col_indices)

        # Valores entrenables del componente sparse
        self.sparse_values = nn.Parameter(torch.zeros(num_nonzero))

        # Bias
        SI bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        SINO:
            self.bias = None

        # Inicializar
        SI existing_weight no es None:
            self._init_from_weight(existing_weight)
        SINO:
            self._init_random()

    _init_from_weight(self, W):
        """Inicializar desde peso existente usando SVD."""
        U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)

        # Low-rank: top-r componentes
        self.U.data = U_full[:, :self.rank]
        self.S.data = S_full[:self.rank]
        self.V.data = Vt_full[:self.rank, :]

        # Sparse: inicializar desde residual
        W_approx = self.U.data @ torch.diag(self.S.data) @ self.V.data
        residual = W - W_approx

        # Extraer valores del residual en los indices del soporte sparse
        PARA i EN range(self.sparse_values.numel()):
            r, c = self.sparse_row_idx[i], self.sparse_col_idx[i]
            self.sparse_values.data[i] = residual[r, c]

    _init_random(self):
        """Inicializacion Kaiming para entrenamiento desde cero."""
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.ones_(self.S)
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        # Sparse: inicializar pequeno
        nn.init.normal_(self.sparse_values, std=0.01)

    _build_sparse_matrix(self):
        """Reconstruir matriz sparse desde indices y valores."""
        sparse = torch.zeros(self.out_features, self.in_features,
                           device=self.sparse_values.device,
                           dtype=self.sparse_values.dtype)
        sparse[self.sparse_row_idx, self.sparse_col_idx] = self.sparse_values
        RETORNAR sparse

    reconstruct_weight(self):
        """W = U @ diag(S) @ V + S_sparse"""
        low_rank = self.U @ torch.diag(self.S) @ self.V
        sparse = self._build_sparse_matrix()
        RETORNAR low_rank + sparse

    forward(self, x):
        """Forward pass eficiente sin reconstruir W completo."""
        # Low-rank path: x @ V^T @ diag(S) @ U^T (mas eficiente que x @ W)
        # x: (batch, in_features)
        out_lr = x @ self.V.T           # (batch, rank)
        out_lr = out_lr * self.S         # (batch, rank) -- broadcast
        out_lr = out_lr @ self.U.T       # (batch, out_features)

        # Sparse path: x @ S_sparse^T
        # Eficiente via gather/scatter, NO materializar matriz completa
        # Para cada elemento nonzero (r, c) con valor v:
        #   output[..., r] += x[..., c] * v
        out_sparse = torch.zeros(*x.shape[:-1], self.out_features,
                                device=x.device, dtype=x.dtype)
        PARA i EN range(self.sparse_values.numel()):
            r = self.sparse_row_idx[i]
            c = self.sparse_col_idx[i]
            v = self.sparse_values[i]
            out_sparse[..., r] += x[..., c] * v

        output = out_lr + out_sparse

        SI self.bias no es None:
            output += self.bias

        RETORNAR output

    @property
    num_parameters(self):
        """Parametros totales entrenables."""
        lr_params = self.rank * (self.out_features + self.in_features) + self.rank
        sparse_params = self.sparse_values.numel()
        bias_params = self.out_features SI self.bias no es None SINO 0
        RETORNAR lr_params + sparse_params + bias_params

    @property
    full_parameters(self):
        """Parametros equivalentes de la capa full-rank."""
        RETORNAR self.out_features * self.in_features + (self.out_features SI self.bias SINO 0)

    grow_rank(self, new_rank):
        """Crecer rango del componente low-rank."""
        SI new_rank <= self.rank:
            RETORNAR
        delta = new_rank - self.rank

        # Opcion: mover energia del sparse al low-rank
        # Reconstruir peso actual completo
        W = self.reconstruct_weight()
        U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)

        self.U = nn.Parameter(U_full[:, :new_rank])
        self.S = nn.Parameter(S_full[:new_rank])
        self.V = nn.Parameter(Vt_full[:new_rank, :])

        # Re-inicializar sparse desde nuevo residual
        W_approx = self.U.data @ torch.diag(self.S.data) @ self.V.data
        residual = W - W_approx
        PARA i EN range(self.sparse_values.numel()):
            r, c = self.sparse_row_idx[i], self.sparse_col_idx[i]
            self.sparse_values.data[i] = residual[r, c]

        self.rank = new_rank
```

### NOTA: Optimizacion del forward sparse

El loop Python sobre sparse_values es INACEPTABLEMENTE lento. En implementacion real, usar:

```python
# Opcion 1: torch.sparse COO
sparse_matrix = torch.sparse_coo_tensor(
    indices=torch.stack([self.sparse_row_idx, self.sparse_col_idx]),
    values=self.sparse_values,
    size=(self.out_features, self.in_features),
)
out_sparse = torch.sparse.mm(sparse_matrix, x.T).T

# Opcion 2: Custom Triton kernel (mas rapido)
# Opcion 3: torch.scatter_add con indexing precomputado
```

### Donde implementar en ZTrain

- **Nuevo archivo** o extender: `src/mztrain/layers.py` — anadir `ZSparseFactorizedLinear`
- **Modificar**: `engine.py:_factorize_model()` — opcion de crear sparse+low-rank
- **Config**: `use_sparse_component: bool = False`, `sparse_density: float = 0.02`

### Impacto

```
Parametros extra por sparse (density=2%):
  Capa 4096x4096: 4096*4096*0.02 = ~335K valores extra
  Low-rank (r=128): 128*(4096+4096)+128 = ~1.05M
  Total: ~1.39M vs full 16.7M = 91.7% ahorro (vs 93.7% sin sparse)

Perplexity improvement (segun SLTrain paper):
  Low-rank puro (1B model): 18.22
  Sparse+Low-rank (1B model): ~14.5 (estimado con density=2%)
  Full-rank (1B model): 13.97
  Brecha reducida de 4.25 a ~0.5 puntos
```

---

## 8. TEORIA 7: Optimizer Hibrido Adaptativo (APOLLO-style)

### Motivacion

Combinar GaLore (proyeccion low-rank del gradiente) con Adam-mini (v escalar por bloque)
para lograr memoria cercana a SGD con convergencia cercana a AdamW. APOLLO (MLSys 2025,
Outstanding Paper) demostro que esto funciona hasta LLaMA-13B.

### Papers base

- APOLLO: MLSys 2025 Outstanding Paper — SGD-level memory, AdamW-level convergence
- Adam-mini: Zhang et al., ICLR 2025 (arXiv:2406.16793) — 50% reduccion
- GaLore: Zhao et al., ICML 2024 (arXiv:2403.03507)
- CAME: Luo et al., ACL 2023 (arXiv:2307.02047) — confidence-guided correction
- Schedule-Free: Defazio et al., NeurIPS 2024 (arXiv:2405.15682)

### Algoritmo Completo

```
CLASE ZAdaptiveOptimizer(torch.optim.Optimizer):
    """
    Optimizer hibrido que combina:
    1. Proyeccion low-rank del gradiente (GaLore)
    2. Segundo momento escalar por bloque (Adam-mini)
    3. Error compensation para la proyeccion

    Resultado: memoria ~= SGD, convergencia ~= AdamW
    """

    INICIALIZAR(params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                weight_decay=1e-4, rank=128, projection_update_freq=200,
                block_structure="auto"):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.rank = rank
        self.projection_update_freq = projection_update_freq
        self.block_structure = block_structure  # "auto", "per_head", "per_layer"

    @torch.no_grad()
    STEP(self):
        PARA CADA group EN self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']

            PARA CADA p EN group['params']:
                SI p.grad es None:
                    CONTINUAR

                grad = p.grad
                state = self.state[p]

                # === LAZY INIT ===
                SI len(state) == 0:
                    state['step'] = 0
                    state = self._init_state(p, grad, state)

                state['step'] += 1
                t = state['step']

                # === WEIGHT DECAY (decoupled) ===
                SI wd > 0:
                    p.data.mul_(1 - lr * wd)

                # === PARAMETROS GRANDES (>= 2D y suficientes elementos) ===
                SI grad.dim() >= 2 Y grad.numel() > 4096:
                    self._step_projected(p, grad, state, group)

                # === PARAMETROS PEQUENOS (biases, norms, etc.) ===
                SINO:
                    self._step_standard_adam(p, grad, state, group)

    _init_state(self, param, grad, state):
        """Inicializar estado segun tipo de parametro."""
        SI grad.dim() >= 2 Y grad.numel() > 4096:
            m, n = grad.shape[0], grad.shape[1] SI grad.dim() >= 2 SINO grad.numel()
            r = min(self.rank, min(m, n) // 2)

            # Momento m: low-rank (r x n)
            state['m'] = torch.zeros(r, n, device=param.device, dtype=param.dtype)

            # Momento v: ESCALAR por bloque (Adam-mini style)
            # Determinar bloques segun estructura
            num_blocks = self._determine_blocks(param)
            state['v'] = torch.zeros(num_blocks, device=param.device, dtype=torch.float32)
            state['block_size'] = grad.numel() // num_blocks

            # Proyector GaLore
            state['projector'] = ZGaLoreProjector(m, n, r, self.projection_update_freq)
            state['rank'] = r
        SINO:
            # Adam estandar para parametros pequenos
            state['m'] = torch.zeros_like(param)
            state['v'] = torch.zeros_like(param)

        RETORNAR state

    _determine_blocks(self, param):
        """Determinar numero de bloques para v escalar."""
        SI self.block_structure == "auto":
            # Heuristica: 1 bloque por cada 1024 elementos
            RETORNAR max(1, param.numel() // 1024)
        ELIF self.block_structure == "per_head":
            # Para attention: asumimos dim=0 es num_heads * head_dim
            RETORNAR max(1, param.shape[0] // 64)  # 64 = head_dim tipico
        RETORNAR 1  # "per_layer": un solo v escalar

    _step_projected(self, param, grad, state, group):
        """Step para parametros grandes con proyeccion + v escalar."""
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        t = state['step']

        # 1. Proyectar gradiente
        grad_2d = grad.reshape(grad.shape[0], -1) SI grad.dim() >= 2 SINO grad.unsqueeze(0)
        g_low = state['projector'].PROYECTAR(grad_2d)  # (r, n)

        # 2. Actualizar m (low-rank, per-element)
        state['m'].mul_(beta1).add_(g_low, alpha=1-beta1)

        # 3. Actualizar v (escalar por bloque)
        # Calcular media de g^2 por bloque
        g_sq_mean = g_low.square().mean()  # escalar global
        # O per-bloque si num_blocks > 1:
        SI state['v'].numel() > 1:
            block_size = state['block_size']
            flat_g_sq = g_low.square().flatten()
            PARA i EN range(state['v'].numel()):
                start = i * block_size
                end = min(start + block_size, flat_g_sq.numel())
                SI start < end:
                    state['v'][i] = beta2 * state['v'][i] + (1-beta2) * flat_g_sq[start:end].mean()
        SINO:
            state['v'][0] = beta2 * state['v'][0] + (1-beta2) * g_sq_mean

        # 4. Bias correction
        m_hat = state['m'] / (1 - beta1**t)
        v_hat = state['v'] / (1 - beta2**t)

        # 5. Computar update en espacio low-rank
        # v es escalar(es), expandir para division
        SI state['v'].numel() == 1:
            update_low = m_hat / (v_hat[0].sqrt() + eps)
        SINO:
            # Expandir v por bloque a las dimensiones de m
            v_expanded = state['v'].repeat_interleave(
                state['block_size']
            )[:state['m'].numel()].reshape(state['m'].shape)
            v_hat_expanded = v_expanded / (1 - beta2**t)
            update_low = m_hat / (v_hat_expanded.sqrt() + eps)

        # 6. Reconstruir y aplicar
        update_full = state['projector'].RECONSTRUIR(update_low)
        param.data.add_(update_full.reshape(param.shape), alpha=-lr)

    _step_standard_adam(self, param, grad, state, group):
        """Adam estandar para parametros pequenos (biases, norms)."""
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        t = state['step']

        state['m'].mul_(beta1).add_(grad, alpha=1-beta1)
        state['v'].mul_(beta2).addcmul_(grad, grad, value=1-beta2)

        m_hat = state['m'] / (1 - beta1**t)
        v_hat = state['v'] / (1 - beta2**t)

        param.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

    get_memory_stats(self):
        """Estadisticas de uso de memoria."""
        total_state_bytes = 0
        total_full_bytes = 0
        projected_params = 0
        standard_params = 0

        PARA p EN self.state:
            state = self.state[p]
            SI 'projector' EN state:
                # Low-rank m + scalar v + projector
                total_state_bytes += state['m'].numel() * state['m'].element_size()
                total_state_bytes += state['v'].numel() * 4  # FP32
                total_state_bytes += state['rank'] * (p.shape[0] + p.shape[-1]) * 4  # Projector P
                total_full_bytes += p.numel() * 2 * 4  # Lo que seria m+v full en FP32
                projected_params += 1
            SINO:
                total_state_bytes += state['m'].numel() * state['m'].element_size()
                total_state_bytes += state['v'].numel() * state['v'].element_size()
                total_full_bytes += p.numel() * 2 * 4
                standard_params += 1

        RETORNAR {
            "state_memory_mb": total_state_bytes / (1024*1024),
            "full_adam_memory_mb": total_full_bytes / (1024*1024),
            "compression_ratio": total_full_bytes / max(total_state_bytes, 1),
            "projected_params": projected_params,
            "standard_params": standard_params,
        }
```

### Donde implementar en ZTrain

- **Nuevo archivo**: `src/mztrain/adaptive_optimizer.py` — `ZAdaptiveOptimizer`
- **Modificar**: `engine.py:__init__()` — opcion de seleccionar optimizer
- **Config**: `optimizer_type: str = "compressed_adam"` (opciones: "compressed_adam", "adaptive", "galore")

### Ahorro esperado

```
Ejemplo: Modelo 1B con 32 capas de 4096x4096

Adam estandar:
  m: 32 * 4096 * 4096 * 4 = 2.15 GB
  v: 32 * 4096 * 4096 * 4 = 2.15 GB
  Total estados: 4.3 GB

ZAdaptiveOptimizer (rank=128, v escalar por bloque):
  m (low-rank): 32 * 128 * 4096 * 4 = 67 MB
  v (escalar):  32 * (4096*4096/1024) * 4 = 2 MB
  P (proyector): 32 * 4096 * 128 * 4 = 67 MB
  Total estados: ~136 MB

Reduccion: ~96.8% (31.6x menos memoria en optimizer)
```

---

## 9. TEORIA 8: Mixed Precision Agresivo FP8/MXFP4

### Motivacion

ZTrain usa AMP estandar (FP16 forward, FP32 backward). El estado del arte ya esta en
FP8 (produccion en Hopper GPUs) y MXFP4 (emergente en Blackwell). COAT (ICLR 2025) demostro
que comprimir SIMULTANEAMENTE optimizer states + activaciones a FP8 funciona sin degradacion.

### Papers base

- NVIDIA Transformer Engine: FP8 E4M3/E5M2 para Hopper
- MOSS: Noviembre 2025 (arXiv:2511.05811) — two-level microscaling, 34% throughput gain
- COAT: ICLR 2025 — FP8 optimizer + FP8 activaciones juntos
- Quartet: NeurIPS 2025 (arXiv:2505.14669) — FP4 end-to-end training
- MXFP4: Mayo 2025, Castro et al. + Chmiel et al. — sub-4bit training

### Algoritmo Completo

```
CLASE ZMultiPrecisionManager:
    """
    Gestion de precision mixta multi-nivel para ZTrain.

    Niveles:
    - STANDARD: FP16/BF16 forward, FP32 backward (actual ZTrain)
    - FP8: E4M3 forward, E5M2 backward, FP32 master weights
    - AGGRESSIVE: FP8 + INT8 activaciones + INT8 optimizer states
    - EXPERIMENTAL: MXFP4 (requiere Blackwell GPU)
    """

    INICIALIZAR(config, device):
        self.level = config.precision_level  # "standard", "fp8", "aggressive", "experimental"
        self.device = device

        # Detectar capacidad del hardware
        SI device.type == "cuda":
            capability = torch.cuda.get_device_capability()
            self.has_fp8 = capability[0] >= 9  # Hopper (SM90+)
            self.has_mxfp4 = capability[0] >= 10  # Blackwell (SM100+)
        SINO:
            self.has_fp8 = False
            self.has_mxfp4 = False

        # Validar nivel vs hardware
        SI self.level == "fp8" Y NOT self.has_fp8:
            logger.warning("FP8 requiere GPU Hopper+. Fallback a standard.")
            self.level = "standard"
        SI self.level == "experimental" Y NOT self.has_mxfp4:
            logger.warning("MXFP4 requiere GPU Blackwell+. Fallback a fp8/standard.")
            self.level = "fp8" SI self.has_fp8 SINO "standard"

    CONTEXT_FORWARD(self):
        """Context manager para forward pass."""
        SI self.level == "standard":
            RETORNAR torch.amp.autocast("cuda", dtype=torch.bfloat16)
        SI self.level == "fp8":
            RETORNAR torch.amp.autocast("cuda", dtype=torch.float8_e4m3fn)
        SI self.level == "aggressive":
            RETORNAR torch.amp.autocast("cuda", dtype=torch.float8_e4m3fn)
        SI self.level == "experimental":
            # MXFP4 via NVIDIA Transformer Engine
            RETORNAR te.fp8_autocast(enabled=True, fp8_recipe=get_mxfp4_recipe())

    QUANTIZE_WEIGHT_FOR_STORAGE(self, weight):
        """Cuantizar pesos factorizados para almacenamiento."""
        SI self.level EN ("aggressive", "fp8"):
            # FP8 E4M3 para pesos (mas precision mantissa)
            RETORNAR self._to_fp8_e4m3(weight)
        RETORNAR weight

    DEQUANTIZE_WEIGHT_FOR_COMPUTE(self, weight_quantized):
        """Descuantizar para computo."""
        SI weight_quantized.dtype == torch.float8_e4m3fn:
            RETORNAR weight_quantized.to(torch.bfloat16)
        RETORNAR weight_quantized

    QUANTIZE_ACTIVATION_FOR_CHECKPOINT(self, activation):
        """Cuantizar activacion para checkpointing."""
        SI self.level EN ("aggressive", "experimental"):
            # INT8 block-wise para activaciones (mas robusto que FP8 para activaciones)
            RETORNAR self._blockwise_int8(activation)
        RETORNAR activation

    QUANTIZE_GRADIENT(self, gradient):
        """Cuantizar gradiente para comunicacion/almacenamiento."""
        SI self.level == "fp8":
            # E5M2 para gradientes (rango dinamico mas amplio)
            RETORNAR self._to_fp8_e5m2(gradient)
        RETORNAR gradient

    _to_fp8_e4m3(self, tensor):
        """Convertir a FP8 E4M3 con scaling per-tensor."""
        abs_max = tensor.abs().max()
        scale = abs_max / 448.0  # E4M3 max value
        SI scale == 0:
            scale = 1.0
        RETORNAR (tensor / scale).to(torch.float8_e4m3fn), scale

    _to_fp8_e5m2(self, tensor):
        """Convertir a FP8 E5M2 con scaling per-tensor."""
        abs_max = tensor.abs().max()
        scale = abs_max / 57344.0  # E5M2 max value
        SI scale == 0:
            scale = 1.0
        RETORNAR (tensor / scale).to(torch.float8_e5m2), scale

    _blockwise_int8(self, tensor):
        """INT8 block-wise quantization para activaciones."""
        block_size = 256
        flat = tensor.flatten()
        n_blocks = (flat.numel() + block_size - 1) // block_size
        # Pad si necesario
        padded = torch.nn.functional.pad(flat, (0, n_blocks * block_size - flat.numel()))
        blocks = padded.reshape(n_blocks, block_size)
        scales = blocks.abs().amax(dim=1, keepdim=True) / 127.0
        quantized = (blocks / scales.clamp(min=1e-8)).round().clamp(-127, 127).to(torch.int8)
        RETORNAR {
            "quantized": quantized,
            "scales": scales.to(torch.float16),
            "original_shape": tensor.shape,
            "original_numel": tensor.numel(),
        }
```

### Donde implementar en ZTrain

- **Nuevo archivo**: `src/mztrain/precision.py` — `ZMultiPrecisionManager`
- **Modificar**: `engine.py:train_epoch()` — usar precision manager para autocast
- **Modificar**: `engine.py:__init__()` — reemplazar scaler manual por precision manager
- **Config**: `precision_level: str = "standard"` (opciones: "standard", "fp8", "aggressive", "experimental")

### Consideraciones importantes

- FP8 E4M3 tiene rango [-448, 448] — overflow posible sin scaling
- E5M2 tiene rango [-57344, 57344] — mejor para gradientes con spikes
- Stochastic rounding es ESENCIAL en MXFP4/FP4 para evitar sesgo acumulado
- Los singular values (S) de la factorizacion SIEMPRE deben mantenerse en FP32

---

## 10. TEORIA 9: Refactorizacion Periodica Anti-Rank-Collapse

### Motivacion

Investigacion 2024-2025 demuestra que el RANGO EFECTIVO de las matrices de peso DECAE
durante el entrenamiento. Weight decay lo acelera (empuja hacia rank-2 en redes ReLU).
Esto significa que crecer el rango progresivamente puede ser inutil si los componentes
existentes colapsan. La solucion: refactorizar periodicamente.

El benchmark de Mayo 2025 demostro que SLTrain con re-starts cada 200 iteraciones logra
14.37 perplexity (1B model), vs 18.22 sin refactorizacion. Es una mejora de 3.85 puntos
— MASIVA.

### Papers base

- Implicit Low-Rank Bias: 2024 (OpenReview:3zw9NhLhBM) — weight decay causa rank collapse
- SLTrain benchmark: Mayo 2025 (arXiv:2505.22922v1) — refactorizacion cada 200 iters
- LDAdam: ICLR 2025 (arXiv:2410.16103) — rotacion suave de momentum entre subespacios
- ReLoRA: ICLR 2024 (arXiv:2307.05695) — merge + reset periodico

### Algoritmo Completo

```
FUNCION refactorize_model(model, optimizer, rank, config):
    """
    Refactorizacion periodica: reconstruir peso → SVD fresca → actualizar factores.

    Se ejecuta cada config.refactorize_interval iteraciones (default: 200).
    Inspirado en SLTrain restarts + LDAdam rotacion de momentum.
    """

    PARA CADA name, module EN model.named_modules():
        SI NOT isinstance(module, (ZFactorizedLinear, ZSparseFactorizedLinear)):
            CONTINUAR

        # ======== PASO 1: Reconstruir peso completo actual ========
        W_current = module.reconstruct_weight().detach()

        # ======== PASO 2: SVD fresca ========
        U_new, S_new, Vt_new = torch.linalg.svd(W_current, full_matrices=False)
        U_new = U_new[:, :rank]
        S_new = S_new[:rank]
        V_new = Vt_new[:rank, :]

        # ======== PASO 3: Computar matrices de rotacion ========
        # Para rotar momentum del optimizer al nuevo subespacio (LDAdam)
        # P_old: proyector izquierdo antiguo = U_old
        # P_new: proyector izquierdo nuevo = U_new
        # Rotacion: R = P_new^T @ P_old (rank x rank)
        U_old = module.U.data
        rotation_U = U_new.T @ U_old  # (rank_new, rank_old)

        V_old = module.V.data
        rotation_V = V_new @ V_old.T  # (rank_new, rank_old)

        # ======== PASO 4: Actualizar parametros ========
        # Guardar referencias para actualizar optimizer states
        old_U_id = id(module.U)
        old_V_id = id(module.V)
        old_S_id = id(module.S)

        module.U.data = U_new
        module.S.data = S_new
        module.V.data = V_new

        # ======== PASO 5: Rotar estados del optimizer ========
        PARA CADA param_group EN optimizer.param_groups:
            PARA CADA p EN param_group['params']:
                SI id(p) == old_U_id Y p EN optimizer.state:
                    state = optimizer.state[p]
                    # Rotar m: m_new = rotation @ m_old
                    SI 'm' EN state Y state['m'].shape == module.U.shape:
                        # m es per-element en espacio del parametro
                        # Para U: m tiene shape (m, rank_old) → rotar en dim rank
                        state['m'] = state['m'] @ rotation_U.T  # (m, rank_new)
                        # v: resetear parcialmente (nuevas direcciones no tienen historial)
                        state['v'] = state['v'] * 0.5  # Decay v para nuevas direcciones

                SI id(p) == old_V_id Y p EN optimizer.state:
                    state = optimizer.state[p]
                    SI 'm' EN state Y state['m'].shape == module.V.shape:
                        state['m'] = rotation_V @ state['m']  # (rank_new, n)
                        state['v'] = state['v'] * 0.5

        # ======== PASO 6: Actualizar sparse component (si existe) ========
        SI isinstance(module, ZSparseFactorizedLinear):
            W_approx = U_new @ torch.diag(S_new) @ V_new
            residual = W_current - W_approx
            PARA i EN range(module.sparse_values.numel()):
                r, c = module.sparse_row_idx[i], module.sparse_col_idx[i]
                module.sparse_values.data[i] = residual[r, c]

    logger.info(f"[ZTrain] Refactorizacion completada. Rank: {rank}")
```

### Integracion con engine.py

```python
# En train_epoch() o en train() loop:
self.global_step = 0

# En train_epoch, al final de cada batch:
self.global_step += 1
if (self.config.refactorize_interval > 0 and
    self.global_step % self.config.refactorize_interval == 0):
    refactorize_model(
        self.model, self.optimizer,
        self.rank_scheduler.current_rank, self.config
    )
```

### Donde implementar en ZTrain

- **Nuevo archivo o funcion**: `src/mztrain/refactorize.py`
- **Modificar**: `engine.py:train_epoch()` — llamar refactorizacion periodica
- **Config**: `refactorize_interval: int = 200` (0 = deshabilitado)

---

## 11. Bugs Criticos a Corregir

Estos deben arreglarse ANTES de cualquier feature nueva.

### Bug 1: Doble unscale_ en AMP path

**Archivo**: `engine.py:275-278`
**Problema**: `scaler.unscale_()` se llama dos veces en el mismo step cuando gradient compression esta activo.

```python
# ACTUAL (ROTO):
if self.config.gradient_compression != GradientCompression.NONE:
    self.scaler.unscale_(self.optimizer)   # Primera vez
    self._compress_gradients()

self.scaler.unscale_(self.optimizer)       # Segunda vez → RuntimeError

# CORREGIDO:
already_unscaled = False
if self.config.gradient_compression != GradientCompression.NONE:
    self.scaler.unscale_(self.optimizer)
    already_unscaled = True
    self._compress_gradients()

if not already_unscaled:
    self.scaler.unscale_(self.optimizer)

torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
```

### Bug 2: Optimizer destruido en rank growth

Ver TEORIA 3 arriba para la solucion completa.

### Bug 3: _best_state duplica modelo en GPU

```python
# ACTUAL (desperdicia VRAM):
self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

# CORREGIDO: guardar en CPU
self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
```

### Bug 4: weights_only=False inseguro

```python
# ACTUAL (inseguro):
checkpoint = torch.load(path, map_location=self.device, weights_only=False)

# CORREGIDO:
checkpoint = torch.load(path, map_location=self.device, weights_only=True)
# Si falla porque hay objetos custom, usar safe_globals:
# torch.serialization.add_safe_globals([ZFactorizedLinear, ...])
```

### Bug 5: export_full_model incompleto

```python
# ACTUAL: solo maneja ZFactorizedLinear
# CORREGIDO: manejar todos los tipos factorizados
for name, module in list(model.named_modules()):
    if isinstance(module, ZFactorizedLinear):
        # ... conversion existente ...
    elif isinstance(module, ZFactorizedAttention):
        linear = module.to_standard_attention()  # Nuevo metodo a implementar
        self._replace_module(model, name, linear)
    elif isinstance(module, ZFactorizedTransformerBlock):
        block = module.to_standard_block()  # Nuevo metodo a implementar
        self._replace_module(model, name, block)
```

### Bug 6: Batch dict no soportado

```python
# ACTUAL:
if isinstance(batch, (list, tuple)):
    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
elif isinstance(batch, torch.Tensor):
    batch = batch.to(self.device)

# CORREGIDO:
def _to_device(self, batch):
    if isinstance(batch, torch.Tensor):
        return batch.to(self.device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(self._to_device(b) for b in batch)
    elif isinstance(batch, dict):
        return {k: self._to_device(v) for k, v in batch.items()}
    return batch  # non-tensor, dejar como esta

# Uso:
batch = self._to_device(batch)
```

---

## 12. Plan de Fases

### FASE 0: Bug Fixes (Prioridad Maxima)

| Tarea | Archivo | Esfuerzo |
|---|---|---|
| Fix doble unscale_ | engine.py:269-284 | 30 min |
| Fix _best_state a CPU | engine.py:432-434 | 10 min |
| Fix weights_only=True | engine.py:590 | 15 min |
| Fix export_full_model | engine.py:542-556 | 1 hora |
| Fix batch dict support | engine.py:259-265 | 30 min |
| Fix memory_log_interval semantica | engine.py:454 | 15 min |

**Tiempo estimado Fase 0**: ~3 horas

### FASE 1: Fundamentos (Alto Impacto, Complejidad Baja-Media)

| Tarea | Teoria | Archivos | Esfuerzo |
|---|---|---|---|
| Error Feedback Compressor | T2 | gradient.py (rewrite), engine.py | 1 dia |
| Warm Momentum Growth | T3 | engine.py, nuevo warmup.py | 1 dia |
| Refactorizacion Periodica | T9 | nuevo refactorize.py, engine.py | 1 dia |

**Tiempo estimado Fase 1**: ~3 dias

### FASE 2: Memoria (Alto Impacto, Complejidad Media)

| Tarea | Teoria | Archivos | Esfuerzo |
|---|---|---|---|
| GaLore Projector | T1 | nuevo projector.py, optimizer.py | 2 dias |
| Activation Checkpointing Real | T5 | checkpoint.py (rewrite), engine.py | 2 dias |

**Tiempo estimado Fase 2**: ~4 dias

### FASE 3: Convergencia (Impacto Medio-Alto, Complejidad Media)

| Tarea | Teoria | Archivos | Esfuerzo |
|---|---|---|---|
| Sparse + Low-Rank Layer | T6 | layers.py (extend), engine.py | 2 dias |
| Spectral Rank Scheduler | T4 | scheduler.py (extend), engine.py | 1 dia |

**Tiempo estimado Fase 3**: ~3 dias

### FASE 4: Optimizacion Avanzada (Impacto Alto, Complejidad Alta)

| Tarea | Teoria | Archivos | Esfuerzo |
|---|---|---|---|
| Adaptive Optimizer (APOLLO) | T7 | nuevo adaptive_optimizer.py | 3 dias |
| Multi-Precision Manager | T8 | nuevo precision.py, engine.py | 3 dias |

**Tiempo estimado Fase 4**: ~6 dias

### Tiempo total estimado: ~19 dias de desarrollo

---

## 13. Tabla de Impacto

| Teoria | Componente | Ahorro Memoria | Impacto Convergencia | Fase |
|---|---|---|---|---|
| Bug Fixes | Varios | Variable | **Critico** (evita crashes) | 0 |
| T2: Error Feedback | Gradient | 0% (fix) | **Critico** (evita divergencia) | 1 |
| T3: Warm Momentum | Rank Growth | 0% (fix) | **Critico** (preserva convergencia) | 1 |
| T9: Refactorizacion | Layers | 0% | **+3.85 perplexity** | 1 |
| T1: ZGaLore | Optimizer | **60-80% estados** | Neutro | 2 |
| T5: Checkpointing | Activaciones | **40-60%** | <1% degradacion | 2 |
| T6: Sparse+LowRank | Layers | -5% (sparse) | **+4 puntos perplexity** | 3 |
| T4: Spectral Schedule | Scheduler | Indirecto | +5-10% eficiencia | 3 |
| T7: APOLLO Optimizer | Optimizer | **90%+ estados** | Neutro | 4 |
| T8: FP8/MXFP4 | Todo | **50-75%** | <0.5% degradacion | 4 |

### Ahorro combinado proyectado (Modelo 1B, todas las fases):

```
Estado actual ZTrain (estimado):
  Pesos factorizados:     ~400 MB (rank 128)
  Optimizer states:       ~4.3 GB (Adam FP32)
  Activaciones:           ~12 GB (seq_len=2048, batch=32)
  Gradientes:             ~800 MB
  TOTAL GPU:              ~17.5 GB

Con todas las teorias aplicadas:
  Pesos factorizados:     ~420 MB (sparse +20 MB)
  Optimizer states:       ~136 MB (APOLLO: -96.8%)
  Activaciones:           ~2 GB (checkpointing comprimido: -83%)
  Gradientes:             ~80 MB (Top-K 10% + error feedback)
  TOTAL GPU:              ~2.6 GB

  Reduccion total: ~85% (17.5 GB → 2.6 GB)
  Factor: 6.7x menos memoria
```

---

## 14. Referencias Academicas

### Low-Rank Training
- **GaLore**: Zhao et al., ICML 2024 Oral — arXiv:2403.03507
- **GaLore 2**: Su & Gu et al., Abril 2025 — arXiv:2504.20437
- **GUM (Unbiased GaLore)**: Pan, Luo, Liu, Octubre 2025 — arXiv:2510.17802
- **Flora**: Hao et al., ICML 2024 — arXiv:2402.03293
- **ReLoRA**: Lialin et al., ICLR 2024 — arXiv:2307.05695
- **SLTrain**: Junio 2024, NeurIPS 2024 — arXiv:2406.02214
- **LOST**: Agosto 2025 — arXiv:2508.02668
- **DoRA**: Liu et al., ICML 2024 Oral — arXiv:2402.09353
- **AdaLoRA**: Zhang et al., ICLR 2023 — arXiv:2303.10512

### Gradient Compression
- **PowerSGD**: Vogels et al., 2019 — arXiv:1905.13727
- **PowerSGD+**: Septiembre 2025 — arXiv:2509.11254
- **MicroAdam**: NeurIPS 2024 — 99% sparsity + error feedback
- **ConEF**: arXiv:2312.08538 — error feedback comprimido
- **ADEF**: Marzo 2025 — arXiv:2503.08427 — Nesterov + error feedback
- **ARC-Top-K**: Octubre 2025 — arXiv:2510.26709
- **L-GreCo**: MLSys 2024 — K adaptativo por capa

### Optimizer Compression
- **8-bit Adam (bitsandbytes)**: Dettmers et al., ICLR 2022 — arXiv:2110.02861
- **Adam-mini**: Zhang et al., ICLR 2025 — arXiv:2406.16793
- **APOLLO**: MLSys 2025 Outstanding Paper — SGD-level memory
- **CAME**: Luo et al., ACL 2023 — arXiv:2307.02047
- **AdaLomo**: Lv et al., ACL Findings 2024 — arXiv:2310.10195
- **LDAdam**: ICLR 2025 — arXiv:2410.16103
- **SOAP**: Vyas et al., NeurIPS 2024 — arXiv:2409.11321
- **Schedule-Free**: Defazio et al., NeurIPS 2024 — arXiv:2405.15682

### Activation Checkpointing
- **GACT**: Liu et al., ICML 2022
- **CompAct**: NAACL 2025 — random projection activations
- **PRAC**: Febrero 2026 — arXiv:2602.23111 — SVD + random
- **NeuZip**: NeurIPS 2024 — arXiv:2410.20650 — entropy-based
- **ALAM**: ICLR 2024 — average quantization, 10-22.5x compression
- **Adacc**: Agosto 2025 — arXiv:2508.00806 — MILP scheduling

### Progressive Training
- **Apollo (progressive depth)**: AAAI 2024 — arXiv:2401.09192
- **CGLS**: Junio 2025 — arXiv:2506.11389 — depth + curriculum
- **LiGO**: ICLR 2023 — linear growth operator
- **Dynamic Rank Adjustment**: Octubre 2025 — arXiv:2508.08625v3
- **muP**: Cerebras practitioner's guide, Septiembre 2024
- **SALT**: ICLR 2025 — arXiv:2410.18779 — small model guidance

### Training Systems
- **DeepSpeed ZeRO++**: Microsoft — deepspeed.ai
- **FSDP2 / SimpleFSDP**: PyTorch, Noviembre 2024 — arXiv:2411.00284
- **TorchTitan**: ICLR 2025 — arXiv:2410.06511
- **Unsloth**: Triton kernels — github.com/unslothai/unsloth
- **Liger-Kernel**: LinkedIn, ICML 2025 — arXiv:2410.10989
- **FlashAttention-3**: NeurIPS 2024 — arXiv:2407.08608
- **FlashAttention-4**: 2025-2026 — together.ai
- **COAT**: ICLR 2025 — FP8 optimizer + activations
- **MOSS**: Noviembre 2025 — arXiv:2511.05811 — FP8 microscaling
- **Quartet**: NeurIPS 2025 — arXiv:2505.14669 — FP4 training

### Rank Collapse / Spectral
- **Implicit Low-Rank Bias**: 2024 — OpenReview:3zw9NhLhBM
- **Spectral Init + Frobenius Decay**: Khodak et al., ICLR 2021 — arXiv:2105.01029
- **Structured FFN for LLMs**: NeurIPS 2024
- **Benchmarking Low-Rank Training**: Mayo 2025 — arXiv:2505.22922v1
