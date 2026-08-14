"""
MZTrain - Capas factorizadas.

Implementa las capas neuronales que entrenan factores SVD directamente
en lugar de tensores completos, reduciendo drasticamente la memoria.
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("mztrain")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    raise ImportError(
        "PyTorch es requerido para mztrain.layers. "
        "Instalar con: pip install torch>=2.0.0"
    )


# ============================================================================
# CAPA FACTORIZADA: EL CORAZON DE MZTRAIN
# ============================================================================

class ZFactorizedLinear(nn.Module):
    """Capa linear que entrena factores SVD directamente.

    En vez de mantener W (m x n) y computar gradientes para m*n params,
    mantiene U (m x r), S (r), V (r x n) y computa gradientes solo para
    (m*r + r + r*n) parametros.

    Para r << min(m,n), esto es significativamente mas eficiente.

    La reconstruccion es: W = U @ diag(S) @ V
    Los gradientes fluyen a traves de U, S, V via autograd.

    Soporta crecimiento progresivo de rango durante entrenamiento.

    Args:
        in_features: Dimension de entrada.
        out_features: Dimension de salida.
        rank: Rango de factorizacion.
        bias: Si incluir sesgo.
        init_method: Metodo de inicializacion ("svd", "random", "kaiming").
        existing_weight: Tensor de pesos existente para transfer learning.

    Example:
        >>> layer = ZFactorizedLinear(768, 768, rank=64)
        >>> x = torch.randn(32, 768)
        >>> y = layer(x)
        >>> print(f"Compresion: {layer.compression_ratio:.2%}")
        Compresion: 16.71%
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        init_method: str = "svd",
        existing_weight: Optional[torch.Tensor] = None,
        epsi_scaling: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))
        self._max_possible_rank = min(in_features, out_features)

        # Factores entrenables: W = U @ diag(S) @ V
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        self.S = nn.Parameter(torch.empty(self.rank))
        self.V = nn.Parameter(torch.empty(self.rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ElasticRank: estado por-direccion (rango bidireccional).
        # - wake_gate: escala la contribucion de cada direccion en [0, 1].
        #   1.0 = direccion normal; una direccion revivida sube de 0 a 1 en
        #   su mini-warmup (S_efectivo = wake_gate * S).
        # - sleep_mask: True = direccion en 'soft sleep' (congelada, grad=0)
        #   pendiente de compactacion al boundary de epoch.
        # Inertes mientras use_elastic_rank=False (gate=1, mask=False).
        self.register_buffer("wake_gate", torch.ones(self.rank))
        self.register_buffer(
            "sleep_mask", torch.zeros(self.rank, dtype=torch.bool)
        )

        # Opt-in FP8 GEMM. Habilitado externamente por ZTrainEngine cuando
        # config.precision_level == "fp8" y hardware tiene FP8 (SM89+).
        # S nunca cae a FP8 (regla critica: singular values siempre FP32).
        self._use_fp8 = False

        # EPSI: escalar los S retenidos para preservar la energia de Frobenius.
        # Necesario al inicializar (evita el colapso de activaciones, bug #7).
        # Al transferir un peso real para inferencia inmediata, epsi_scaling=False
        # da la SVD truncada fiel al mapa lineal original (ver factorize_existing_model).
        self._epsi_scaling = epsi_scaling

        # Metrica de reconstruccion: la fija _init_from_weight con el error real;
        # queda en 0.0 si no se inicializa desde un peso existente. (Antes se
        # reseteaba a 0.0 DESPUES de _init_from_weight, perdiendo el valor real.)
        self._reconstruction_error = 0.0

        # Inicializar factores
        if existing_weight is not None:
            self._init_from_weight(existing_weight)
        elif init_method == "svd":
            self._init_svd()
        elif init_method == "kaiming":
            self._init_kaiming()
        else:
            self._init_random()

        # Metricas
        self._forward_count = 0

    def _init_svd(self) -> None:
        """Inicializar via SVD truncada con Energy-Preserving Scaling (EPSI).

        Bug anterior: truncar los top-r singular values de una matriz Xavier
        perdia 50-70% de la energia de Frobenius, colapsando la varianza AGREGADA
        de las activaciones (sum_j Var(y_j) = ||W||_F^2). Las activaciones
        decaian a traves de capas profundas, impidiendo convergencia.

        Fix (EPSI):
        1. Usar kaiming_uniform_ como matriz target (mismo init que nn.Linear).
        2. Escalar los singular values retenidos por
           alpha = ||W||_F / ||W_r||_F = sqrt(energia_total / energia_retenida)
           (>= 1), que preserva EXACTAMENTE la energia de Frobenius total.

        ALCANCE de la garantia: EPSI preserva la varianza AGREGADA/pooled
        (sum_j Var(y_j)), no la varianza POR-NEURONA. Un escalar global reparte
        la energia pero no restaura una fila (neurona) que la truncacion vacio:
        con entrada blanca (Cov(x)=I) la suma se conserva pero neuronas
        individuales pueden inflarse o colapsar. La medicion pooled "~1.07-1.16x
        del dense" corresponde a esa cantidad agregada, no a Var(y_j) por neurona.
        """
        W = torch.empty(self.out_features, self.in_features)
        # Kaiming uniform matching nn.Linear default init
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # EPSI: escalar S para preservar la energia de Frobenius total
        total_energy = S.pow(2).sum()
        retained_energy = S[:self.rank].pow(2).sum().clamp(min=1e-12)
        alpha = (total_energy / retained_energy).sqrt()

        with torch.no_grad():
            self.U.copy_(U[:, :self.rank])
            self.S.copy_(S[:self.rank] * alpha)  # <-- EPSI scaling
            self.V.copy_(Vh[:self.rank, :])

    def _init_kaiming(self) -> None:
        """Inicializacion Kaiming para factores individuales."""
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.ones_(self.S)
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        fan_in = self.in_features
        std = 1.0 / math.sqrt(fan_in)
        with torch.no_grad():
            self.S.mul_(std)

    def _init_random(self) -> None:
        """Inicializacion aleatoria simple."""
        nn.init.normal_(self.U, 0, 0.02)
        nn.init.ones_(self.S)
        nn.init.normal_(self.V, 0, 0.02)

    def _init_from_weight(self, W: torch.Tensor) -> None:
        """Inicializar factores desde un tensor de pesos existente.

        Aplica EPSI scaling consistente con _init_svd: cuando el rango es bajo
        respecto al rango efectivo de W, la SVD truncada pierde una fraccion
        significativa de la energia de Frobenius, lo que colapsa la varianza de
        las activaciones a traves de capas profundas (Var(y) ~ rho*Var(y_dense)
        con rho << 1) e impide la convergencia.

        EPSI escala los singular values retenidos por alpha = sqrt(||W||_F / ||W_r||_F)
        para preservar la norma de Frobenius total. Para pesos con energia
        concentrada en los top-r componentes (modelos pre-entrenados) alpha ~= 1
        y el efecto es despreciable; para pesos Kaiming-init alpha puede llegar a
        sqrt(min_dim/r), salvando el modelo del colapso.

        Args:
            W: Tensor de forma (out_features, in_features).

        Raises:
            AssertionError: Si la forma del tensor no coincide.
        """
        assert W.shape == (self.out_features, self.in_features), \
            f"Shape mismatch: {W.shape} vs ({self.out_features}, {self.in_features})"

        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)

        # EPSI: preservar la energia de Frobenius total escalando los S retenidos.
        # Con epsi_scaling=False se usa la SVD truncada pura (alpha=1), que es la
        # mejor aproximacion rango-r del mapa lineal original (fidelidad de mapa),
        # a costa de posible colapso de varianza si el rango es agresivo.
        if getattr(self, "_epsi_scaling", True):
            total_energy = S.pow(2).sum()
            retained_energy = S[:self.rank].pow(2).sum().clamp(min=1e-12)
            alpha = (total_energy / retained_energy).sqrt()
        else:
            alpha = torch.ones((), dtype=S.dtype, device=S.device)

        S_scaled = S[:self.rank] * alpha

        with torch.no_grad():
            self.U.copy_(U[:, :self.rank])
            self.S.copy_(S_scaled)
            self.V.copy_(Vh[:self.rank, :])

        # Reconstruccion error reportado contra los factores efectivos (con EPSI).
        W_recon = (U[:, :self.rank] * S_scaled.unsqueeze(0)) @ Vh[:self.rank, :]
        self._reconstruction_error = float(
            torch.norm(W - W_recon) / max(torch.norm(W).item(), 1e-8)
        )

    def _gated_s(self) -> torch.Tensor:
        """Valores singulares efectivos: S * wake_gate (ElasticRank).

        wake_gate es 1.0 para direcciones normales; una direccion revivida
        sube de 0 a 1 durante su mini-warmup. El guard de tamaño protege
        contra mutadores externos (p. ej. refactorize) que reemplazan U/S/V
        sin tocar los buffers: si hay desajuste, se ignora el gate.
        """
        g = self.wake_gate
        if g.numel() == self.S.numel():
            return self.S * g.to(self.S.dtype)
        return self.S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass eficiente sin reconstruir W completo.

        Computa y = ((x @ V^T) * S) @ U^T en O(batch * in * r + batch * r * out)
        en vez de O(batch * in * out).

        Args:
            x: Tensor de entrada (batch, ..., in_features).

        Returns:
            Tensor de salida (batch, ..., out_features).
        """
        self._forward_count += 1

        s_eff = self._gated_s()

        if self._use_fp8 and x.is_cuda:
            # Path FP8: GEMMs en FP8 e4m3 via torch._scaled_mm (SM89+).
            # S se mantiene FP32 (regla: singular values nunca en baja precision).
            # Output se castea de vuelta al dtype del input para mantener
            # interoperabilidad con capas downstream (dropout, LayerNorm, etc.).
            from .precision import fp8_linear
            in_dtype = x.dtype
            h = fp8_linear(x, self.V, out_dtype=torch.bfloat16)
            h = h * s_eff.to(h.dtype)
            out = fp8_linear(h, self.U, out_dtype=torch.bfloat16)
            out = out.to(in_dtype)
        else:
            # Path estandar: F.linear (BF16/FP32 segun autocast).
            h = F.linear(x, self.V)
            h = h * s_eff.unsqueeze(0)
            out = F.linear(h, self.U)

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out

    def grow_rank(self, new_rank: int, preserve_weights: bool = True) -> None:
        """Incrementar el rango de factorizacion (entrenamiento progresivo).

        Preserva los factores existentes y agrega nuevos componentes
        inicializados cerca de cero para no perturbar el modelo.

        Args:
            new_rank: Nuevo rango objetivo.
            preserve_weights: Si preservar pesos existentes o re-factorizar.
        """
        new_rank = min(new_rank, self._max_possible_rank)
        if new_rank <= self.rank:
            return

        old_rank = self.rank
        device = self.U.device
        dtype = self.U.dtype

        if preserve_weights:
            new_U = torch.zeros(self.out_features, new_rank, device=device, dtype=dtype)
            new_S = torch.zeros(new_rank, device=device, dtype=dtype)
            new_V = torch.zeros(new_rank, self.in_features, device=device, dtype=dtype)

            new_U[:, :old_rank] = self.U.data
            new_S[:old_rank] = self.S.data
            new_V[:old_rank, :] = self.V.data

            noise_scale = float(self.S.data.abs().min()) * 0.01
            nn.init.normal_(new_U[:, old_rank:], 0, 0.01)
            nn.init.normal_(new_V[old_rank:, :], 0, 0.01)
            new_S[old_rank:] = noise_scale

            self.U = nn.Parameter(new_U)
            self.S = nn.Parameter(new_S)
            self.V = nn.Parameter(new_V)
        else:
            W = self.reconstruct_weight()
            self.rank = new_rank
            self.U = nn.Parameter(torch.empty(self.out_features, new_rank, device=device))
            self.S = nn.Parameter(torch.empty(new_rank, device=device))
            self.V = nn.Parameter(torch.empty(new_rank, self.in_features, device=device))
            self._init_from_weight(W)

        self.rank = new_rank
        self._resync_elastic_buffers(old_rank)
        logger.info(
            f"ZFactorizedLinear rank: {old_rank} -> {new_rank} "
            f"({self.out_features}x{self.in_features})"
        )

    def _resync_elastic_buffers(self, old_rank: int) -> None:
        """Reajustar wake_gate/sleep_mask tras un cambio de rango por el
        camino legacy (grow_rank). Las direcciones nuevas entran como normales
        (gate=1, no dormidas); el prefijo existente conserva su estado.
        """
        device = self.S.device
        new_g = torch.ones(self.rank, device=device)
        new_m = torch.zeros(self.rank, dtype=torch.bool, device=device)
        n = min(old_rank, int(self.wake_gate.numel()), self.rank)
        if n > 0:
            new_g[:n] = self.wake_gate[:n].to(device=device, dtype=new_g.dtype)
            new_m[:n] = self.sleep_mask[:n].to(device=device)
        self.wake_gate = new_g
        self.sleep_mask = new_m

    def resize_rank(self, new_rank: int) -> None:
        """Ajustar la topologia (crecer O reducir) a new_rank, reinicializando
        los factores y buffers a esa forma.

        Pensado para reconstruir la topologia ANTES de load_state_dict (los
        valores se sobrescriben con el checkpoint), soportando checkpoints de
        ElasticRank con rango por-capa no uniforme, incluidas capas por debajo
        del rango actual (que grow_rank, solo-crece, no puede reconstruir).
        """
        new_rank = min(new_rank, self._max_possible_rank)
        if new_rank == self.rank:
            return
        old_rank = self.rank
        device = self.U.device
        dtype = self.U.dtype
        self.rank = new_rank
        self.U = nn.Parameter(
            torch.empty(self.out_features, new_rank, device=device, dtype=dtype)
        )
        self.S = nn.Parameter(torch.empty(new_rank, device=device, dtype=dtype))
        self.V = nn.Parameter(
            torch.empty(new_rank, self.in_features, device=device, dtype=dtype)
        )
        self._resync_elastic_buffers(old_rank)

    def elastic_replace_factors(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        V: torch.Tensor,
        wake_gate: torch.Tensor,
        sleep_mask: torch.Tensor,
    ) -> None:
        """Reemplazo atomico de factores + estado ElasticRank.

        Punto de mutacion unico usado por ElasticRankController al compactar
        (dormir) o expandir (revivir/crecer) el rango activo. El optimizer se
        reconstruye y migra fuera de aqui (lo orquesta el engine), igual que
        en el camino de crecimiento existente.

        Args:
            U: Nuevo factor izquierdo (out_features, new_rank).
            S: Nuevos valores singulares (new_rank,).
            V: Nuevo factor derecho (new_rank, in_features).
            wake_gate: Gate por direccion (new_rank,) en [0, 1].
            sleep_mask: Mascara soft-sleep por direccion (new_rank,) bool.
        """
        device = self.U.device
        self.U = nn.Parameter(U.to(device).contiguous())
        self.S = nn.Parameter(S.to(device).contiguous())
        self.V = nn.Parameter(V.to(device).contiguous())
        self.rank = int(S.numel())
        self.wake_gate = wake_gate.to(device=device, dtype=torch.float32).contiguous()
        self.sleep_mask = sleep_mask.to(device=device, dtype=torch.bool).contiguous()

    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruir el tensor de pesos completo W = U @ diag(S) @ V."""
        with torch.no_grad():
            return (self.U * self.S.unsqueeze(0)) @ self.V

    @property
    def num_parameters(self) -> int:
        """Numero total de parametros entrenables."""
        n = self.U.numel() + self.S.numel() + self.V.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n

    @property
    def full_parameters(self) -> int:
        """Numero de parametros equivalente a un nn.Linear completo."""
        n = self.out_features * self.in_features
        if self.bias is not None:
            n += self.out_features
        return n

    @property
    def compression_ratio(self) -> float:
        """Ratio de compresion: params_factorizado / params_completo."""
        return self.num_parameters / self.full_parameters

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadisticas de la capa.

        Returns:
            Diccionario con metricas de la capa factorizada.
        """
        return {
            "shape": f"({self.out_features}, {self.in_features})",
            "rank": self.rank,
            "max_rank": self._max_possible_rank,
            "num_params": self.num_parameters,
            "full_params": self.full_parameters,
            "compression_ratio": f"{self.compression_ratio:.3f}",
            "memory_saved_pct": f"{(1 - self.compression_ratio) * 100:.1f}%",
            "forward_count": self._forward_count,
            "reconstruction_error": f"{self._reconstruction_error:.6f}",
        }

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}/{self._max_possible_rank}, "
            f"compress={self.compression_ratio:.2%}, "
            f"bias={self.bias is not None}"
        )


# ============================================================================
# FACTORIZED ATTENTION
# ============================================================================

class ZFactorizedAttention(nn.Module):
    """Multi-Head Attention con proyecciones Q,K,V,O factorizadas.

    Cada proyeccion lineal usa ZFactorizedLinear, reduciendo la memoria
    de 4 * embed_dim^2 a 4 * (embed_dim * rank * 2 + rank).

    Args:
        embed_dim: Dimension del embedding.
        num_heads: Numero de cabezas de atencion.
        rank: Rango de factorizacion para cada proyeccion.
        dropout: Probabilidad de dropout.
        bias: Si incluir sesgo en proyecciones.

    Example:
        >>> attn = ZFactorizedAttention(512, 8, rank=32)
        >>> x = torch.randn(4, 128, 512)
        >>> out = attn(x)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int = 64,
        dropout: float = 0.1,
        bias: bool = True,
        init_method: str = "svd",
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim debe ser divisible por num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout

        self.q_proj = ZFactorizedLinear(
            embed_dim, embed_dim, rank=rank, bias=bias, init_method=init_method
        )
        self.k_proj = ZFactorizedLinear(
            embed_dim, embed_dim, rank=rank, bias=bias, init_method=init_method
        )
        self.v_proj = ZFactorizedLinear(
            embed_dim, embed_dim, rank=rank, bias=bias, init_method=init_method
        )
        self.out_proj = ZFactorizedLinear(
            embed_dim, embed_dim, rank=rank, bias=bias, init_method=init_method
        )

        # Conservado para backward compat con codigo que accedia self.dropout.
        # SDPA aplica dropout internamente via dropout_p; este modulo no se
        # llama en forward, solo existe para que state_dict no rompa.
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass de atencion multi-cabeza factorizada con Flash Attention.

        Usa torch.nn.functional.scaled_dot_product_attention que en CUDA con
        shapes compatibles llama automaticamente al kernel de FlashAttention
        (~2-4x speedup, ~50% menos memoria de activaciones).

        Args:
            x: Tensor de entrada (batch, seq_len, embed_dim).
            mask: Mascara de atencion opcional. Convencion del modelo:
                  mask == 0 indica posiciones a IGNORAR (compatible con
                  el codigo previo). SDPA espera mascaras booleanas donde
                  True = atender; convertimos antes de llamar.

        Returns:
            Tensor de salida (batch, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Convertir mask al formato de SDPA (True = atender).
        attn_mask = None
        if mask is not None:
            attn_mask = mask != 0
        if causal:
            causal_mask = torch.ones(
                seq_len, seq_len, device=x.device, dtype=torch.bool
            ).tril()
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
            attn_mask = causal_mask if attn_mask is None else (attn_mask & causal_mask)

        # SDPA: aplica scale 1/sqrt(d), softmax, dropout y matmul con V en
        # un solo kernel fused. Backend FlashAttention en CUDA + dtypes
        # compatibles (fp16/bf16/fp32 con shapes pares).
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        return out

    def grow_rank(self, new_rank: int) -> None:
        """Crecer rango en todas las proyecciones Q, K, V, O."""
        self.q_proj.grow_rank(new_rank)
        self.k_proj.grow_rank(new_rank)
        self.v_proj.grow_rank(new_rank)
        self.out_proj.grow_rank(new_rank)


# ============================================================================
# FACTORIZED TRANSFORMER BLOCK
# ============================================================================

class ZFactorizedTransformerBlock(nn.Module):
    """Bloque Transformer completo con todas las capas factorizadas.

    Estructura: LN -> ZFactorizedAttention -> residual -> LN -> ZFactorizedMLP -> residual

    Args:
        embed_dim: Dimension del embedding.
        num_heads: Numero de cabezas de atencion.
        rank: Rango de factorizacion.
        mlp_ratio: Multiplicador para dimension oculta del MLP.
        dropout: Probabilidad de dropout.
        activation: Funcion de activacion ("gelu" o "relu").

    Example:
        >>> block = ZFactorizedTransformerBlock(512, 8, rank=32)
        >>> x = torch.randn(4, 128, 512)
        >>> out = block(x)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        init_method: str = "svd",
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = ZFactorizedAttention(
            embed_dim, num_heads, rank=rank, dropout=dropout, init_method=init_method
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            ZFactorizedLinear(embed_dim, mlp_dim, rank=rank, init_method=init_method),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            ZFactorizedLinear(mlp_dim, embed_dim, rank=rank, init_method=init_method),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass del bloque transformer.

        Args:
            x: Tensor de entrada (batch, seq_len, embed_dim).
            mask: Mascara de atencion opcional.

        Returns:
            Tensor de salida (batch, seq_len, embed_dim).
        """
        x = x + self.attention(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_causal(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass causal para entrenamiento tipo decoder/prefix LM."""
        x = x + self.attention(self.norm1(x), mask=mask, causal=True)
        x = x + self.mlp(self.norm2(x))
        return x

    def grow_rank(self, new_rank: int) -> None:
        """Crecer rango en attention y MLP."""
        self.attention.grow_rank(new_rank)
        for module in self.mlp:
            if isinstance(module, ZFactorizedLinear):
                module.grow_rank(new_rank)


# ============================================================================
# SPARSE + LOW-RANK LAYER (T6)
# ============================================================================

class ZSparseFactorizedLinear(ZFactorizedLinear):
    """Capa linear factorizada con componente sparse: W = U @ diag(S) @ V + S_sparse.

    Extiende ZFactorizedLinear para capturar el espectro de cola de los pesos
    mediante un componente sparse entrenable. El componente low-rank captura
    los valores singulares dominantes, mientras el sparse captura residuales
    importantes que low-rank puro no puede representar.

    Cierra la brecha de perplexidad de ~4.25 puntos (low-rank puro) a ~0.5
    puntos vs full-rank, con un overhead de parametros de ~33%.

    Args:
        in_features: Dimension de entrada.
        out_features: Dimension de salida.
        rank: Rango de factorizacion.
        bias: Si incluir sesgo.
        init_method: Metodo de inicializacion.
        existing_weight: Tensor de pesos existente para transfer learning.
        sparse_density: Fraccion de elementos no-cero en el componente sparse.

    References:
        - SLTrain: NeurIPS 2024 (arXiv:2406.02214)
        - LOST: Agosto 2025 (arXiv:2508.02668)

    Example:
        >>> layer = ZSparseFactorizedLinear(768, 768, rank=64, sparse_density=0.02)
        >>> x = torch.randn(32, 768)
        >>> y = layer(x)
        >>> print(f"Compresion: {layer.compression_ratio:.2%}")
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        init_method: str = "svd",
        existing_weight: Optional[torch.Tensor] = None,
        sparse_density: float = 0.02,
    ):
        self.sparse_density = sparse_density
        # Calcular numero de nonzeros
        total_elements = out_features * in_features
        self._num_sparse = max(1, int(total_elements * sparse_density))

        # Inicializar base (U, S, V) — NO inicializa sparse todavia
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=bias,
            init_method=init_method if existing_weight is None else "svd",
            existing_weight=existing_weight,
        )

        # Generar indices sparse aleatorios (fijos, no entrenables)
        flat_indices = torch.randperm(total_elements)[:self._num_sparse]
        row_idx = flat_indices // in_features
        col_idx = flat_indices % in_features

        self.register_buffer('sparse_row_idx', row_idx)
        self.register_buffer('sparse_col_idx', col_idx)

        # Valores sparse entrenables
        if existing_weight is not None:
            # Inicializar desde residual (operar en el device del peso)
            with torch.no_grad():
                W_lr = (self.U * self.S.unsqueeze(0)) @ self.V
                ew = existing_weight.to(device=W_lr.device, dtype=W_lr.dtype)
                residual = ew - W_lr
                sparse_vals = residual[row_idx.to(W_lr.device), col_idx.to(W_lr.device)]
        else:
            sparse_vals = torch.zeros(self._num_sparse)

        self.sparse_values = nn.Parameter(sparse_vals)

        # Cache de la sparse matrix (invalidar cuando valores cambian)
        self._sparse_matrix_cache = None

    def _build_sparse_matrix(self) -> torch.Tensor:
        """Construir torch.sparse_coo_tensor desde indices y valores."""
        indices = torch.stack([self.sparse_row_idx, self.sparse_col_idx])
        return torch.sparse_coo_tensor(
            indices=indices,
            values=self.sparse_values,
            size=(self.out_features, self.in_features),
        ).coalesce()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: low-rank path + sparse path.

        out = ((x @ V^T) * S) @ U^T + sparse_mm(x) + bias

        Args:
            x: Tensor de entrada (batch, ..., in_features).

        Returns:
            Tensor de salida (batch, ..., out_features).
        """
        self._forward_count += 1

        # Low-rank path (heredado, inlined para evitar doble bias)
        h = F.linear(x, self.V)       # x @ V^T
        h = h * self._gated_s().unsqueeze(0)    # h * S (gated por ElasticRank)
        out = F.linear(h, self.U)      # h @ U^T

        # Sparse path (torch.sparse.mm no soporta FP16 en CUDA)
        # Desactivar autocast explicitamente para evitar cast a Half
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            sparse_matrix = self._build_sparse_matrix().float()
            out_sparse = torch.sparse.mm(sparse_matrix, x_2d.float().T).T
        out_sparse = out_sparse.to(dtype=out.dtype).reshape(*orig_shape[:-1], self.out_features)

        out = out + out_sparse

        if self.bias is not None:
            out = out + self.bias

        return out

    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruir W = U @ diag(S) @ V + sparse."""
        with torch.no_grad():
            W_lr = (self.U * self.S.unsqueeze(0)) @ self.V
            # Materializar sparse a denso para reconstruccion
            W_sparse = torch.zeros_like(W_lr)
            W_sparse[self.sparse_row_idx, self.sparse_col_idx] = self.sparse_values.to(W_lr.dtype)
            return W_lr + W_sparse

    def grow_rank(self, new_rank: int, preserve_weights: bool = True) -> None:
        """Incrementar rango y re-calcular componente sparse desde nuevo residual.

        Args:
            new_rank: Nuevo rango objetivo.
            preserve_weights: Si preservar pesos existentes.
        """
        new_rank = min(new_rank, self._max_possible_rank)
        if new_rank <= self.rank:
            return

        # Reconstruir W completo antes de crecer
        W_full = self.reconstruct_weight().detach()

        # Crecer low-rank (llama a ZFactorizedLinear.grow_rank)
        super().grow_rank(new_rank, preserve_weights=False)
        # grow_rank con preserve_weights=False hace re-SVD sobre W,
        # pero nosotros queremos re-SVD sobre W_full (incluye sparse)
        # Asi que re-hacemos la SVD manualmente
        U, S, Vh = torch.linalg.svd(W_full.float(), full_matrices=False)
        with torch.no_grad():
            self.U.data.copy_(U[:, :self.rank].to(W_full.dtype))
            self.S.data.copy_(S[:self.rank].to(W_full.dtype))
            self.V.data.copy_(Vh[:self.rank, :].to(W_full.dtype))

        # Re-calcular sparse desde nuevo residual
        W_lr_new = (self.U * self.S.unsqueeze(0)) @ self.V
        residual = W_full.to(W_lr_new.dtype) - W_lr_new
        with torch.no_grad():
            self.sparse_values.data.copy_(
                residual[self.sparse_row_idx, self.sparse_col_idx]
            )

    @property
    def num_parameters(self) -> int:
        """Numero total de parametros entrenables (low-rank + sparse)."""
        n = self.U.numel() + self.S.numel() + self.V.numel() + self.sparse_values.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadisticas incluyendo componente sparse."""
        stats = super().get_stats()
        stats.update({
            "sparse_density": self.sparse_density,
            "sparse_nnz": self._num_sparse,
            "sparse_params": self.sparse_values.numel(),
            "sparse_pct_of_total": f"{self.sparse_values.numel() / self.num_parameters * 100:.1f}%",
        })
        return stats

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}/{self._max_possible_rank}, "
            f"sparse_density={self.sparse_density}, "
            f"sparse_nnz={self._num_sparse}, "
            f"compress={self.compression_ratio:.2%}, "
            f"bias={self.bias is not None}"
        )
