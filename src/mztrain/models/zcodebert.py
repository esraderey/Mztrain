"""
MZTrain - ZCodeBERT: Modelo de comprension de codigo con factorizacion SVD.

Arquitectura BERT-Large scale (~355M params equivalentes) que entrena
solo ~67-110M params reales gracias a la factorizacion SVD de MZTrain,
con integracion MNEME para compresion post-entrenamiento.

Componentes:
    - ZCodeBERTConfig: Configuracion del modelo
    - ZCodeBERTEmbeddings: Token + Position + Type embeddings
    - ZCodeBERTEncoder: Stack de ZFactorizedTransformerBlock
    - ZCodeBERTPooler: Pooling con capa factorizada
    - ZCodeBERT: Modelo base
    - ZCodeBERTForMLM: Pre-entrenamiento con Masked Language Modeling
    - ZCodeBERTForSequenceClassification: Fine-tuning para clasificacion
"""

import math
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ZFactorizedLinear, ZFactorizedTransformerBlock
from ..checkpoint import z_checkpoint

logger = logging.getLogger("mztrain")

# MNEME integration (optional)
try:
    from mneme import (
        compress_model,
        get_compression_stats,
        CompressionConfig,
        ZSpace,
        MnemeConfig,
        CompressionLevel,
    )
    HAS_MNEME = True
except ImportError:
    HAS_MNEME = False

try:
    from mneme import SecureStorageBackend, StorageConfig, create_secure_config
    HAS_MNEME_STORAGE = True
except ImportError:
    HAS_MNEME_STORAGE = False


@dataclass
class ZCodeBERTConfig:
    """Configuracion para ZCodeBERT.

    Escala BERT-Large por defecto: 24 capas, 1024 hidden, 16 heads.

    Attributes:
        vocab_size: Tamano del vocabulario.
        hidden_size: Dimension del embedding y capas ocultas.
        num_hidden_layers: Numero de bloques transformer.
        num_attention_heads: Numero de cabezas de atencion.
        intermediate_size: Dimension del MLP intermedio (4x hidden).
        max_position_embeddings: Longitud maxima de secuencia.
        type_vocab_size: Tipos de segmento (2 para sentence pair).
        hidden_dropout_prob: Dropout para embeddings y capas.
        attention_probs_dropout_prob: Dropout para attention weights.
        rank: Rango de factorizacion SVD para capas lineales.
        activation: Funcion de activacion ("gelu" o "relu").
        layer_norm_eps: Epsilon para LayerNorm.
        use_activation_checkpointing: Checkpointing comprimido de activaciones.
        mneme_compression_level: Nivel de compresion MNEME post-entrenamiento.
        factor_init_method: Inicializacion de capas factorizadas
            ("svd", "random", "kaiming"). Usar "kaiming" para modelos grandes
            inicializados desde cero, porque evita materializar pesos densos.
    """
    vocab_size: int = 50000
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    rank: int = 32
    activation: str = "gelu"
    layer_norm_eps: float = 1e-12
    use_activation_checkpointing: bool = True
    mneme_compression_level: str = "balanced"
    factor_init_method: str = "svd"

    @property
    def mlp_ratio(self) -> float:
        return self.intermediate_size / self.hidden_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "rank": self.rank,
            "activation": self.activation,
            "layer_norm_eps": self.layer_norm_eps,
            "use_activation_checkpointing": self.use_activation_checkpointing,
            "mneme_compression_level": self.mneme_compression_level,
            "factor_init_method": self.factor_init_method,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZCodeBERTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def base(cls) -> "ZCodeBERTConfig":
        """Config BERT-Base scale (110M equiv, ~22M real)."""
        return cls(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            rank=32,
        )

    @classmethod
    def large(cls) -> "ZCodeBERTConfig":
        """Config BERT-Large scale (355M equiv, ~67M real)."""
        return cls()  # Default is large

    @classmethod
    def coder_1b(cls, rank: int = 64) -> "ZCodeBERTConfig":
        """Config causal de codigo ~1B parametros densos equivalentes.

        La version comprimida mantiene las proyecciones lineales en factores
        low-rank MZTrain. Usa init Kaiming directo en factores para evitar SVD
        densa durante el arranque.
        """
        return cls(
            hidden_size=1792,
            num_hidden_layers=24,
            num_attention_heads=28,
            intermediate_size=7168,
            rank=rank,
            factor_init_method="kaiming",
            use_activation_checkpointing=True,
        )

    @classmethod
    def small(cls) -> "ZCodeBERTConfig":
        """Config reducida para testing en CPU."""
        return cls(
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            rank=16,
            use_activation_checkpointing=False,
        )


class ZCodeBERTEmbeddings(nn.Module):
    """Embeddings: token + position + token_type + LayerNorm + Dropout.

    Sigue la arquitectura BERT estandar para embeddings.
    """

    def __init__(self, config: ZCodeBERTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Registrar posiciones como buffer (no entrenable)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) IDs de tokens.
            token_type_ids: (batch, seq_len) IDs de tipo de segmento.

        Returns:
            (batch, seq_len, hidden_size) embeddings.
        """
        seq_len = input_ids.size(1)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = self.position_ids[:, :seq_len]

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ZCodeBERTEncoder(nn.Module):
    """Stack de ZFactorizedTransformerBlock.

    Cada bloque usa capas factorizadas SVD para reducir parametros.
    Soporta activation checkpointing comprimido via MNEME.
    """

    def __init__(self, config: ZCodeBERTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            ZFactorizedTransformerBlock(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                rank=config.rank,
                mlp_ratio=config.mlp_ratio,
                dropout=config.hidden_dropout_prob,
                activation=config.activation,
                init_method=config.factor_init_method,
            )
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size).
            attention_mask: (batch, 1, 1, seq_len) broadcastable mask.

        Returns:
            (batch, seq_len, hidden_size) encoder output.
        """
        for i, layer in enumerate(self.layers):
            if self.config.use_activation_checkpointing and self.training:
                hidden_states = self._checkpoint_layer(layer, hidden_states, attention_mask, causal)
            elif causal:
                hidden_states = layer.forward_causal(hidden_states, mask=attention_mask)
            else:
                hidden_states = layer(hidden_states, mask=attention_mask)

        return hidden_states

    @staticmethod
    def _checkpoint_layer(
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal: bool = False,
    ) -> torch.Tensor:
        """Ejecutar capa con activation checkpointing."""
        if attention_mask is None:
            def run_layer(x):
                if causal:
                    return layer.forward_causal(x, mask=None)
                return layer(x, mask=None)
            return z_checkpoint(run_layer, hidden_states)

        def run_layer(x, mask):
            if causal:
                return layer.forward_causal(x, mask=mask)
            return layer(x, mask=mask)

        return z_checkpoint(run_layer, hidden_states, attention_mask)


class ZCodeBERTPooler(nn.Module):
    """Pooler: toma el hidden state del [CLS] token y lo proyecta."""

    def __init__(self, config: ZCodeBERTConfig):
        super().__init__()
        self.dense = ZFactorizedLinear(
            config.hidden_size,
            config.hidden_size,
            rank=config.rank,
            init_method=config.factor_init_method,
        )
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size).

        Returns:
            (batch, hidden_size) pooled output del [CLS] token.
        """
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        pooled = self.activation(pooled)
        return pooled

    def grow_rank(self, new_rank: int) -> None:
        self.dense.grow_rank(new_rank)


class ZCodeBERT(nn.Module):
    """ZCodeBERT - Modelo base de comprension de codigo.

    Arquitectura BERT con capas factorizadas SVD y soporte MNEME.

    ~355M parametros equivalentes con solo ~67M parametros reales (r=32).

    Args:
        config: ZCodeBERTConfig con hiperparametros del modelo.

    Example:
        >>> config = ZCodeBERTConfig.large()
        >>> model = ZCodeBERT(config)
        >>> input_ids = torch.randint(0, 50000, (2, 128))
        >>> output, pooled = model(input_ids)
    """

    def __init__(self, config: ZCodeBERTConfig):
        super().__init__()
        self.config = config

        self.embeddings = ZCodeBERTEmbeddings(config)
        self.encoder = ZCodeBERTEncoder(config)
        self.pooler = ZCodeBERTPooler(config)

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializar embeddings con distribucion normal."""
        for module in [self.embeddings.word_embeddings,
                       self.embeddings.position_embeddings,
                       self.embeddings.token_type_embeddings]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) IDs de tokens.
            attention_mask: (batch, seq_len) 1 para tokens reales, 0 para padding.
            token_type_ids: (batch, seq_len) IDs de tipo de segmento.

        Returns:
            Tuple de:
                - sequence_output: (batch, seq_len, hidden_size)
                - pooled_output: (batch, hidden_size)
        """
        if attention_mask is not None:
            # Convertir (batch, seq_len) -> (batch, 1, 1, seq_len) para broadcasting
            # ZFactorizedAttention espera: 1=keep, 0=mask (masked_fill donde mask==0)
            extended_mask = attention_mask[:, None, None, :].float()
        else:
            extended_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output, extended_mask, causal=causal)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output

    def grow_rank(self, new_rank: int) -> None:
        """Crecer rango de todas las capas factorizadas.

        Args:
            new_rank: Nuevo rango objetivo.
        """
        for layer in self.encoder.layers:
            layer.grow_rank(new_rank)
        self.pooler.grow_rank(new_rank)
        self.config.rank = new_rank
        logger.info(f"[ZCodeBERT] Rango crecido a {new_rank}")

    def get_model_stats(self) -> Dict[str, Any]:
        """Obtener estadisticas del modelo."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Calcular params equivalentes (si fueran capas completas)
        equiv_params = 0
        factorized_count = 0
        for module in self.modules():
            if isinstance(module, ZFactorizedLinear):
                equiv_params += module.full_parameters
                factorized_count += 1

        # Sumar params de embeddings (no factorizados)
        embed_params = sum(
            p.numel() for p in self.embeddings.parameters()
        )
        equiv_params += embed_params

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "equivalent_params": equiv_params,
            "factorized_layers": factorized_count,
            "compression_ratio": total_params / max(equiv_params, 1),
            "rank": self.config.rank,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_hidden_layers,
            "num_heads": self.config.num_attention_heads,
        }

    def compress_with_mneme(self, compression_level: str = "balanced") -> Optional[Dict[str, Any]]:
        """Comprimir modelo exportado con MNEME.

        Aplica compresion SVD/INT8/LZ4 de MNEME sobre los pesos del modelo
        para reducir aun mas el tamano en memoria/disco.

        Args:
            compression_level: "fast", "balanced", "maximum".

        Returns:
            Estadisticas de compresion o None si MNEME no esta disponible.
        """
        if not HAS_MNEME:
            logger.warning(
                "[ZCodeBERT] MNEME no disponible para compresion post-entrenamiento. "
                "Instalar con: pip install mneme"
            )
            return None

        # Exportar a modelo completo primero
        full_model = self._export_full_model()

        # Configurar compresion MNEME
        mneme_config = CompressionConfig()

        # Comprimir
        compressed = compress_model(full_model, config=mneme_config)
        stats = get_compression_stats(compressed)

        logger.info(
            f"[ZCodeBERT] Compresion MNEME completada:\n"
            f"  Capas comprimidas: {stats.get('compressed_layers', 'N/A')}\n"
            f"  Ratio compresion:  {stats.get('compression_ratios', [])}\n"
        )

        return stats

    def save_mneme(self, path: str, key: Optional[str] = None) -> bool:
        """Guardar modelo con MNEME SecureStorageBackend.

        Guarda checkpoint con verificacion HMAC para integridad.

        Args:
            path: Directorio de almacenamiento.
            key: Clave de seguridad opcional.

        Returns:
            True si se guardo exitosamente.
        """
        if not HAS_MNEME_STORAGE:
            logger.warning("[ZCodeBERT] MNEME storage no disponible.")
            # Fallback a torch.save
            torch.save({
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            }, path)
            logger.info(f"[ZCodeBERT] Modelo guardado con torch.save: {path}")
            return True

        storage_config = StorageConfig(storage_path=path)
        storage = SecureStorageBackend(storage_config)

        # Guardar state dict y config
        state = {
            "model_state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "config": self.config.to_dict(),
        }
        storage.save("zcodebert_checkpoint", state)
        logger.info(f"[ZCodeBERT] Modelo guardado con MNEME storage: {path}")
        return True

    def _export_full_model(self) -> nn.Module:
        """Exportar con pesos completos (factores -> matriz completa)."""
        model = copy.deepcopy(self)

        for name, module in list(model.named_modules()):
            if isinstance(module, ZFactorizedLinear):
                linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                linear.weight.data = module.reconstruct_weight()
                if module.bias is not None:
                    linear.bias.data = module.bias.data.clone()

                # Replace in parent
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)
                if parts[-1].isdigit():
                    parent[int(parts[-1])] = linear
                else:
                    setattr(parent, parts[-1], linear)

        return model


class ZCodeBERTForMLM(nn.Module):
    """ZCodeBERT con cabeza de Masked Language Modeling.

    Pre-entrenamiento con objetivo MLM: predecir tokens maskeados.

    Args:
        config: ZCodeBERTConfig con hiperparametros.

    Example:
        >>> config = ZCodeBERTConfig.small()
        >>> model = ZCodeBERTForMLM(config)
        >>> input_ids = torch.randint(0, 50000, (2, 64))
        >>> labels = torch.randint(0, 50000, (2, 64))
        >>> loss, logits = model(input_ids, labels=labels)
    """

    def __init__(self, config: ZCodeBERTConfig):
        super().__init__()
        self.config = config
        self.bert = ZCodeBERT(config)

        # MLM head: hidden -> hidden -> GELU -> LN -> vocab
        self.mlm_dense = ZFactorizedLinear(
            config.hidden_size,
            config.hidden_size,
            rank=config.rank,
            init_method=config.factor_init_method,
        )
        self.mlm_activation = nn.GELU()
        self.mlm_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Tie weights: decoder comparte embeddings
        self.mlm_decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) IDs de tokens (con masking).
            attention_mask: (batch, seq_len) mascara de atencion.
            token_type_ids: (batch, seq_len) tipo de segmento.
            labels: (batch, seq_len) IDs originales, -100 para no-mask.

        Returns:
            Tuple (loss, logits):
                - loss: Scalar si labels proporcionados, None otherwise.
                - logits: (batch, seq_len, vocab_size) predicciones.
        """
        sequence_output, _ = self.bert(input_ids, attention_mask, token_type_ids)

        # MLM head
        hidden = self.mlm_dense(sequence_output)
        hidden = self.mlm_activation(hidden)
        hidden = self.mlm_norm(hidden)
        logits = self.mlm_decoder(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits

    def grow_rank(self, new_rank: int) -> None:
        """Crecer rango de todas las capas factorizadas."""
        self.bert.grow_rank(new_rank)
        self.mlm_dense.grow_rank(new_rank)

    def get_model_stats(self) -> Dict[str, Any]:
        """Obtener estadisticas del modelo completo."""
        stats = self.bert.get_model_stats()
        stats["total_params"] = sum(p.numel() for p in self.parameters())
        stats["task"] = "MLM"
        return stats

    def compress_with_mneme(self, compression_level: str = "balanced"):
        return self.bert.compress_with_mneme(compression_level)

    def save_mneme(self, path: str, key: Optional[str] = None):
        return self.bert.save_mneme(path, key)


class ZCodeBERTForCausalLM(nn.Module):
    """ZCodeBERT en modo causal para completado autoregresivo de codigo.

    Reusa la misma columna vertebral y nombres de cabeza que ZCodeBERTForMLM
    para poder cargar checkpoints MLM existentes. La diferencia principal es
    que el encoder se ejecuta con mascara causal y la loss predice el siguiente
    token (shift-left), lo que alinea el entrenamiento con completado de codigo.
    """

    def __init__(self, config: ZCodeBERTConfig):
        super().__init__()
        self.config = config
        self.bert = ZCodeBERT(config)

        # Mantener nombres mlm_* para compatibilidad con checkpoints MLM.
        self.mlm_dense = ZFactorizedLinear(
            config.hidden_size,
            config.hidden_size,
            rank=config.rank,
            init_method=config.factor_init_method,
        )
        self.mlm_activation = nn.GELU()
        self.mlm_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.mlm_decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Forward causal.

        labels sigue la convencion GPT/HuggingFace: misma forma que input_ids;
        internamente se predice labels[:, 1:] desde logits[:, :-1].
        """
        sequence_output, _ = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            causal=True,
        )

        hidden = self.mlm_dense(sequence_output)
        hidden = self.mlm_activation(hidden)
        hidden = self.mlm_norm(hidden)
        logits = self.mlm_decoder(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generacion autoregresiva simple con greedy o sampling top-k."""
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_position_embeddings:
                break

            if attention_mask is None:
                current_mask = (generated != pad_token_id).long()
            else:
                current_mask = attention_mask[:, :generated.shape[1]].long()

            _loss, logits = self(generated, attention_mask=current_mask)
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, next_logits.shape[-1])
                    values, indices = torch.topk(next_logits, k=k, dim=-1)
                    probs = torch.softmax(values, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1)
                    next_token = indices.gather(-1, sampled)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)],
                    dim=1,
                )

            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        return generated

    def grow_rank(self, new_rank: int) -> None:
        self.bert.grow_rank(new_rank)
        self.mlm_dense.grow_rank(new_rank)

    def get_model_stats(self) -> Dict[str, Any]:
        stats = self.bert.get_model_stats()
        stats["total_params"] = sum(p.numel() for p in self.parameters())
        stats["task"] = "CausalLM"
        return stats

    def compress_with_mneme(self, compression_level: str = "balanced"):
        return self.bert.compress_with_mneme(compression_level)

    def save_mneme(self, path: str, key: Optional[str] = None):
        return self.bert.save_mneme(path, key)


class ZCodeBERTForSequenceClassification(nn.Module):
    """ZCodeBERT con cabeza de clasificacion de secuencias.

    Fine-tuning para tareas como deteccion de bugs, clasificacion de lenguaje,
    analisis de complejidad, etc.

    Args:
        config: ZCodeBERTConfig con hiperparametros.
        num_labels: Numero de clases de clasificacion.

    Example:
        >>> config = ZCodeBERTConfig.small()
        >>> model = ZCodeBERTForSequenceClassification(config, num_labels=5)
        >>> input_ids = torch.randint(0, 50000, (2, 64))
        >>> labels = torch.tensor([1, 3])
        >>> loss, logits = model(input_ids, labels=labels)
    """

    def __init__(self, config: ZCodeBERTConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.bert = ZCodeBERT(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = ZFactorizedLinear(
            config.hidden_size,
            num_labels,
            rank=min(config.rank, num_labels),
            init_method=config.factor_init_method,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len).
            attention_mask: (batch, seq_len).
            token_type_ids: (batch, seq_len).
            labels: (batch,) class labels.

        Returns:
            Tuple (loss, logits).
        """
        _, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return loss, logits

    def grow_rank(self, new_rank: int) -> None:
        self.bert.grow_rank(new_rank)
        self.classifier.grow_rank(min(new_rank, self.num_labels))

    def get_model_stats(self) -> Dict[str, Any]:
        stats = self.bert.get_model_stats()
        stats["total_params"] = sum(p.numel() for p in self.parameters())
        stats["task"] = "SequenceClassification"
        stats["num_labels"] = self.num_labels
        return stats

    def compress_with_mneme(self, compression_level: str = "balanced"):
        return self.bert.compress_with_mneme(compression_level)

    def save_mneme(self, path: str, key: Optional[str] = None):
        return self.bert.save_mneme(path, key)
