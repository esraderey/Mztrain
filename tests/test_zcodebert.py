"""
Tests para ZCodeBERT - Modelo, datos e integracion.
"""

import pytest
import torch
import torch.nn as nn

from mztrain.data.tokenizer import (
    CodeTokenizer, PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID,
)
from mztrain.data.dataset import SyntheticCodeGenerator, CodeDataset, MLMDataset, CausalCodeDataset
from mztrain.models.zcodebert import (
    ZCodeBERTConfig,
    ZCodeBERTEmbeddings,
    ZCodeBERTEncoder,
    ZCodeBERTPooler,
    ZCodeBERT,
    ZCodeBERTForMLM,
    ZCodeBERTForCausalLM,
    ZCodeBERTForSequenceClassification,
)
from mztrain.layers import ZFactorizedLinear


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_config():
    """Config reducida para tests rapidos."""
    return ZCodeBERTConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=32,
        rank=8,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        use_activation_checkpointing=False,
    )


@pytest.fixture
def tokenizer():
    return CodeTokenizer(vocab_size=1000, max_length=32)


@pytest.fixture
def generator():
    return SyntheticCodeGenerator(seed=42)


# ============================================================================
# Tests: CodeTokenizer
# ============================================================================

class TestCodeTokenizer:
    """Tests para CodeTokenizer."""

    def test_init(self, tokenizer):
        assert tokenizer.vocab_size == 1000
        assert tokenizer.max_length == 32

    def test_special_tokens(self, tokenizer):
        assert tokenizer.pad_token_id == PAD_ID
        assert tokenizer.cls_token_id == CLS_ID
        assert tokenizer.sep_token_id == SEP_ID
        assert tokenizer.mask_token_id == MASK_ID
        assert tokenizer.unk_token_id == UNK_ID

    def test_encode_basic(self, tokenizer):
        ids = tokenizer.encode("x = 1")
        assert isinstance(ids, list)
        assert len(ids) == 32  # padded
        assert ids[0] == CLS_ID
        # SEP should be somewhere before padding
        assert SEP_ID in ids

    def test_encode_no_special(self, tokenizer):
        ids = tokenizer.encode("x = 1", add_special_tokens=False)
        assert ids[0] != CLS_ID

    def test_encode_no_padding(self, tokenizer):
        ids = tokenizer.encode("x", padding=False)
        assert len(ids) < 32

    def test_decode(self, tokenizer):
        ids = tokenizer.encode("def foo", padding=False)
        text = tokenizer.decode(ids)
        assert "def" in text
        assert "foo" in text

    def test_encode_long_input(self, tokenizer):
        long_code = "x = 1\n" * 100
        ids = tokenizer.encode(long_code)
        assert len(ids) == 32  # truncated + padded

    def test_camel_case_split(self, tokenizer):
        ids = tokenizer.encode("myVariableName", padding=False)
        text = tokenizer.decode(ids)
        # Should contain subparts
        assert len(text) > 0

    def test_snake_case_split(self, tokenizer):
        ids = tokenizer.encode("my_variable_name", padding=False)
        text = tokenizer.decode(ids)
        assert len(text) > 0

    def test_operators(self, tokenizer):
        ids = tokenizer.encode("a == b && c != d", padding=False)
        assert len(ids) > 0

    def test_vocab_mapping_consistent(self, tokenizer):
        for token, idx in tokenizer.token_to_id.items():
            assert tokenizer.id_to_token[idx] == token


# ============================================================================
# Tests: SyntheticCodeGenerator
# ============================================================================

class TestSyntheticCodeGenerator:
    """Tests para SyntheticCodeGenerator."""

    def test_generate(self, generator):
        snippets = generator.generate(10)
        assert len(snippets) == 10
        assert all(isinstance(s, str) for s in snippets)
        assert all(len(s) > 0 for s in snippets)

    def test_seed_reproducibility(self):
        gen1 = SyntheticCodeGenerator(seed=42)
        gen2 = SyntheticCodeGenerator(seed=42)
        s1 = gen1.generate(5)
        s2 = gen2.generate(5)
        assert s1 == s2

    def test_all_languages(self, generator):
        snippets = generator.generate(100)
        # Should generate non-empty snippets for any language
        assert all(len(s) > 10 for s in snippets)

    def test_single_language(self):
        gen = SyntheticCodeGenerator(languages=["python"], seed=1)
        snippets = gen.generate(5)
        assert len(snippets) == 5

    def test_generate_large_batch(self, generator):
        snippets = generator.generate(500)
        assert len(snippets) == 500


# ============================================================================
# Tests: CodeDataset
# ============================================================================

class TestCodeDataset:
    """Tests para CodeDataset."""

    def test_init(self, generator, tokenizer):
        snippets = generator.generate(10)
        dataset = CodeDataset(snippets, tokenizer, max_length=32)
        assert len(dataset) == 10

    def test_getitem(self, generator, tokenizer):
        snippets = generator.generate(5)
        dataset = CodeDataset(snippets, tokenizer, max_length=32)
        input_ids, attention_mask = dataset[0]
        assert input_ids.shape == (32,)
        assert attention_mask.shape == (32,)
        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.long

    def test_attention_mask_values(self, generator, tokenizer):
        snippets = generator.generate(5)
        dataset = CodeDataset(snippets, tokenizer, max_length=32)
        _, mask = dataset[0]
        # All values should be 0 or 1
        assert ((mask == 0) | (mask == 1)).all()


# ============================================================================
# Tests: MLMDataset
# ============================================================================

class TestMLMDataset:
    """Tests para MLMDataset."""

    def test_init(self, generator, tokenizer):
        snippets = generator.generate(10)
        code_ds = CodeDataset(snippets, tokenizer, max_length=32)
        mlm_ds = MLMDataset(code_ds, mask_prob=0.15, seed=42)
        assert len(mlm_ds) == 10

    def test_getitem(self, generator, tokenizer):
        snippets = generator.generate(5)
        code_ds = CodeDataset(snippets, tokenizer, max_length=32)
        mlm_ds = MLMDataset(code_ds, mask_prob=0.15, seed=42)

        masked_ids, attention_mask, labels = mlm_ds[0]
        assert masked_ids.shape == (32,)
        assert attention_mask.shape == (32,)
        assert labels.shape == (32,)

    def test_masking_applied(self, generator, tokenizer):
        snippets = generator.generate(20)
        code_ds = CodeDataset(snippets, tokenizer, max_length=32)
        mlm_ds = MLMDataset(code_ds, mask_prob=0.5, seed=42)  # High prob for testing

        # Check that some tokens are masked across multiple samples
        total_masked = 0
        for i in range(len(mlm_ds)):
            _, _, labels = mlm_ds[i]
            total_masked += (labels != -100).sum().item()

        assert total_masked > 0, "No tokens were masked"

    def test_labels_negative_100_for_unmasked(self, generator, tokenizer):
        snippets = generator.generate(5)
        code_ds = CodeDataset(snippets, tokenizer, max_length=32)
        mlm_ds = MLMDataset(code_ds, mask_prob=0.15, seed=42)

        _, _, labels = mlm_ds[0]
        # Most labels should be -100 (unmasked)
        assert (labels == -100).sum() > 0

    def test_mask_id_in_masked_positions(self, generator, tokenizer):
        snippets = generator.generate(20)
        code_ds = CodeDataset(snippets, tokenizer, max_length=32)
        mlm_ds = MLMDataset(code_ds, mask_prob=0.5, seed=42)

        # Check across multiple samples that MASK_ID appears
        mask_count = 0
        for i in range(len(mlm_ds)):
            masked_ids, _, _ = mlm_ds[i]
            mask_count += (masked_ids == MASK_ID).sum().item()
        assert mask_count > 0, "No MASK tokens found"

    def test_ensures_at_least_one_mask_for_eligible_tokens(self, tokenizer):
        code_ds = CodeDataset(["def foo(): return 1"], tokenizer, max_length=32)
        mlm_ds = MLMDataset(code_ds, mask_prob=0.0, seed=42)

        _, _, labels = mlm_ds[0]

        assert (labels != -100).sum().item() == 1

    def test_can_disable_forced_mask(self, tokenizer):
        code_ds = CodeDataset(["def foo(): return 1"], tokenizer, max_length=32)
        mlm_ds = MLMDataset(
            code_ds,
            mask_prob=0.0,
            seed=42,
            ensure_at_least_one_mask=False,
        )

        _, _, labels = mlm_ds[0]

        assert (labels != -100).sum().item() == 0

    def test_invalid_mask_prob_raises(self, generator, tokenizer):
        snippets = generator.generate(1)
        code_ds = CodeDataset(snippets, tokenizer, max_length=32)

        with pytest.raises(ValueError, match="mask_prob"):
            MLMDataset(code_ds, mask_prob=1.5)


class TestCausalCodeDataset:
    """Tests para dataset causal/prefix-LM."""

    def test_getitem(self, tokenizer):
        code_ds = CodeDataset(["def foo(): return 1"], tokenizer, max_length=32)
        causal_ds = CausalCodeDataset(code_ds)

        input_ids, attention_mask, labels = causal_ds[0]

        assert input_ids.shape == (32,)
        assert attention_mask.shape == (32,)
        assert labels.shape == (32,)
        assert labels[0].item() == -100  # CLS ignorado
        assert (labels[input_ids == PAD_ID] == -100).all()

    def test_can_keep_cls_label_when_requested(self, tokenizer):
        code_ds = CodeDataset(["def foo(): return 1"], tokenizer, max_length=32)
        causal_ds = CausalCodeDataset(code_ds, ignore_special_tokens=False)

        input_ids, _, labels = causal_ds[0]

        assert labels[0].item() == input_ids[0].item()


# ============================================================================
# Tests: ZCodeBERTConfig
# ============================================================================

class TestZCodeBERTConfig:
    """Tests para ZCodeBERTConfig."""

    def test_default_config(self):
        config = ZCodeBERTConfig()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.vocab_size == 50000

    def test_small_config(self):
        config = ZCodeBERTConfig.small()
        assert config.hidden_size == 128
        assert config.num_hidden_layers == 4

    def test_base_config(self):
        config = ZCodeBERTConfig.base()
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12

    def test_large_config(self):
        config = ZCodeBERTConfig.large()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24

    def test_coder_1b_config_uses_fast_factor_init(self):
        config = ZCodeBERTConfig.coder_1b(rank=64)
        assert config.hidden_size == 1792
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 28
        assert config.intermediate_size == 7168
        assert config.rank == 64
        assert config.factor_init_method == "kaiming"

    def test_to_dict(self):
        config = ZCodeBERTConfig.small()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "hidden_size" in d
        assert d["hidden_size"] == 128

    def test_from_dict(self):
        config = ZCodeBERTConfig.small()
        d = config.to_dict()
        config2 = ZCodeBERTConfig.from_dict(d)
        assert config2.hidden_size == config.hidden_size
        assert config2.num_hidden_layers == config.num_hidden_layers

    def test_from_dict_ignores_unknown_keys(self):
        config = ZCodeBERTConfig.from_dict({"hidden_size": 256, "unknown": "ignored"})
        assert config.hidden_size == 256
        assert not hasattr(config, "unknown")

    def test_mlp_ratio(self):
        config = ZCodeBERTConfig(hidden_size=256, intermediate_size=1024)
        assert config.mlp_ratio == 4.0


# ============================================================================
# Tests: ZCodeBERTEmbeddings
# ============================================================================

class TestZCodeBERTEmbeddings:
    """Tests para ZCodeBERTEmbeddings."""

    def test_forward(self, small_config):
        emb = ZCodeBERTEmbeddings(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        out = emb(input_ids)
        assert out.shape == (2, 16, small_config.hidden_size)

    def test_with_token_type(self, small_config):
        emb = ZCodeBERTEmbeddings(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        token_type_ids = torch.zeros_like(input_ids)
        out = emb(input_ids, token_type_ids)
        assert out.shape == (2, 16, small_config.hidden_size)


# ============================================================================
# Tests: ZCodeBERTEncoder
# ============================================================================

class TestZCodeBERTEncoder:
    """Tests para ZCodeBERTEncoder."""

    def test_forward(self, small_config):
        encoder = ZCodeBERTEncoder(small_config)
        x = torch.randn(2, 16, small_config.hidden_size)
        out = encoder(x)
        assert out.shape == x.shape

    def test_with_mask(self, small_config):
        encoder = ZCodeBERTEncoder(small_config)
        x = torch.randn(2, 16, small_config.hidden_size)
        mask = torch.ones(2, 1, 1, 16)
        out = encoder(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_num_layers(self, small_config):
        encoder = ZCodeBERTEncoder(small_config)
        assert len(encoder.layers) == small_config.num_hidden_layers


# ============================================================================
# Tests: ZCodeBERT (modelo base)
# ============================================================================

class TestZCodeBERT:
    """Tests para ZCodeBERT."""

    def test_forward(self, small_config):
        model = ZCodeBERT(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        seq_out, pooled_out = model(input_ids)
        assert seq_out.shape == (2, 16, small_config.hidden_size)
        assert pooled_out.shape == (2, small_config.hidden_size)

    def test_with_attention_mask(self, small_config):
        model = ZCodeBERT(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        mask = torch.ones(2, 16)
        mask[:, 12:] = 0  # Mask last 4 tokens
        seq_out, pooled_out = model(input_ids, attention_mask=mask)
        assert seq_out.shape == (2, 16, small_config.hidden_size)

    def test_grow_rank(self, small_config):
        model = ZCodeBERT(small_config)
        old_params = sum(p.numel() for p in model.parameters())
        model.grow_rank(16)
        new_params = sum(p.numel() for p in model.parameters())
        assert new_params > old_params
        assert model.config.rank == 16

    def test_get_model_stats(self, small_config):
        model = ZCodeBERT(small_config)
        stats = model.get_model_stats()
        assert "total_params" in stats
        assert "equivalent_params" in stats
        assert "factorized_layers" in stats
        assert "compression_ratio" in stats
        assert stats["total_params"] > 0
        assert stats["equivalent_params"] > stats["total_params"]
        assert stats["factorized_layers"] > 0

    def test_export_full_model(self, small_config):
        model = ZCodeBERT(small_config)
        full_model = model._export_full_model()
        # No ZFactorizedLinear should remain
        for module in full_model.modules():
            assert not isinstance(module, ZFactorizedLinear), \
                f"Found ZFactorizedLinear in exported model: {module}"

    def test_gradient_flow(self, small_config):
        model = ZCodeBERT(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 8))
        seq_out, pooled_out = model(input_ids)
        loss = pooled_out.sum()
        loss.backward()
        # Check some gradients exist
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found"


# ============================================================================
# Tests: ZCodeBERTForMLM
# ============================================================================

class TestZCodeBERTForMLM:
    """Tests para ZCodeBERTForMLM."""

    def test_forward_no_labels(self, small_config):
        model = ZCodeBERTForMLM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        loss, logits = model(input_ids)
        assert loss is None
        assert logits.shape == (2, 16, small_config.vocab_size)

    def test_forward_with_labels(self, small_config):
        model = ZCodeBERTForMLM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        labels = torch.full((2, 16), -100, dtype=torch.long)
        # Mask some positions
        labels[:, 3:6] = torch.randint(0, small_config.vocab_size, (2, 3))
        loss, logits = model(input_ids, labels=labels)
        assert loss is not None
        assert loss.dim() == 0  # scalar
        assert logits.shape == (2, 16, small_config.vocab_size)

    def test_backward(self, small_config):
        model = ZCodeBERTForMLM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        labels = torch.randint(0, small_config.vocab_size, (2, 16))
        loss, _ = model(input_ids, labels=labels)
        loss.backward()
        # Check optimizer can step
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.step()

    def test_grow_rank(self, small_config):
        model = ZCodeBERTForMLM(small_config)
        model.grow_rank(16)
        # Should still work
        input_ids = torch.randint(0, small_config.vocab_size, (2, 8))
        loss, logits = model(input_ids, labels=input_ids)
        assert loss is not None

    def test_weight_tying(self, small_config):
        model = ZCodeBERTForMLM(small_config)
        # MLM decoder weight should be same object as word embeddings
        assert model.mlm_decoder.weight is model.bert.embeddings.word_embeddings.weight

    def test_get_model_stats(self, small_config):
        model = ZCodeBERTForMLM(small_config)
        stats = model.get_model_stats()
        assert stats["task"] == "MLM"
        assert stats["total_params"] > 0


class TestZCodeBERTForCausalLM:
    """Tests para variante causal de completado."""

    def test_forward_no_labels(self, small_config):
        model = ZCodeBERTForCausalLM(small_config)
        input_ids = torch.randint(5, small_config.vocab_size, (2, 16))

        loss, logits = model(input_ids)

        assert loss is None
        assert logits.shape == (2, 16, small_config.vocab_size)

    def test_forward_with_kaiming_factor_init(self, small_config):
        small_config.factor_init_method = "kaiming"
        model = ZCodeBERTForCausalLM(small_config)
        input_ids = torch.randint(5, small_config.vocab_size, (2, 16))

        loss, logits = model(input_ids, labels=input_ids)

        assert loss is not None
        assert logits.shape == (2, 16, small_config.vocab_size)

    def test_forward_with_labels(self, small_config):
        model = ZCodeBERTForCausalLM(small_config)
        input_ids = torch.randint(5, small_config.vocab_size, (2, 16))
        labels = input_ids.clone()

        loss, logits = model(input_ids, labels=labels)

        assert loss is not None
        assert loss.dim() == 0
        assert logits.shape == (2, 16, small_config.vocab_size)

    def test_backward(self, small_config):
        model = ZCodeBERTForCausalLM(small_config)
        input_ids = torch.randint(5, small_config.vocab_size, (2, 16))
        labels = input_ids.clone()

        loss, _ = model(input_ids, labels=labels)
        loss.backward()

        assert any(p.grad is not None for p in model.parameters())

    def test_generate_extends_sequence(self, small_config):
        model = ZCodeBERTForCausalLM(small_config)
        input_ids = torch.randint(5, small_config.vocab_size, (1, 6))

        generated = model.generate(input_ids, max_new_tokens=3, temperature=0.0)

        assert generated.shape[0] == 1
        assert generated.shape[1] >= input_ids.shape[1]
        assert generated.shape[1] <= input_ids.shape[1] + 3

    def test_can_load_mlm_checkpoint_strictly(self, small_config):
        mlm = ZCodeBERTForMLM(small_config)
        causal = ZCodeBERTForCausalLM(small_config)

        causal.load_state_dict(mlm.state_dict(), strict=True)

    def test_causal_logits_do_not_depend_on_future_tokens(self, small_config):
        model = ZCodeBERTForCausalLM(small_config)
        model.eval()
        prefix = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
        suffix_a = torch.tensor([[21, 22, 23, 24]], dtype=torch.long)
        suffix_b = torch.tensor([[31, 32, 33, 34]], dtype=torch.long)
        x_a = torch.cat([prefix, suffix_a], dim=1)
        x_b = torch.cat([prefix, suffix_b], dim=1)

        with torch.no_grad():
            _, logits_a = model(x_a)
            _, logits_b = model(x_b)

        assert torch.allclose(logits_a[:, :prefix.shape[1]], logits_b[:, :prefix.shape[1]], atol=1e-5)

    def test_get_model_stats(self, small_config):
        model = ZCodeBERTForCausalLM(small_config)
        stats = model.get_model_stats()

        assert stats["task"] == "CausalLM"
        assert stats["total_params"] > 0


# ============================================================================
# Tests: ZCodeBERTForSequenceClassification
# ============================================================================

class TestZCodeBERTForSequenceClassification:
    """Tests para ZCodeBERTForSequenceClassification."""

    def test_forward_no_labels(self, small_config):
        model = ZCodeBERTForSequenceClassification(small_config, num_labels=5)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        loss, logits = model(input_ids)
        assert loss is None
        assert logits.shape == (2, 5)

    def test_forward_with_labels(self, small_config):
        model = ZCodeBERTForSequenceClassification(small_config, num_labels=5)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        labels = torch.tensor([1, 3])
        loss, logits = model(input_ids, labels=labels)
        assert loss is not None
        assert loss.dim() == 0
        assert logits.shape == (2, 5)

    def test_backward(self, small_config):
        model = ZCodeBERTForSequenceClassification(small_config, num_labels=3)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        labels = torch.tensor([0, 2])
        loss, _ = model(input_ids, labels=labels)
        loss.backward()

    def test_binary_classification(self, small_config):
        model = ZCodeBERTForSequenceClassification(small_config, num_labels=2)
        input_ids = torch.randint(0, small_config.vocab_size, (4, 16))
        labels = torch.tensor([0, 1, 0, 1])
        loss, logits = model(input_ids, labels=labels)
        assert logits.shape == (4, 2)

    def test_get_model_stats(self, small_config):
        model = ZCodeBERTForSequenceClassification(small_config, num_labels=5)
        stats = model.get_model_stats()
        assert stats["task"] == "SequenceClassification"
        assert stats["num_labels"] == 5


# ============================================================================
# Tests: Integracion end-to-end
# ============================================================================

class TestIntegration:
    """Tests de integracion end-to-end."""

    def test_full_pipeline(self, small_config):
        """Test completo: datos -> modelo -> train step -> stats."""
        # Generar datos
        gen = SyntheticCodeGenerator(seed=42)
        tok = CodeTokenizer(vocab_size=small_config.vocab_size, max_length=16)
        snippets = gen.generate(20)
        code_ds = CodeDataset(snippets, tok, max_length=16)
        mlm_ds = MLMDataset(code_ds, mask_prob=0.15, seed=42)

        # Crear modelo
        model = ZCodeBERTForMLM(small_config)

        # Train step
        masked_ids, attn_mask, labels = mlm_ds[0]
        masked_ids = masked_ids.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)
        labels = labels.unsqueeze(0)

        loss, logits = model(masked_ids, attention_mask=attn_mask, labels=labels)
        assert loss is not None
        loss.backward()

        # Stats
        stats = model.get_model_stats()
        assert stats["total_params"] > 0

    def test_dataloader_pipeline(self, small_config):
        """Test con DataLoader."""
        from torch.utils.data import DataLoader

        gen = SyntheticCodeGenerator(seed=42)
        tok = CodeTokenizer(vocab_size=small_config.vocab_size, max_length=16)
        snippets = gen.generate(16)
        code_ds = CodeDataset(snippets, tok, max_length=16)
        mlm_ds = MLMDataset(code_ds, mask_prob=0.15, seed=42)
        loader = DataLoader(mlm_ds, batch_size=4)

        model = ZCodeBERTForMLM(small_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for batch in loader:
            masked_ids, attn_mask, labels = batch
            optimizer.zero_grad()
            loss, _ = model(masked_ids, attention_mask=attn_mask, labels=labels)
            loss.backward()
            optimizer.step()
            break  # Just one batch

    def test_rank_growth_training(self, small_config):
        """Test entrenamiento con crecimiento de rango."""
        model = ZCodeBERTForMLM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 8))
        labels = torch.randint(0, small_config.vocab_size, (2, 8))

        # Train con rango inicial
        loss1, _ = model(input_ids, labels=labels)
        loss1.backward()

        # Crecer rango
        model.grow_rank(16)

        # Train con rango nuevo
        model.zero_grad()
        loss2, _ = model(input_ids, labels=labels)
        loss2.backward()
        assert loss2.item() > 0

    def test_classification_pipeline(self, small_config):
        """Test pipeline de clasificacion."""
        model = ZCodeBERTForSequenceClassification(small_config, num_labels=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(3):
            input_ids = torch.randint(0, small_config.vocab_size, (4, 8))
            labels = torch.randint(0, 3, (4,))
            optimizer.zero_grad()
            loss, logits = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()

        assert logits.shape == (4, 3)
