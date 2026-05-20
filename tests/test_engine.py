"""
Tests para mztrain.engine - ZTrainEngine.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mztrain.config import ZTrainConfig, RankSchedule, GradientCompression
from mztrain.engine import ZTrainEngine
from mztrain.layers import ZFactorizedLinear


@pytest.fixture
def train_loader():
    """DataLoader de entrenamiento sintetico."""
    x = torch.randn(64, 128)
    y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def val_loader():
    """DataLoader de validacion sintetico."""
    x = torch.randn(32, 128)
    y = torch.randint(0, 10, (32,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16)


def loss_fn(model, batch):
    """Funcion de perdida para tests."""
    x, y = batch
    return F.cross_entropy(model(x), y)


class TestZTrainEngine:
    """Tests para ZTrainEngine."""

    def test_creation(self, simple_model):
        engine = ZTrainEngine(simple_model)
        assert engine.model is not None
        assert engine.optimizer is not None
        assert engine.grad_compressor is not None

    def test_creation_with_config(self, simple_model):
        config = ZTrainConfig(initial_rank=16, max_rank=64)
        engine = ZTrainEngine(simple_model, config)
        assert engine.config.initial_rank == 16
        assert engine.config.max_rank == 64

    def test_factorization(self, simple_model):
        config = ZTrainConfig(initial_rank=16, min_params_to_factorize=100)
        engine = ZTrainEngine(simple_model, config)
        # Verificar que hay capas factorizadas
        has_factorized = any(
            isinstance(m, ZFactorizedLinear)
            for m in engine.model.modules()
        )
        assert has_factorized

    def test_factorization_skip_small(self):
        model = nn.Sequential(
            nn.Linear(8, 4),  # 32 params - muy pequeno
        )
        config = ZTrainConfig(min_params_to_factorize=4096)
        engine = ZTrainEngine(model, config)
        # No deberia haber capas factorizadas
        has_factorized = any(
            isinstance(m, ZFactorizedLinear)
            for m in engine.model.modules()
        )
        assert not has_factorized

    def test_train_epoch(self, simple_model, train_loader):
        config = ZTrainConfig(initial_rank=8, use_amp=False)
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        loss = engine.train_epoch(train_loader, loss_fn, epoch=0)
        assert isinstance(loss, float)
        assert loss > 0

    def test_validate(self, simple_model, val_loader):
        config = ZTrainConfig(initial_rank=8, use_amp=False)
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        loss = engine.validate(val_loader, loss_fn)
        assert isinstance(loss, float)
        assert loss > 0

    def test_full_training(self, simple_model, train_loader, val_loader):
        config = ZTrainConfig(
            initial_rank=8, max_rank=16, use_amp=False,
            log_interval=100,
        )
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        summary = engine.train(
            train_loader, val_loader, loss_fn,
            epochs=3, early_stopping_patience=5,
        )
        assert "train_losses" in summary
        assert len(summary["train_losses"]) == 3
        assert "val_losses" in summary
        assert "ranks" in summary

    def test_training_with_gradient_compression(self, simple_model, train_loader):
        config = ZTrainConfig(
            initial_rank=8, use_amp=False,
            gradient_compression=GradientCompression.TOP_K,
            gradient_top_k_ratio=0.5,
        )
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        summary = engine.train(
            train_loader, None, loss_fn, epochs=2,
        )
        assert len(summary["train_losses"]) == 2

    def test_early_stopping(self, simple_model, train_loader, val_loader):
        config = ZTrainConfig(initial_rank=8, use_amp=False)
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        # Con paciencia muy baja y epochs altos
        summary = engine.train(
            train_loader, val_loader, loss_fn,
            epochs=100, early_stopping_patience=2,
        )
        # Deberia parar antes de 100 epochs
        assert len(summary["train_losses"]) < 100

    def test_export_full_model(self, simple_model):
        config = ZTrainConfig(initial_rank=8, min_params_to_factorize=100)
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        full_model = engine.export_full_model()
        # No deberia tener capas factorizadas
        has_factorized = any(
            isinstance(m, ZFactorizedLinear)
            for m in full_model.modules()
        )
        assert not has_factorized
        # Deberia funcionar
        x = torch.randn(4, 128)
        y = full_model(x)
        assert y.shape == (4, 10)

    def test_save_load_checkpoint(self, simple_model, train_loader):
        config = ZTrainConfig(initial_rank=8, use_amp=False)
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        engine.train(train_loader, None, loss_fn, epochs=2)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            engine.save_checkpoint(path)
            assert os.path.exists(path)

            # Crear nuevo engine y cargar
            engine2 = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
            engine2.load_checkpoint(path)
        finally:
            os.unlink(path)

    def test_training_summary(self, simple_model, train_loader):
        config = ZTrainConfig(initial_rank=8, use_amp=False)
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        engine.train(train_loader, None, loss_fn, epochs=2)

        summary = engine.get_training_summary()
        assert "config" in summary
        assert "optimizer_stats" in summary
        assert "gradient_stats" in summary
        assert "layer_stats" in summary

    def test_callbacks(self, simple_model, train_loader):
        callback_calls = []

        def my_callback(engine, epoch, metrics):
            callback_calls.append(metrics)

        config = ZTrainConfig(initial_rank=8, use_amp=False)
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        engine.train(
            train_loader, None, loss_fn, epochs=3,
            callbacks=[my_callback],
        )
        assert len(callback_calls) == 3

    def test_rank_growth(self, simple_model, train_loader):
        config = ZTrainConfig(
            initial_rank=8, max_rank=32, use_amp=False,
            rank_schedule=RankSchedule.EXPONENTIAL,
            rank_growth_interval=1, rank_growth_factor=2.0,
        )
        engine = ZTrainEngine(simple_model, config, device=torch.device("cpu"))
        summary = engine.train(
            train_loader, None, loss_fn, epochs=3,
        )
        # Rango deberia haber crecido
        assert summary["ranks"][-1] > 8
