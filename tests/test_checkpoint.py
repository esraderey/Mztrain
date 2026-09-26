"""
Tests para mztrain.checkpoint - Activation checkpointing.
"""

import pytest
import torch

from mztrain.checkpoint import ZActivationCheckpoint, z_checkpoint


class TestZActivationCheckpoint:
    """Tests para ZActivationCheckpoint."""

    def setup_method(self):
        """Reset antes de cada test."""
        ZActivationCheckpoint.reset()

    def test_compress_decompress(self):
        tensor = torch.randn(4, 64)
        key, shape, dtype = ZActivationCheckpoint.compress_activation(tensor)
        recovered = ZActivationCheckpoint.decompress_activation(key, shape, dtype)
        # Con fallback INT8, no es exacto pero cercano
        assert recovered.shape == tensor.shape
        assert torch.allclose(tensor, recovered, atol=0.1)

    def test_stats(self):
        tensor = torch.randn(4, 64)
        ZActivationCheckpoint.compress_activation(tensor)
        stats = ZActivationCheckpoint.get_stats()
        assert stats["total_compressed"] == 1

    def test_multiple_compress(self):
        for _ in range(5):
            tensor = torch.randn(4, 64)
            ZActivationCheckpoint.compress_activation(tensor)
        stats = ZActivationCheckpoint.get_stats()
        assert stats["total_compressed"] == 5

    def test_reset(self):
        tensor = torch.randn(4, 64)
        ZActivationCheckpoint.compress_activation(tensor)
        ZActivationCheckpoint.reset()
        stats = ZActivationCheckpoint.get_stats()
        assert stats["total_compressed"] == 0

    def test_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="not found"):
            ZActivationCheckpoint.decompress_activation(
                "nonexistent_key", torch.Size([4, 64]), torch.float32
            )

    def test_dtype_preservation(self):
        for dtype in [torch.float32, torch.float64]:
            ZActivationCheckpoint.reset()
            tensor = torch.randn(4, 64, dtype=dtype)
            key, shape, dt = ZActivationCheckpoint.compress_activation(tensor)
            assert dt == dtype


class TestZCheckpoint:
    """Tests para z_checkpoint function."""

    def setup_method(self):
        ZActivationCheckpoint.reset()

    def test_basic_checkpoint(self):
        import torch.nn as nn
        layer = nn.Linear(64, 32)
        x = torch.randn(4, 64, requires_grad=True)
        out = z_checkpoint(layer, x)
        assert out.shape == (4, 32)

    def test_gradient_through_checkpoint(self):
        import torch.nn as nn
        layer = nn.Linear(64, 32)
        x = torch.randn(4, 64, requires_grad=True)
        out = z_checkpoint(layer, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestZSpaceBackendAdapter:
    """Adaptador de backend ZSpace (MNEME/MNEMOSYS): opt-in, eviccion obligatoria,
    kwargs de cuantizacion, device sin kwarg y sonda de fidelidad."""

    def setup_method(self):
        ZActivationCheckpoint.reset()
        ZActivationCheckpoint.prefer_zspace = False

    def teardown_method(self):
        ZActivationCheckpoint.reset()
        ZActivationCheckpoint.prefer_zspace = False

    @staticmethod
    def _install(monkeypatch, zspace_cls):
        import mztrain.checkpoint as ck

        class FakeConfig:
            def __init__(self):
                self.compression_level = None
                self.storage_path = "./mneme_storage"

        class FakeLevel:
            ULTRA_FAST = "ultra_fast"

        monkeypatch.setattr(ck, "HAS_MNEME", True)
        monkeypatch.setattr(ck, "ZSpace", zspace_cls)
        monkeypatch.setattr(ck, "MnemeConfig", FakeConfig)
        monkeypatch.setattr(ck, "CompressionLevel", FakeLevel)

    class _Good:
        """Backend estilo mnemosys 1.0: register(**kwargs), load(name) sin device, delete."""
        instances = []

        def __init__(self, config):
            self.config = config
            self.store = {}
            self.kwargs_seen = []
            type(self).instances.append(self)

        def register(self, name, tensor, **kwargs):
            self.kwargs_seen.append(dict(kwargs))
            self.store[name] = tensor.detach().clone().cpu()

        def load(self, name):
            return self.store[name]

        def delete(self, name):
            return self.store.pop(name, None) is not None

    def _roundtrip(self, n=5, device="cpu"):
        t = torch.randn(4, 64, device=device)
        for _ in range(n):
            key, shape, dtype = ZActivationCheckpoint.compress_activation(t)
            back = ZActivationCheckpoint.decompress_activation(key, shape, dtype)
        return t, back

    def test_opt_in_por_defecto_no_construye_zspace(self, monkeypatch):
        self._Good.instances.clear()
        self._install(monkeypatch, self._Good)
        self._roundtrip()
        assert ZActivationCheckpoint._zspace is None
        assert self._Good.instances == []

    def test_backend_sin_eviccion_se_deshabilita_con_aviso(self, monkeypatch):
        class NoEvict(self._Good):
            delete = None

        self._install(monkeypatch, NoEvict)
        ZActivationCheckpoint.prefer_zspace = True
        with pytest.warns(RuntimeWarning, match=r"remove\(\)/delete\(\)"):
            t, back = self._roundtrip(n=2)
        assert ZActivationCheckpoint._zspace is None
        assert "eviccion" in ZActivationCheckpoint._zspace_disabled_reason
        assert torch.allclose(t, back, atol=0.1)  # fallback INT8 sigue funcionando

    def test_registra_con_int8_y_evicta_por_clave(self, monkeypatch):
        self._Good.instances.clear()
        self._install(monkeypatch, self._Good)
        ZActivationCheckpoint.prefer_zspace = True
        t, back = self._roundtrip(n=6)
        z = self._Good.instances[0]
        assert ZActivationCheckpoint._zspace is z
        assert all(kw == {"quantization_type": "int8"} for kw in z.kwargs_seen)
        assert z.store == {}, "la eviccion por clave debe dejar el registro vacio"
        assert torch.equal(back, t)
        assert z.config.storage_path != "./mneme_storage"  # directorio temporal propio

    def test_backend_que_rechaza_kwargs_usa_register_plano(self, monkeypatch):
        class Plain(self._Good):
            def register(self, name, tensor):
                self.kwargs_seen.append({})
                self.store[name] = tensor.detach().clone().cpu()

        Plain.instances = []
        self._install(monkeypatch, Plain)
        ZActivationCheckpoint.prefer_zspace = True
        t, back = self._roundtrip(n=3)
        assert Plain.instances[0].store == {} and torch.equal(back, t)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
    def test_load_sin_kwarg_device_devuelve_en_el_device_original(self, monkeypatch):
        self._Good.instances.clear()
        self._install(monkeypatch, self._Good)
        ZActivationCheckpoint.prefer_zspace = True
        t, back = self._roundtrip(n=2, device="cuda")
        assert back.device == t.device and torch.equal(back.cpu(), t.cpu())

    def test_backend_de_baja_fidelidad_se_deshabilita(self, monkeypatch):
        class Noisy(self._Good):
            def load(self, name):
                return torch.randn_like(self.store[name])

        Noisy.instances = []
        self._install(monkeypatch, Noisy)
        ZActivationCheckpoint.prefer_zspace = True
        with pytest.warns(RuntimeWarning, match="fidelidad insuficiente"):
            t, back = self._roundtrip(n=2)
        assert ZActivationCheckpoint._zspace is None
        assert torch.allclose(t, back, atol=0.1)

    def test_reset_conserva_prefer_zspace(self, monkeypatch):
        self._install(monkeypatch, self._Good)
        ZActivationCheckpoint.prefer_zspace = True
        self._roundtrip(n=1)
        ZActivationCheckpoint.reset()
        assert ZActivationCheckpoint.prefer_zspace is True
        assert ZActivationCheckpoint._zspace is None and ZActivationCheckpoint._zspace_probed is False

    def test_sonda_usa_la_forma_real_y_detecta_corrupcion_por_tamano(self, monkeypatch):
        class BigCorrupt(self._Good):
            """Corrompe solo tensores grandes (como el enrutado tensorial de mnemosys)."""
            def load(self, name):
                t = self.store[name]
                return torch.randn_like(t) if t.numel() > 4096 else t

        BigCorrupt.instances = []
        self._install(monkeypatch, BigCorrupt)
        ZActivationCheckpoint.prefer_zspace = True
        t = torch.randn(16, 64, 64)
        with pytest.warns(RuntimeWarning, match="fidelidad insuficiente"):
            key, shape, dtype = ZActivationCheckpoint.compress_activation(t)
        back = ZActivationCheckpoint.decompress_activation(key, shape, dtype)
        assert ZActivationCheckpoint._zspace is None
        assert torch.allclose(t, back, atol=0.1)

    def test_fallo_al_construir_zspace_deshabilita_con_aviso(self, monkeypatch):
        class Boom(self._Good):
            def __init__(self, config):
                raise RuntimeError("sin backend")

        self._install(monkeypatch, Boom)
        ZActivationCheckpoint.prefer_zspace = True
        with pytest.warns(RuntimeWarning, match="no se pudo construir ZSpace"):
            t, back = self._roundtrip(n=2)
        assert ZActivationCheckpoint._zspace is None and torch.allclose(t, back, atol=0.1)
        assert ZActivationCheckpoint._zspace_storage_dir is None

    def test_opt_in_sin_mneme_instalado_avisa(self, monkeypatch):
        import mztrain.checkpoint as ck

        monkeypatch.setattr(ck, "HAS_MNEME", False)
        ZActivationCheckpoint.prefer_zspace = True
        with pytest.warns(RuntimeWarning, match="MNEME no esta instalado"):
            t, back = self._roundtrip(n=1)
        assert torch.allclose(t, back, atol=0.1)

    def test_activacion_int8_se_recupera_aunque_se_habilite_el_store_despues(self, monkeypatch):
        self._Good.instances.clear()
        self._install(monkeypatch, self._Good)
        t = torch.randn(4, 64)
        key, shape, dtype = ZActivationCheckpoint.compress_activation(t)  # INT8 (opt-in apagado)
        ZActivationCheckpoint.prefer_zspace = True  # el flag cambia entre forward y backward
        back = ZActivationCheckpoint.decompress_activation(key, shape, dtype)
        assert torch.allclose(t, back, atol=0.1)

    def test_register_reenvia_typeerror_ajeno_al_kwarg(self, monkeypatch):
        class BadTensor(self._Good):
            def register(self, name, tensor, **kwargs):
                raise TypeError("dtype no soportado por el backend")

        BadTensor.instances = []
        self._install(monkeypatch, BadTensor)
        ZActivationCheckpoint.prefer_zspace = True
        with pytest.warns(RuntimeWarning, match="round-trip de prueba fallo"):
            t, back = self._roundtrip(n=1)  # la sonda absorbe el fallo y deshabilita el store
        assert ZActivationCheckpoint._zspace is None and torch.allclose(t, back, atol=0.1)

    def test_reset_borra_el_directorio_temporal(self, monkeypatch):
        import os

        self._Good.instances.clear()
        self._install(monkeypatch, self._Good)
        ZActivationCheckpoint.prefer_zspace = True
        self._roundtrip(n=1)
        d = ZActivationCheckpoint._zspace_storage_dir
        assert d and os.path.isdir(d)
        ZActivationCheckpoint.reset()
        assert not os.path.exists(d)
