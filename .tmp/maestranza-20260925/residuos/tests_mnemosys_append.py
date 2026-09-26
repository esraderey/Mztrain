"""Anade a tests/test_checkpoint.py los tests del adaptador de backend MNEME/MNEMOSYS (D5 del contrato)."""
import pathlib
p = pathlib.Path("D:/mztrain/tests/test_checkpoint.py"); raw = p.read_bytes(); crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
assert "class TestZSpaceBackendAdapter" not in s
s += '''

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
        with pytest.warns(RuntimeWarning, match="remove\\(\\)/delete\\(\\)"):
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
'''
if crlf: s = s.replace("\n", "\r\n")
p.write_bytes(s.encode("utf-8")); print("tests anadidos")
