"""Anade tests de los contra-arreglos del G4 al final de TestZSpaceBackendAdapter (tests/test_checkpoint.py)."""
import pathlib
p = pathlib.Path("D:/mztrain/tests/test_checkpoint.py"); raw = p.read_bytes(); crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
assert "test_reset_conserva_prefer_zspace" in s and "test_sonda_usa_la_forma_real" not in s
s += '''
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
'''
if crlf: s = s.replace("\n", "\r\n")
p.write_bytes(s.encode("utf-8")); print("tests G4 anadidos")
