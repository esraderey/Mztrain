"""
Anclas de regresion del peritaje 2026-08-10.

Cada test reproduce un hallazgo confirmado: FALLA antes del arreglo, PASA despues.
No debilitar estos tests para obtener verde (regla de forja/maestranza).

  A1: seal.py excluye src/mztrain/data/ del sello ("data" en EXCLUDE_DIRS casa a
      cualquier profundidad) -> tokenizer/dataset manipulables sin que verify proteste.
  N1: files_canonical_sha256 se publica pero verify nunca lo comprueba.
  A2: _grow_model_rank descarta el estado Adam de params no-factorizados (y de U/S/V
      cuando estan comprimidos, el default).
  A3: load_checkpoint crashea al reanudar tras un crecimiento de rango.
  A5: el store de activaciones no libera la clave tras decompress (rama ZSpace).
  M7: rank_growth_warmup_steps=0 deja el LR clavado en 0.1x sin restaurar.
  M10: el sleep bank guarda S en fp16, violando la regla "S SIEMPRE FP32".
  B2: CodeTokenizer.encode materializa todo el input antes de truncar.
  B3: estimate_memory_savings cuenta doble el bias de capas factorizables.
  B4: LINEAR/COSINE nunca alcanzan max_rank en la ultima epoch (off-by-one).
  B5: growth_interval=0 provoca ZeroDivisionError sin validacion.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _make_growth_engine(compress: bool, warmup: int = 50):
    from mztrain import ZTrainEngine, ZTrainConfig, ZFactorizedLinear
    from mztrain.config import RankSchedule
    torch.manual_seed(0)
    model = nn.Sequential(
        ZFactorizedLinear(16, 32, rank=4, init_method="svd", bias=True),
        nn.ReLU(),
        ZFactorizedLinear(32, 8, rank=4, init_method="svd", bias=True),
    )
    cfg = ZTrainConfig(
        initial_rank=4, max_rank=16,
        rank_schedule=RankSchedule.EXPONENTIAL,
        use_elastic_rank=False,
        compress_optimizer_states=compress,
        rank_growth_warmup_steps=warmup,
        use_amp=False,
    )
    return ZTrainEngine(model, cfg)


def _factor_layers(model):
    from mztrain import ZFactorizedLinear
    return [x for x in model.modules() if isinstance(x, ZFactorizedLinear)]


def _nonzero_state(opt, p) -> bool:
    from mztrain.elastic_rank import decompress_opt_state
    st = opt.state.get(p)
    if not st:
        return False
    ea = decompress_opt_state(opt, st, "exp_avg", p)
    return ea is not None and float(ea.abs().sum()) > 0


def _train_steps(eng, n):
    x = torch.randn(24, 16, device=eng.device)
    for _ in range(n):
        eng.optimizer.zero_grad()
        eng.model(x).pow(2).mean().backward()
        eng.optimizer.step()


def _load_seal():
    spec = importlib.util.spec_from_file_location(
        "seal_mod", ROOT / "scripts" / "seal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# A1 — el sello debe cubrir src/mztrain/data/
# ----------------------------------------------------------------------

class TestA1SealCoversDataPackage:
    def test_src_mztrain_data_is_sealed(self):
        seal = _load_seal()
        p = ROOT / "src" / "mztrain" / "data" / "tokenizer.py"
        assert seal._should_include(p, ROOT) is True, (
            "A1: src/mztrain/data/tokenizer.py debe entrar al sello; "
            "el nombre 'data' se excluia a cualquier profundidad."
        )

    def test_root_data_dir_still_excluded(self):
        # control: el directorio de datos en la RAIZ sigue excluido
        seal = _load_seal()
        p = ROOT / "data" / "some_dataset.py"
        assert seal._should_include(p, ROOT) is False


# ----------------------------------------------------------------------
# N1 — verify debe comprobar el hash canonico global
# ----------------------------------------------------------------------

class TestN1CanonicalHashEnforced:
    def test_tampered_canonical_hash_fails_verify(self, tmp_path):
        import json
        seal = _load_seal()
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("# doc\n", encoding="utf-8")
        assert seal.cmd_seal(tmp_path) == 0

        sealp = tmp_path / "SEAL.json"
        data = json.loads(sealp.read_text(encoding="utf-8"))
        data["files_canonical_sha256"] = "0" * 64  # corromper SOLO el canonico
        sealp.write_text(json.dumps(data, indent=2), encoding="utf-8")

        assert seal.cmd_verify(tmp_path) == 1, (
            "N1: un files_canonical_sha256 corrupto debe hacer fallar verify."
        )


# ----------------------------------------------------------------------
# M1 — firma Ed25519 (autoria)
# ----------------------------------------------------------------------

class TestM1SealSignature:
    def test_sign_verify_roundtrip_and_tamper(self, tmp_path, monkeypatch):
        seal = _load_seal()
        if not seal.HAS_CRYPTO:
            pytest.skip("cryptography no disponible")
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        key = tmp_path / "priv.pem"
        pub = seal.cmd_keygen(key)
        assert pub and key.exists()
        assert seal.cmd_sign(tmp_path, key) == 0
        # roundtrip en modo estricto: la clave del test es la de confianza
        # (aislado de la TRUSTED_PUBLIC_KEYS de produccion, que ya esta poblada)
        monkeypatch.setattr(seal, "TRUSTED_PUBLIC_KEYS", [pub])
        assert seal.cmd_verify(tmp_path) == 0  # firma valida
        # tamper tras firmar: el merkle cambia -> la firma ya no valida
        (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
        assert seal.cmd_verify(tmp_path) == 1

    def test_verify_rejects_untrusted_signer(self, tmp_path, monkeypatch):
        seal = _load_seal()
        if not seal.HAS_CRYPTO:
            pytest.skip("cryptography no disponible")
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        key = tmp_path / "priv.pem"
        seal.cmd_keygen(key)
        seal.cmd_sign(tmp_path, key)
        # una lista de confianza que NO incluye la clave firmante -> rechazo
        monkeypatch.setattr(seal, "TRUSTED_PUBLIC_KEYS", ["ab" * 32])
        assert seal.cmd_verify(tmp_path) == 1

    def test_v2_populated_trust_requires_signature(self, tmp_path, monkeypatch):
        # downgrade: sello SIN firma con lista de confianza poblada -> debe fallar
        seal = _load_seal()
        if not seal.HAS_CRYPTO:
            pytest.skip("cryptography no disponible")
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        key = tmp_path / "priv.pem"
        pub = seal.cmd_keygen(key)
        seal.cmd_seal(tmp_path)  # sello sin firma
        monkeypatch.setattr(seal, "TRUSTED_PUBLIC_KEYS", [pub])
        assert seal.cmd_verify(tmp_path) == 1, "V2: sello sin firma no rechazado"
        # firmado por la clave de confianza -> OK
        seal.cmd_sign(tmp_path, key)
        assert seal.cmd_verify(tmp_path) == 0

    def test_v3_signature_covers_metadata(self, tmp_path, monkeypatch):
        # falsificar generated_at_utc sin tocar files -> la firma debe fallar
        seal = _load_seal()
        if not seal.HAS_CRYPTO:
            pytest.skip("cryptography no disponible")
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        key = tmp_path / "priv.pem"
        pub = seal.cmd_keygen(key)
        seal.cmd_sign(tmp_path, key)
        monkeypatch.setattr(seal, "TRUSTED_PUBLIC_KEYS", [pub])
        assert seal.cmd_verify(tmp_path) == 0
        import json
        sp = tmp_path / "SEAL.json"
        d = json.loads(sp.read_text(encoding="utf-8"))
        d["generated_at_utc"] = "1999-01-01T00:00:00Z"  # antedatar
        sp.write_text(json.dumps(d, indent=2), encoding="utf-8")
        assert seal.cmd_verify(tmp_path) == 1, "V3: metadatos fuera de la firma"

    def test_v4_fail_closed_without_crypto(self, tmp_path, monkeypatch):
        seal = _load_seal()
        if not seal.HAS_CRYPTO:
            pytest.skip("cryptography no disponible")
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        key = tmp_path / "priv.pem"
        pub = seal.cmd_keygen(key)
        seal.cmd_sign(tmp_path, key)
        monkeypatch.setattr(seal, "TRUSTED_PUBLIC_KEYS", [pub])
        monkeypatch.setattr(seal, "HAS_CRYPTO", False)  # simular ausencia de la lib
        assert seal.cmd_verify(tmp_path) == 1, "V4: fail-open sin cryptography"


# ----------------------------------------------------------------------
# M5 — oom_guarded cableado al train loop + retry que libera de verdad
# ----------------------------------------------------------------------

def _governor(retry: bool):
    from mztrain.vram_governor import ZVRAMGovernor
    from mztrain.config import ZTrainConfig
    return ZVRAMGovernor(ZTrainConfig(use_vram_governor=True, vram_oom_retry=retry))


class TestM5OomRecovery:
    def test_oom_guarded_retries_once(self):
        gov = _governor(retry=True)
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.cuda.OutOfMemoryError("simulado")
            return 42

        assert gov.oom_guarded(fn) == 42
        assert calls["n"] == 2

    def test_oom_guarded_no_retry_propagates(self):
        gov = _governor(retry=False)

        def fn():
            raise torch.cuda.OutOfMemoryError("simulado")

        with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError)):
            gov.oom_guarded(fn)

    def test_oom_guarded_ignores_non_oom(self):
        gov = _governor(retry=True)

        def fn():
            raise ValueError("no es OOM")

        with pytest.raises(ValueError):
            gov.oom_guarded(fn)

    def test_train_recovers_from_transient_oom(self):
        # ancla del cableado: sin envolver el step en oom_guarded, el OOM
        # transitorio propaga y train() crashea.
        import torch.utils.data as tud
        from mztrain import ZTrainEngine, ZTrainConfig, ZFactorizedLinear
        from mztrain.config import RankSchedule
        torch.manual_seed(0)
        model = nn.Sequential(
            ZFactorizedLinear(16, 8, rank=4, init_method="svd", bias=True)
        )
        cfg = ZTrainConfig(
            initial_rank=4, max_rank=4, rank_schedule=RankSchedule.CONSTANT,
            use_vram_governor=True, vram_oom_retry=True, use_amp=False,
        )
        eng = ZTrainEngine(model, cfg)
        fired = {"done": False}

        def loss_fn(m, b):
            out = m(b[0]).pow(2).mean()
            if not fired["done"]:
                fired["done"] = True
                raise torch.cuda.OutOfMemoryError("transient")
            return out

        x = torch.randn(8, 16, device=eng.device)
        loader = tud.DataLoader(tud.TensorDataset(x), batch_size=8)
        eng.train(loader, None, loss_fn, epochs=1, early_stopping_patience=99)
        assert fired["done"], "el OOM simulado deberia haberse disparado y recuperado"

    def test_oom_retry_does_not_double_compress(self):
        # M5-1: la compresion de gradientes es stateful (muta el error-feedback);
        # un OOM en forward + retry no debe ejecutarla dos veces.
        import torch.utils.data as tud
        from mztrain import ZTrainEngine, ZTrainConfig, ZFactorizedLinear
        from mztrain.config import RankSchedule, GradientCompression

        def build():
            torch.manual_seed(0)
            m = nn.Sequential(
                ZFactorizedLinear(16, 8, rank=4, init_method="svd", bias=True)
            )
            cfg = ZTrainConfig(
                initial_rank=4, max_rank=4, rank_schedule=RankSchedule.CONSTANT,
                use_vram_governor=True, vram_oom_retry=True, use_amp=False,
                gradient_compression=GradientCompression.TOP_K,
                min_size_to_compress_grad=1,
            )
            return ZTrainEngine(m, cfg)

        x = torch.randn(8, 16)
        e1 = build()
        l1 = tud.DataLoader(tud.TensorDataset(x.to(e1.device)), batch_size=8)
        e1.train(l1, None, lambda m, b: m(b[0]).pow(2).mean(),
                 epochs=1, early_stopping_patience=99)
        base = e1.grad_compressor.get_stats().get("total_compressed", 0)

        e2 = build()
        l2 = tud.DataLoader(tud.TensorDataset(x.to(e2.device)), batch_size=8)
        fired = {"done": False}

        def loss_fn(m, b):
            out = m(b[0]).pow(2).mean()
            if not fired["done"]:
                fired["done"] = True
                raise torch.cuda.OutOfMemoryError("transient")
            return out

        e2.train(l2, None, loss_fn, epochs=1, early_stopping_patience=99)
        with_oom = e2.grad_compressor.get_stats().get("total_compressed", 0)
        assert base > 0 and with_oom == base, (
            f"M5-1: OOM+retry duplico la compresion ({with_oom} vs {base})."
        )


# ----------------------------------------------------------------------
# B4 / B5 — scheduler
# ----------------------------------------------------------------------

class TestB4B5Scheduler:
    def test_linear_reaches_max_rank_on_last_epoch(self):
        from mztrain.scheduler import ZRankScheduler
        from mztrain.config import RankSchedule
        s = ZRankScheduler(4, 16, total_epochs=10, schedule=RankSchedule.LINEAR)
        # el bucle recorre epoch in range(total_epochs) => ultima epoch = 9
        assert s.get_rank(epoch=9) == 16, (
            "B4: LINEAR nunca alcanzaba max_rank (progress=epoch/total<1)."
        )

    def test_cosine_reaches_max_rank_on_last_epoch(self):
        from mztrain.scheduler import ZRankScheduler
        from mztrain.config import RankSchedule
        s = ZRankScheduler(4, 16, total_epochs=10, schedule=RankSchedule.COSINE)
        assert s.get_rank(epoch=9) == 16

    def test_exponential_growth_interval_zero_no_crash(self):
        from mztrain.scheduler import ZRankScheduler
        from mztrain.config import RankSchedule
        s = ZRankScheduler(
            4, 16, total_epochs=10,
            schedule=RankSchedule.EXPONENTIAL, growth_interval=0,
        )
        r = s.get_rank(epoch=3)  # antes: ZeroDivisionError
        assert 4 <= r <= 16


# ----------------------------------------------------------------------
# B3 — estimate_memory_savings no debe contar doble el bias
# ----------------------------------------------------------------------

class TestB3BiasDoubleCount:
    def test_bias_not_double_counted(self):
        import torch.nn as nn
        from mztrain.utils import estimate_memory_savings
        est = estimate_memory_savings(nn.Linear(1024, 1024, bias=True), rank=32)
        d = est["details"]
        non_fact = d["total_params"] - d["factorizable_params"]
        assert non_fact == 0, (
            f"B3: el bias de una capa factorizable queda en non_factorizable "
            f"({non_fact}) Y en factorized_params -> doble conteo."
        )


# ----------------------------------------------------------------------
# B2 — encode acota el input crudo antes de tokenizar
# ----------------------------------------------------------------------

class TestB2TokenizerBound:
    def test_encode_caps_raw_input_before_split(self, monkeypatch):
        from mztrain.data.tokenizer import CodeTokenizer
        tok = CodeTokenizer(max_length=16)
        seen = {}
        orig = tok._split_code

        def spy(text):
            seen["len"] = len(text)
            return orig(text)

        monkeypatch.setattr(tok, "_split_code", spy)
        tok.encode("x " * 1_000_000)  # 2M chars
        assert seen["len"] <= 16 * CodeTokenizer._MAX_CHARS_PER_TOKEN, (
            "B2: _split_code recibio el input completo sin acotar."
        )


# ----------------------------------------------------------------------
# config — validaciones ausentes (rank_growth_interval / warmup_steps)
# ----------------------------------------------------------------------

class TestConfigValidations:
    def test_rejects_zero_growth_interval(self):
        from mztrain.config import ZTrainConfig
        with pytest.raises(ValueError):
            ZTrainConfig(rank_growth_interval=0).validate()

    def test_rejects_negative_warmup_steps(self):
        from mztrain.config import ZTrainConfig
        with pytest.raises(ValueError):
            ZTrainConfig(rank_growth_warmup_steps=-1).validate()


# ----------------------------------------------------------------------
# A5 — el store de activaciones libera la clave tras decompress (rama ZSpace)
# ----------------------------------------------------------------------

class TestA5ActivationStoreFrees:
    def test_zspace_key_freed_after_decompress(self, monkeypatch):
        import torch
        from mztrain.checkpoint import ZActivationCheckpoint as AC

        class FakeZSpace:
            def __init__(self):
                self.store = {}

            def register(self, k, t):
                self.store[k] = t

            def load(self, k, device=None):
                return self.store[k]

            def remove(self, k):
                self.store.pop(k, None)

        fake = FakeZSpace()
        AC.reset()
        monkeypatch.setattr(AC, "_get_zspace", classmethod(lambda cls: fake))

        key, shape, dtype = AC.compress_activation(torch.randn(4))
        assert key in fake.store, "precondicion: la activacion se registro"
        AC.decompress_activation(key, shape, dtype)
        assert key not in fake.store, (
            "A5: el store ZSpace retuvo la clave tras decompress -> leak ~ steps."
        )


# ----------------------------------------------------------------------
# A2 — _grow_model_rank preserva el estado Adam de TODOS los parametros
# ----------------------------------------------------------------------

class TestA2GrowthPreservesOptState:
    def test_bias_state_survives_growth_uncompressed(self):
        eng = _make_growth_engine(compress=False)
        _train_steps(eng, 6)
        bias = _factor_layers(eng.model)[0].bias
        assert _nonzero_state(eng.optimizer, bias), "precondicion"
        eng._grow_model_rank(8)
        bias_after = _factor_layers(eng.model)[0].bias
        assert _nonzero_state(eng.optimizer, bias_after), (
            "A2: el momentum del bias (no factorizado) se perdio en el grow."
        )

    def test_factor_state_survives_growth_compressed(self):
        eng = _make_growth_engine(compress=True)  # default del framework
        _train_steps(eng, 10)  # step 10 => estado comprimido (tupla)
        U = _factor_layers(eng.model)[0].U
        assert _nonzero_state(eng.optimizer, U), "precondicion"
        eng._grow_model_rank(8)
        U_after = _factor_layers(eng.model)[0].U
        assert _nonzero_state(eng.optimizer, U_after), (
            "A2: el momentum de U se perdio (formato comprimido saltado en migracion)."
        )


# ----------------------------------------------------------------------
# A3 — load_checkpoint reconstruye la topologia antes de cargar
# ----------------------------------------------------------------------

class TestA3ResumeAfterGrowth:
    def test_resume_after_rank_growth(self, tmp_path):
        eng = _make_growth_engine(compress=False)
        _train_steps(eng, 3)
        eng._grow_model_rank(8)
        if eng.rank_scheduler is not None:
            eng.rank_scheduler.current_rank = 8
        ckpt = str(tmp_path / "ck.pt")
        eng.save_checkpoint(ckpt)

        eng2 = _make_growth_engine(compress=False)  # empieza en rank 4
        eng2.load_checkpoint(ckpt)  # antes: RuntimeError size mismatch
        assert _factor_layers(eng2.model)[0].rank == 8, (
            "A3: el modelo no se reconstruyo al rango del checkpoint."
        )

    def test_resume_with_nonuniform_ranks(self, tmp_path):
        # checkpoint de ElasticRank: cada capa con un rango distinto. El A3
        # uniforme (crecer todas al max) daria size mismatch en la capa menor.
        eng = _make_growth_engine(compress=False)
        layers = _factor_layers(eng.model)
        layers[0].grow_rank(10)
        layers[1].grow_rank(6)
        eng.optimizer = eng._create_optimizer()
        _train_steps(eng, 1)
        ckpt = str(tmp_path / "ck_nu.pt")
        eng.save_checkpoint(ckpt)

        eng2 = _make_growth_engine(compress=False)  # ambas en rank 4
        eng2.load_checkpoint(ckpt)
        l2 = _factor_layers(eng2.model)
        assert l2[0].rank == 10 and l2[1].rank == 6, (
            f"A3 por-capa: rangos reconstruidos {l2[0].rank},{l2[1].rank} "
            f"(esperado 10,6)."
        )

    def test_resume_reduces_rank(self, tmp_path):
        # checkpoint con rango MENOR que el modelo destino (capa comprimida por
        # ElasticRank / initial_rank mayor): load debe REDUCIR, no solo crecer.
        from mztrain import ZTrainEngine, ZTrainConfig, ZFactorizedLinear
        from mztrain.config import RankSchedule
        eng = _make_growth_engine(compress=False)  # 2 capas en rank 4
        _train_steps(eng, 1)
        ckpt = str(tmp_path / "ck_reduce.pt")
        eng.save_checkpoint(ckpt)

        torch.manual_seed(0)
        model = nn.Sequential(
            ZFactorizedLinear(16, 32, rank=8, init_method="svd", bias=True),
            nn.ReLU(),
            ZFactorizedLinear(32, 8, rank=8, init_method="svd", bias=True),
        )
        cfg = ZTrainConfig(
            initial_rank=8, max_rank=16, rank_schedule=RankSchedule.CONSTANT,
            use_elastic_rank=False, compress_optimizer_states=False, use_amp=False,
        )
        eng2 = ZTrainEngine(model, cfg)  # capas en rank 8
        eng2.load_checkpoint(ckpt)       # debe reducir a rank 4
        assert all(l.rank == 4 for l in _factor_layers(eng2.model)), (
            "A3-1: no se redujo el rango al del checkpoint."
        )


# ----------------------------------------------------------------------
# M7 — rank_growth_warmup_steps=0 no debe clavar el LR
# ----------------------------------------------------------------------

class TestM7WarmupZero:
    def test_warmup_zero_keeps_lr(self):
        eng = _make_growth_engine(compress=False, warmup=0)
        _train_steps(eng, 3)
        lr0 = eng.optimizer.param_groups[0]["lr"]
        eng._grow_model_rank(8)
        lr1 = eng.optimizer.param_groups[0]["lr"]
        assert lr1 == pytest.approx(lr0), (
            f"M7: con warmup_steps=0 el LR quedo clavado ({lr1} vs {lr0})."
        )

    def test_warmup_zero_survives_refactorization(self):
        # tercer sitio del patron de warmup (ruta de refactorizacion):
        # con warmup_steps=0 el LR se clavaba en 0.1x y decaia geometricamente.
        import torch.utils.data as tud
        from mztrain import ZTrainConfig, ZTrainEngine, ZFactorizedLinear
        from mztrain.config import RankSchedule
        torch.manual_seed(0)
        model = nn.Sequential(
            ZFactorizedLinear(16, 32, rank=6, init_method="svd", bias=True),
            nn.ReLU(),
            ZFactorizedLinear(32, 8, rank=6, init_method="svd", bias=True),
        )
        cfg = ZTrainConfig(
            initial_rank=6, max_rank=8,
            rank_schedule=RankSchedule.CONSTANT,   # aislar la ruta refactor
            use_elastic_rank=False,
            refactorize_interval=1,                # refactorizar cada batch
            rank_growth_warmup_steps=0,
            use_amp=False,
        )
        eng = ZTrainEngine(model, cfg)
        lr0 = eng.optimizer.param_groups[0]["lr"]
        x = torch.randn(16, 16, device=eng.device)
        loader = tud.DataLoader(tud.TensorDataset(x), batch_size=4)
        eng.train(
            loader, None,
            lambda m, b: m(b[0]).pow(2).mean(),
            epochs=2, early_stopping_patience=99,
        )
        lr1 = eng.optimizer.param_groups[0]["lr"]
        assert lr1 == pytest.approx(lr0), (
            f"M7-refactor: el LR colapso por refactorizacion ({lr1} vs {lr0})."
        )


# ----------------------------------------------------------------------
# A4 — invalidacion explicita del error-feedback de gradientes
# ----------------------------------------------------------------------

class TestA4ErrorFeedbackInvalidation:
    def test_invalidate_clears_buffer(self):
        from mztrain.gradient import ZGradientCompressor
        from mztrain.config import GradientCompression
        c = ZGradientCompressor(
            method=GradientCompression.TOP_K,
            top_k_ratio=0.5, min_size_to_compress=1,
        )
        c.compress("layer.U", torch.randn(8))
        assert "layer.U" in c._error_feedback, "precondicion: buffer creado"
        c.invalidate("layer.U")
        assert "layer.U" not in c._error_feedback, (
            "A4: invalidate no limpio el error-feedback del parametro."
        )


# ----------------------------------------------------------------------
# MATH-A1 — RANDOM_K no debe divergir (error-feedback expansivo)
# ----------------------------------------------------------------------

class TestMathA1RandomKNoDivergence:
    def test_random_k_output_bounded(self):
        from mztrain.gradient import ZGradientCompressor
        from mztrain.config import GradientCompression
        c = ZGradientCompressor(
            method=GradientCompression.RANDOM_K,
            top_k_ratio=0.1, min_size_to_compress=1,
        )
        torch.manual_seed(0)
        g = torch.randn(2000)
        gnorm = g.norm().item()
        out = None
        for _ in range(30):
            out = c.compress("p", g.clone())
        # con EF (bug) el buffer explota geometricamente y ||out|| >> ||g||;
        # RANDOM_K insesgado sin EF: ||out|| ~ sqrt(1/p)*||g|| ~ 3.16*||g||
        assert out.norm().item() < 100 * gnorm, (
            f"MATH-A1: RANDOM_K+EF divergio: ||out||={out.norm().item():.1f} "
            f"vs ||g||={gnorm:.1f}"
        )


# ----------------------------------------------------------------------
# MATH-A2 — GaLore no debe colapsar v (2o momento) con INT8 lineal
# ----------------------------------------------------------------------

class TestMathA2GaLoreVNoCollapse:
    def test_galore_v_small_values_survive(self):
        import torch.nn as nn
        from mztrain.projector import ZGaLoreOptimizer
        opt = ZGaLoreOptimizer([nn.Parameter(torch.zeros(1))], lr=1e-3)
        # bloque con un outlier grande + valores pequenos: INT8 lineal los
        # redondea a 0 -> denom=eps -> paso ~1/eps. La log-quant los preserva.
        v = torch.cat([torch.tensor([100.0]), torch.full((2047,), 1e-4)])
        comp = opt._compress_v(v)
        rec = opt._decompress_v(comp, v.dtype)
        assert (rec[1:] > 0).all(), "MATH-A2: v pequeno colapso a 0 (cuant lineal)"
        rel = ((v[1:] - rec[1:]).abs() / v[1:]).max().item()
        assert rel < 0.5, f"MATH-A2: error relativo de v excesivo: {rel}"


# ----------------------------------------------------------------------
# M6 — EPSI opcional: preservar el mapa lineal en transferencia de pesos
# ----------------------------------------------------------------------

class TestM6EpsiPreserveMap:
    def test_epsi_off_gives_faithful_truncated_svd(self):
        from mztrain import ZFactorizedLinear
        torch.manual_seed(0)
        W = torch.randn(64, 64)  # espectro no concentrado -> EPSI infla alpha
        z_epsi = ZFactorizedLinear(64, 64, rank=8, bias=False,
                                   existing_weight=W, epsi_scaling=True)
        z_map = ZFactorizedLinear(64, 64, rank=8, bias=False,
                                  existing_weight=W, epsi_scaling=False)
        # sin EPSI la reconstruccion es la SVD truncada pura (mapa fiel) y su
        # error es estrictamente menor que con EPSI para un espectro no concentrado
        assert z_map._reconstruction_error < z_epsi._reconstruction_error
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        W_trunc = (U[:, :8] * S[:8]) @ Vh[:8, :]
        err_theory = float((W - W_trunc).norm() / W.norm())
        assert abs(z_map._reconstruction_error - err_theory) < 1e-5

    def test_factorize_existing_preserve_map(self):
        from mztrain.utils import factorize_existing_model
        torch.manual_seed(0)
        lin = nn.Linear(64, 64, bias=False)
        zf, _ = factorize_existing_model(
            lin, rank=8, min_params=1, preserve_map=True
        )
        U, S, Vh = torch.linalg.svd(lin.weight.data, full_matrices=False)
        W_trunc = (U[:, :8] * S[:8]) @ Vh[:8, :]
        err = float((lin.weight.data - W_trunc).norm() / lin.weight.data.norm())
        assert abs(zf._reconstruction_error - err) < 1e-5


# ----------------------------------------------------------------------
# M4 — train() limpia el estado de clase del checkpointing de activaciones
# ----------------------------------------------------------------------

class TestM4CheckpointStateReset:
    def test_train_clears_stale_class_state(self):
        import torch.utils.data as tud
        from mztrain.checkpoint import ZActivationCheckpoint as AC

        AC._device_map["stale_key_from_prev_run"] = torch.device("cpu")
        eng = _make_growth_engine(compress=False, warmup=0)
        x = torch.randn(8, 16, device=eng.device)
        loader = tud.DataLoader(tud.TensorDataset(x), batch_size=4)
        eng.train(
            loader, None,
            lambda m, b: m(b[0]).pow(2).mean(),
            epochs=1, early_stopping_patience=99,
        )
        assert "stale_key_from_prev_run" not in AC._device_map, (
            "M4: train() no limpio el estado de clase de un run previo."
        )
