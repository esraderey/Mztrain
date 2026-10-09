"""Linea base sana (v1.5.0): embedding N(0, 0.02^2), AdamW beta2 0.95 / wd 0.01 / eps 1e-8, recorte
global 1.0 y calendario rampa + coseno por defecto (evidencia: docs/evidencia/T20-VEREDICTO.md).

Los valores esperados del calendario y del warmup post-crecimiento estan escritos a mano o con
una formula propia del test: no se calculan con las funciones del engine.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.utils.data as tud

from mztrain import ZTrainConfig, ZTrainEngine, ZFactorizedLinear
from mztrain.config import RankSchedule
from mztrain.elastic_shape import GPT, dense_lin, fact_lin, widen_gpt

CPU = torch.device("cpu")


# ------------------------------------------------------------------ embedding
def test_gpt_embeddings_son_los_de_antes_escalados_y_no_consumen_rng():
    torch.manual_seed(1)
    m = GPT(50, 16, d=32, layers=2, heads=4, lin=dense_lin)
    rng_nuevo = torch.get_rng_state()
    torch.manual_seed(1)
    viejo = GPT(50, 16, d=32, layers=2, heads=4, lin=dense_lin, emb_std=None)   # init de 1.4.0: N(0, 1)
    rng_viejo = torch.get_rng_state()
    assert m.emb_std == 0.02 and viejo.emb_std is None
    assert torch.equal(rng_nuevo, rng_viejo)                                    # el init sano no consume RNG
    assert torch.equal(m.tok.weight, viejo.tok.weight * 0.02) and torch.equal(m.pos.weight, viejo.pos.weight * 0.02)
    assert abs(float(viejo.tok.weight.std()) - 1.0) < 0.05 and abs(float(m.tok.weight.std()) - 0.02) < 0.002
    for (n1, p1), (n2, p2) in zip(m.blocks.named_parameters(), viejo.blocks.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), n1                             # bloques identicos a igual semilla
    assert torch.equal(m.lnf.weight, viejo.lnf.weight)
    torch.manual_seed(1)
    otro = GPT(50, 16, d=32, layers=2, heads=4, lin=fact_lin(8), emb_std=0.5)
    assert torch.equal(otro.tok.weight, viejo.tok.weight * 0.5)
    for malo in (-1.0, 0.0, float("nan"), float("inf"), True, "0.02"):
        with pytest.raises(ValueError):
            GPT(50, 16, d=32, layers=2, heads=4, lin=dense_lin, emb_std=malo)


def test_bpc_inicial_del_orden_del_uniforme_con_objetivos_desplazados():
    """Con la cabeza atada y N(0, 1), el modelo arranca prediciendo el token ACTUAL: con objetivos
    desplazados su perdida inicial se dispara; con 0.02 queda cerca de ln(vocab)."""
    torch.manual_seed(1)
    m = GPT(50, 16, d=32, layers=2, heads=4, lin=dense_lin)
    torch.manual_seed(1)
    viejo = GPT(50, 16, d=32, layers=2, heads=4, lin=dense_lin, emb_std=None)
    x = torch.randint(50, (4, 16))
    y = torch.roll(x, -1, dims=1)
    _, loss = m(x, y)
    _, loss_viejo = viejo(x, y)
    assert float(loss) / math.log(2) < 8.0 and float(loss_viejo) > 2.0 * float(loss)   # uniforme: 5.64 bits


def test_deriva_de_widen_bajo_la_linea_base_sana_vigilada_en_varias_semillas():
    """Interaccion medida, no contrato: las cotas de deriva de widen_gpt de la SPEC (<= 7 %) se
    midieron con embeddings N(0, 1). Con emb_std=0.02 la deriva relativa (ruido 0, toy 48 -> 72)
    es mayor y supera el 7 % en la mayoria de las semillas; aqui se vigila que no pase de 0.2."""
    derivas = []
    for seed in range(10):
        torch.manual_seed(seed)
        m = GPT(50, 16, d=48, layers=2, heads=2, lin=fact_lin(12))
        x = torch.randint(50, (2, 16))
        with torch.no_grad():
            ref = m(x)[0].clone()
        widen_gpt(m, 72, noise_scale=0.0)
        with torch.no_grad():
            derivas.append(float((m(x)[0] - ref).norm() / ref.norm()))
    assert max(derivas) <= 0.2 and min(derivas) >= 0.0, derivas


# ------------------------------------------------------------------ config y optimizador
def test_config_por_defecto_es_la_linea_base_sana():
    cfg = ZTrainConfig()
    cfg.validate()
    assert cfg.betas == (0.9, 0.95) and cfg.eps == 1e-8 and cfg.weight_decay == 0.01 and cfg.max_grad_norm == 1.0
    assert cfg.warmup_steps == 200 and cfg.lr_schedule == "warmup_cosine" and cfg.lr_warmup_floor == 0.1
    assert cfg.lr_final_factor == 0.0 and cfg.lr_warmup_max_fraction == 0.1
    ZTrainConfig(betas=[0.9, 0.95]).validate()                                           # lista (JSON / YAML)
    ZTrainConfig(betas=(np.float32(0.9), np.float32(0.95)), eps=np.float64(1e-8), weight_decay=np.float32(0.01)).validate()
    malos = [dict(betas=(0.9, 1.0)), dict(betas=(-0.1, 0.9)), dict(betas=(0.9,)), dict(betas=iter([0.9, 0.95])),
             dict(betas=(float("nan"), 0.9)), dict(betas=(True, 0.9)), dict(eps=0.0), dict(eps=float("nan")),
             dict(weight_decay=-1.0), dict(weight_decay=float("nan")), dict(weight_decay=float("inf")),
             dict(warmup_steps=-1), dict(warmup_steps=2.5), dict(warmup_steps=True), dict(lr_schedule="otro"),
             dict(lr_warmup_floor=1.5), dict(lr_warmup_floor=float("nan")), dict(lr_final_factor=-0.1),
             dict(lr_warmup_max_fraction=0.0), dict(lr_warmup_max_fraction=1.5)]
    for kw in malos:
        with pytest.raises(ValueError):
            ZTrainConfig(**kw).validate()


def _modelo():
    torch.manual_seed(0)
    return nn.Sequential(ZFactorizedLinear(16, 32, rank=4, init_method="svd", bias=True), nn.ReLU(),
                         ZFactorizedLinear(32, 8, rank=4, init_method="svd", bias=True))


def _cfg(**kw):
    base = dict(initial_rank=4, max_rank=4, rank_schedule=RankSchedule.CONSTANT, use_elastic_rank=False, use_amp=False,
                compress_optimizer_states=False, refactorize_interval=0)
    base.update(kw)
    return ZTrainConfig(**base)


def _loader(n=8, bs=4):
    torch.manual_seed(3)
    return tud.DataLoader(tud.TensorDataset(torch.randn(n, 16)), batch_size=bs)


def _loss(m, b):
    return m(b[0]).pow(2).mean()


def _espia(eng, registro):
    """loss_fn que anota, en el forward de cada paso, (paso global, warmup restante, LR que usara el paso)."""
    def loss(m, b):
        registro.append((eng._global_step, eng._rank_growth_warmup_remaining, eng.optimizer.param_groups[0]["lr"]))
        return m(b[0]).pow(2).mean()
    return loss


@pytest.mark.parametrize("opt_type", ["compressed_adam", "adaptive", "galore"])
def test_los_tres_optimizadores_reciben_betas_eps_y_wd_de_la_config(opt_type):
    eng = ZTrainEngine(_modelo(), _cfg(optimizer_type=opt_type, galore_rank=2), device=CPU)
    g = eng.optimizer.param_groups[0]
    assert tuple(g["betas"]) == (0.9, 0.95) and g["eps"] == 1e-8 and g["weight_decay"] == 0.01     # linea base sana
    eng2 = ZTrainEngine(_modelo(), _cfg(optimizer_type=opt_type, galore_rank=2, betas=(0.8, 0.999), eps=1e-6,
                                        weight_decay=0.1), device=CPU)
    g2 = eng2.optimizer.param_groups[0]
    assert tuple(g2["betas"]) == (0.8, 0.999) and g2["eps"] == 1e-6 and g2["weight_decay"] == 0.1


# ------------------------------------------------------------------ calendario
def _coseno(t, w, total, floor=0.1, fin=0.0):
    """Formula del test (independiente del engine): rampa lineal de w pasos y coseno hasta `fin`."""
    if w > 0 and t < w:
        return floor + (1.0 - floor) * t / w
    if total <= w:
        return 1.0
    p = min(1.0, (t - w) / (total - w))
    return fin + (1.0 - fin) * 0.5 * (1.0 + math.cos(math.pi * p))


def test_lr_factor_rampa_coseno_y_tope_de_la_rampa():
    eng = ZTrainEngine(_modelo(), _cfg(warmup_steps=4, lr_warmup_max_fraction=1.0), device=CPU)
    eng._lr_total_steps = 20
    f = eng.lr_factor
    assert [f(0), f(2), f(4), f(12), f(20), f(25)] == pytest.approx([0.1, 0.55, 1.0, 0.5, 0.0, 0.0], abs=1e-12)
    assert all(0.0 <= f(t) <= 1.0 for t in range(0, 30))
    # tope: la rampa no pasa del 10 % del presupuesto (por defecto 200 pasos)
    eng = ZTrainEngine(_modelo(), _cfg(), device=CPU)
    eng._lr_total_steps = 20
    assert eng.lr_warmup_efectivo() == 2
    assert [eng.lr_factor(t) for t in (0, 1, 2, 20)] == pytest.approx([0.1, 0.55, 1.0, 0.0], abs=1e-12)
    eng._lr_total_steps = 6000                                                    # regimen de T20: 200 pasos, sin tope
    assert eng.lr_warmup_efectivo() == 200
    assert [eng.lr_factor(t) for t in (0, 100, 200, 3100, 6000)] == pytest.approx([0.1, 0.55, 1.0, 0.5, 0.0], abs=1e-12)
    eng._lr_total_steps = 5                                                       # presupuesto minimo: sin rampa
    assert eng.lr_warmup_efectivo() == 0 and eng.lr_factor(0) == pytest.approx(1.0)
    eng = ZTrainEngine(_modelo(), _cfg(warmup_steps=4, lr_warmup_max_fraction=1.0, lr_final_factor=0.2, lr_warmup_floor=0.5), device=CPU)
    eng._lr_total_steps = 20
    assert [eng.lr_factor(t) for t in (0, 4, 12, 20)] == pytest.approx([0.5, 1.0, 0.6, 0.2], abs=1e-12)


def test_train_aplica_el_calendario_por_paso_y_cada_llamada_empieza_de_nuevo():
    eng = ZTrainEngine(_modelo(), _cfg(warmup_steps=2, lr_warmup_max_fraction=1.0, learning_rate=1e-3), device=CPU)
    esperado = [1e-3 * x for x in (0.1, 0.55, 1.0, 0.8535533906, 0.5, 0.1464466094)]   # rampa de 2 y coseno sobre 6 pasos
    reg = []
    eng.train(_loader(n=8, bs=4), None, _espia(eng, reg), epochs=3, early_stopping_patience=99)
    assert [t for t, _, _ in reg] == [0, 1, 2, 3, 4, 5]
    assert [lr for _, _, lr in reg] == pytest.approx(esperado, abs=1e-12)
    assert eng._lr_schedule_active is False
    assert eng.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)             # al salir, el LR vuelve al pico
    # segunda llamada: el calendario empieza de nuevo sobre su presupuesto (no sigue donde quedo)
    reg2 = []
    eng.train(_loader(n=8, bs=4), None, _espia(eng, reg2), epochs=3, early_stopping_patience=99)
    assert [t for t, _, _ in reg2] == [6, 7, 8, 9, 10, 11]
    assert [lr for _, _, lr in reg2] == pytest.approx(esperado, abs=1e-12)


def test_tras_train_una_train_epoch_directa_entrena_al_pico():
    eng = ZTrainEngine(_modelo(), _cfg(learning_rate=1e-3), device=CPU)
    eng.train(_loader(), None, _loss, epochs=1, early_stopping_patience=99)
    antes = [p.detach().clone() for p in eng.model.parameters()]
    reg = []
    eng.train_epoch(_loader(), _espia(eng, reg), 0)
    assert [lr for _, _, lr in reg] == pytest.approx([1e-3, 1e-3])                # ni 0 ni calendario heredado
    assert any(not torch.equal(a, p) for a, p in zip(antes, eng.model.parameters()))


def test_una_excepcion_dentro_de_train_desarma_el_calendario():
    eng = ZTrainEngine(_modelo(), _cfg(learning_rate=1e-3), device=CPU)
    llamadas = []

    def loss_que_falla(m, b):
        llamadas.append(1)
        if len(llamadas) == 2:
            raise RuntimeError("fallo a mitad")
        return m(b[0]).pow(2).mean()

    with pytest.raises(RuntimeError, match="fallo a mitad"):
        eng.train(_loader(), None, loss_que_falla, epochs=2, early_stopping_patience=99)
    assert eng._lr_schedule_active is False and eng.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    reg = []
    eng.train_epoch(_loader(), _espia(eng, reg), 0)
    assert [lr for _, _, lr in reg] == pytest.approx([1e-3, 1e-3])


def test_lr_schedule_none_deja_el_lr_constante_como_antes():
    eng = ZTrainEngine(_modelo(), _cfg(lr_schedule="none", learning_rate=1e-3), device=CPU)
    reg = []
    eng.train(_loader(), None, _espia(eng, reg), epochs=2, early_stopping_patience=99)
    assert [lr for _, _, lr in reg] == pytest.approx([1e-3] * 4)
    assert eng.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_un_scheduler_externo_manda():
    eng = ZTrainEngine(_modelo(), _cfg(learning_rate=1e-3), device=CPU)
    sch = torch.optim.lr_scheduler.StepLR(eng.optimizer, step_size=1, gamma=0.5)
    reg = []
    eng.train(_loader(), None, _espia(eng, reg), epochs=2, early_stopping_patience=99, scheduler=sch)
    assert [lr for _, _, lr in reg] == pytest.approx([1e-3, 1e-3, 5e-4, 5e-4])
    assert eng.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)           # el engine no lo toca al salir


def test_el_warmup_post_crecimiento_compone_con_el_calendario_con_la_forma_de_1_4_0():
    """Con el calendario activo, el warmup tras crecer el rango es un factor sobre el LR programado.
    Para rank_growth_warmup_steps = 4 la secuencia de 1.4.0 es 0.1, 0.1, 0.325, 0.55 y despues 1.0."""
    forma = {4: 0.1, 3: 0.1, 2: 0.325, 1: 0.55, 0: 1.0}                           # a mano: factor segun el warmup restante
    cfg = _cfg(initial_rank=4, max_rank=8, rank_schedule=RankSchedule.EXPONENTIAL, rank_growth_interval=1,
               rank_growth_warmup_steps=4, warmup_steps=0, learning_rate=1e-3)
    eng = ZTrainEngine(_modelo(), cfg, device=CPU)
    reg = []
    eng.train(_loader(n=32, bs=4), None, _espia(eng, reg), epochs=3, early_stopping_patience=99)   # 8 pasos por epoch, 24 en total
    assert len(reg) == 24 and {r for _, r, _ in reg} == {4, 3, 2, 1, 0}, sorted({r for _, r, _ in reg})
    for t, r, lr in reg:
        assert lr == pytest.approx(1e-3 * _coseno(t, 0, 24) * forma[r], abs=1e-12), (t, r, lr)
    crecio = [t for t, r, _ in reg if r == 4]
    assert crecio and all(t % 8 == 0 for t in crecio)                             # el arranque (0.1) cae en el primer paso del epoch


def test_un_crecimiento_durante_otro_warmup_arranca_en_0_1_del_lr_programado():
    """Dos crecimientos en epochs seguidos con un warmup de 8 pasos y 3 pasos por epoch: el segundo
    llega con el primero a medias. Su primer paso usa 0.1 x el LR programado, no 0.1 x un LR ya
    atenuado por el warmup anterior."""
    forma = {8: 0.1, 7: 0.1, 6: 0.2125, 5: 0.325, 4: 0.4375, 3: 0.55, 2: 0.6625, 1: 0.775, 0: 1.0}   # a mano (1.4.0)
    cfg = _cfg(initial_rank=4, max_rank=16, rank_schedule=RankSchedule.EXPONENTIAL, rank_growth_interval=1,
               rank_growth_warmup_steps=8, warmup_steps=0, learning_rate=1e-3)
    eng = ZTrainEngine(_modelo(), cfg, device=CPU)
    reg = []
    eng.train(_loader(n=12, bs=4), None, _espia(eng, reg), epochs=5, early_stopping_patience=99)   # 3 pasos por epoch, 15 en total
    restantes = [r for _, r, _ in reg]
    assert len(reg) == 15 and set(restantes) == set(forma), restantes
    durante = [i for i in range(1, 15) if restantes[i] == 8 and restantes[i - 1] > 0]
    assert durante, restantes                                                     # hubo un crecimiento con otro warmup en curso
    for t, r, lr in reg:
        assert lr == pytest.approx(1e-3 * _coseno(t, 0, 15) * forma[r], abs=1e-12), (t, r, lr)
