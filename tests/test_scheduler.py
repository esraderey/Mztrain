"""
Tests para mztrain.scheduler - ZRankScheduler.
"""

import pytest

from mztrain.config import RankSchedule
from mztrain.scheduler import ZRankScheduler


class TestZRankScheduler:
    """Tests para ZRankScheduler."""

    def test_constant(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=256, total_epochs=100,
            schedule=RankSchedule.CONSTANT,
        )
        for epoch in range(100):
            assert scheduler.get_rank(epoch) == 32

    def test_linear(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=256, total_epochs=100,
            schedule=RankSchedule.LINEAR,
        )
        rank_0 = scheduler.get_rank(0)
        rank_50 = scheduler.get_rank(50)
        rank_100 = scheduler.get_rank(100)
        assert rank_0 == 32
        assert rank_50 > rank_0
        assert rank_100 >= rank_50

    def test_exponential(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=256, total_epochs=100,
            schedule=RankSchedule.EXPONENTIAL,
            growth_interval=10, growth_factor=2.0,
        )
        assert scheduler.get_rank(0) == 32
        assert scheduler.get_rank(10) == 64
        assert scheduler.get_rank(20) == 128
        assert scheduler.get_rank(30) == 256  # Clamped to max

    def test_cosine(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=256, total_epochs=100,
            schedule=RankSchedule.COSINE,
        )
        ranks = [scheduler.get_rank(e) for e in range(101)]
        # Deberia crecer monotonicamente (o al menos no decrecer mucho)
        assert ranks[0] == 32
        assert ranks[-1] == 256
        # Medio deberia estar entre inicio y final
        assert 32 < ranks[50] < 256

    def test_adaptive(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=256, total_epochs=100,
            schedule=RankSchedule.ADAPTIVE,
            growth_interval=3, growth_factor=2.0,
            adaptive_threshold=0.01,
        )
        # Simular loss estancado y simular tambien el lado del engine: cuando
        # get_rank devuelve un rango mayor, el "engine" lo aplica actualizando
        # current_rank. (get_rank por si solo es read-only desde el fix de bug #1.)
        ranks = []
        for e in range(10):
            new_rank = scheduler.get_rank(e, current_loss=1.0)
            ranks.append(new_rank)
            if new_rank > scheduler.current_rank:
                scheduler.current_rank = new_rank  # rol del engine
        # Deberia haber crecido en algun momento
        assert max(ranks) > 32
        assert scheduler.current_rank > 32

    def test_adaptive_improving(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=256, total_epochs=100,
            schedule=RankSchedule.ADAPTIVE,
            growth_interval=3, adaptive_threshold=0.01,
        )
        # Simular loss mejorando, con el engine aplicando los growths.
        ranks = []
        for e in range(10):
            new_rank = scheduler.get_rank(e, current_loss=10.0 - e * 1.0)
            ranks.append(new_rank)
            if new_rank > scheduler.current_rank:
                scheduler.current_rank = new_rank
        # No deberia crecer mucho si loss mejora
        assert max(ranks) <= 64
        assert scheduler.current_rank <= 64

    def test_max_rank_clamp(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=128, total_epochs=100,
            schedule=RankSchedule.EXPONENTIAL,
            growth_interval=5, growth_factor=2.0,
        )
        for epoch in range(100):
            rank = scheduler.get_rank(epoch)
            assert rank <= 128

    def test_should_grow_constant(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=256, total_epochs=100,
            schedule=RankSchedule.CONSTANT,
        )
        assert not scheduler.should_grow(0)
        assert not scheduler.should_grow(50)

    def test_should_grow_exponential(self):
        scheduler = ZRankScheduler(
            initial_rank=32, max_rank=256, total_epochs=100,
            schedule=RankSchedule.EXPONENTIAL,
            growth_interval=10,
        )
        # No deberia crecer antes del intervalo
        assert not scheduler.should_grow(5)

    def test_initial_rank_preserved(self):
        scheduler = ZRankScheduler(
            initial_rank=64, max_rank=256, total_epochs=100,
            schedule=RankSchedule.LINEAR,
        )
        # Nunca deberia ir por debajo del rango inicial
        for epoch in range(100):
            assert scheduler.get_rank(epoch) >= 64
