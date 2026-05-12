import pytest
from datetime import datetime, timezone, timedelta
from lib.expedition.scorer import (
    Rarity, Newness, ScoreResult,
    compute_rarity, compute_newness, compute_score,
)

def _dt(days_ago: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()

class TestComputeRarity:
    def test_legendary(self):
        assert compute_rarity(15_000_000) == Rarity.LEGENDARY
    def test_legendary_boundary(self):
        assert compute_rarity(10_000_000) == Rarity.LEGENDARY
    def test_rare(self):
        assert compute_rarity(5_000_000) == Rarity.RARE
    def test_rare_boundary(self):
        assert compute_rarity(1_000_000) == Rarity.RARE
    def test_uncommon(self):
        assert compute_rarity(500_000) == Rarity.UNCOMMON
    def test_uncommon_boundary(self):
        assert compute_rarity(100_000) == Rarity.UNCOMMON
    def test_common(self):
        assert compute_rarity(50_000) == Rarity.COMMON
    def test_common_zero(self):
        assert compute_rarity(0) == Rarity.COMMON
    def test_none_is_familiar(self):
        assert compute_rarity(None) == Rarity.FAMILIAR

class TestComputeNewness:
    def test_zero_day(self):
        assert compute_newness(_dt(0.5), is_first_ever=True) == Newness.ZERO_DAY
    def test_hot(self):
        assert compute_newness(_dt(3), is_first_ever=True) == Newness.HOT
    def test_fresh(self):
        assert compute_newness(_dt(15), is_first_ever=True) == Newness.FRESH
    def test_recent(self):
        assert compute_newness(_dt(60), is_first_ever=True) == Newness.RECENT
    def test_established(self):
        assert compute_newness(_dt(200), is_first_ever=True) == Newness.ESTABLISHED
    def test_not_first_ever_always_established(self):
        assert compute_newness(_dt(0.5), is_first_ever=False) == Newness.ESTABLISHED
    def test_none_date_is_familiar(self):
        assert compute_newness(None, is_first_ever=True) == Newness.FAMILIAR

    def test_malformed_date_returns_established(self):
        # Exercises the except (ValueError, TypeError) branch at scorer.py:100
        assert compute_newness("not-a-date", is_first_ever=True) == Newness.ESTABLISHED

    def test_naive_datetime_string_works(self):
        # ISO string without timezone info — exercises the tzinfo=None branch at scorer.py:97
        naive_dt = (datetime.now() - timedelta(days=3)).isoformat()  # no +00:00
        assert compute_newness(naive_dt, is_first_ever=True) == Newness.HOT

class TestComputeScore:
    def test_failure(self):
        result = compute_score(success=False, is_first_ever=False,
                               rarity=Rarity.COMMON, newness=Newness.ESTABLISHED, streak=0)
        assert result.pts == -10
    def test_basic_success(self):
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED, streak=0)
        assert result.pts == 50
    def test_first_ever_bonus(self):
        result = compute_score(success=True, is_first_ever=True,
                               rarity=Rarity.COMMON, newness=Newness.ESTABLISHED, streak=0)
        assert result.pts == 150  # 50 + 100
    def test_rarity_multiplier_legendary(self):
        result = compute_score(success=True, is_first_ever=True,
                               rarity=Rarity.LEGENDARY, newness=Newness.ESTABLISHED, streak=0)
        assert result.pts == 600  # (50+100) * 4
    def test_zero_day_multiplier(self):
        result = compute_score(success=True, is_first_ever=True,
                               rarity=Rarity.LEGENDARY, newness=Newness.ZERO_DAY, streak=0)
        assert result.pts == 3000  # (50+100) * 4 * 5
    def test_streak_multiplier(self):
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED, streak=5)
        assert result.pts == 75  # 50 * 1.5
    def test_streak_capped_at_2x(self):
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED, streak=100)
        assert result.pts == 100  # 50 * 2.0 capped
    def test_mesh_mult_4chip(self):
        # mesh_mult = 1.0 + (4-1)*0.5 = 2.5; pts = 50 * 2.5 = 125
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
                               streak=0, mesh_chips=4)
        assert result.pts == 125
    def test_mesh_mult_galaxy(self):
        # mesh_mult = 1.0 + (32-1)*0.5 = 16.5; pts = 50 * 16.5 = 825
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
                               streak=0, mesh_chips=32)
        assert result.pts == 825
    def test_score_result_has_breakdown(self):
        result = compute_score(success=True, is_first_ever=True,
                               rarity=Rarity.RARE, newness=Newness.HOT, streak=3)
        assert "base" in result.breakdown
        assert "first_ever_bonus" in result.breakdown
        assert "rarity_mult" in result.breakdown
        assert "newness_mult" in result.breakdown
        assert "streak_mult" in result.breakdown
        assert "mesh_mult" in result.breakdown
