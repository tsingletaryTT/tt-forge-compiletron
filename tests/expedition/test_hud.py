# tests/expedition/test_hud.py
import os
import pytest
from lib.expedition.scorer import ScoreResult, Rarity, Newness
from lib.expedition.hud import ChipHUD, ChipState


def _score(pts: int, first_ever: bool = False) -> ScoreResult:
    return ScoreResult(
        pts=pts, is_first_ever=first_ever,
        rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
        streak_at_score=0,
    )


@pytest.fixture
def hud(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPEDITION_STATUS_DIR", str(tmp_path))
    return ChipHUD(chip_id=0, total_models=10)


class TestChipHUD:
    def test_initial_state(self, hud):
        assert hud.state.pts == 0
        assert hud.state.streak == 0
        assert hud.state.successes == 0
        assert hud.state.failures == 0

    def test_record_success_increments_pts(self, hud):
        hud.record_success("bert/qa", _score(150, first_ever=True))
        assert hud.state.pts == 150
        assert hud.state.successes == 1

    def test_record_success_increments_streak(self, hud):
        hud.record_success("bert/qa", _score(50))
        hud.record_success("resnet50", _score(50))
        assert hud.state.streak == 2

    def test_record_failure_resets_streak(self, hud):
        hud.record_success("bert/qa", _score(50))
        hud.record_success("resnet50", _score(50))
        hud.record_failure("mistral-7b")
        assert hud.state.streak == 0
        assert hud.state.failures == 1

    def test_best_streak_preserved_after_reset(self, hud):
        hud.record_success("a", _score(50))
        hud.record_success("b", _score(50))
        hud.record_success("c", _score(50))
        hud.record_failure("d")
        assert hud.state.best_streak == 3

    def test_failure_deducts_pts(self, hud):
        hud.record_success("bert/qa", _score(100))
        hud.record_failure("mistral")
        assert hud.state.pts == 90

    def test_set_current(self, hud):
        hud.set_current("bert/qa", index=3)
        assert hud.state.current_model == "bert/qa"
        assert hud.state.current_index == 3

    def test_write_status_creates_file(self, hud, tmp_path):
        hud.set_current("bert/qa", index=1)
        hud.record_success("bert/qa", _score(50))
        hud.write_status()
        status_file = tmp_path / "expedition_chip_0.status"
        assert status_file.exists()
        content = status_file.read_text()
        assert "pts=50" in content
        assert "chip_id=0" in content

    def test_write_status_includes_streak(self, hud, tmp_path):
        hud.record_success("a", _score(50))
        hud.record_success("b", _score(50))
        hud.write_status()
        content = (tmp_path / "expedition_chip_0.status").read_text()
        assert "streak=2" in content

    def test_write_status_includes_done_flag(self, hud, tmp_path):
        hud.mark_done()
        hud.write_status()
        content = (tmp_path / "expedition_chip_0.status").read_text()
        assert "done=1" in content
