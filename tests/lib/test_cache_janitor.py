import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from lib.expedition.cache_janitor import snapshot_preexisting, is_gold_star, maybe_evict
from lib.expedition.scorer import ScoreResult, Rarity, Newness


def _score(pts: int, rarity=Rarity.COMMON, first_ever=False) -> ScoreResult:
    return ScoreResult(
        pts=pts, is_first_ever=first_ever, rarity=rarity,
        newness=Newness.ESTABLISHED, streak_at_score=0,
    )


class TestSnapshotPreexisting:
    def test_returns_frozenset_of_repo_ids(self):
        mock_repo = MagicMock()
        mock_repo.repo_id = "openai-community/gpt2"
        mock_info = MagicMock()
        mock_info.repos = [mock_repo]
        with patch("lib.expedition.cache_janitor.scan_cache_dir", return_value=mock_info):
            result = snapshot_preexisting()
        assert result == frozenset({"openai-community/gpt2"})

    def test_returns_empty_on_scan_error(self):
        with patch("lib.expedition.cache_janitor.scan_cache_dir", side_effect=Exception("no cache")):
            result = snapshot_preexisting()
        assert result == frozenset()

    def test_returns_frozenset_type(self):
        mock_info = MagicMock()
        mock_info.repos = []
        with patch("lib.expedition.cache_janitor.scan_cache_dir", return_value=mock_info):
            result = snapshot_preexisting()
        assert isinstance(result, frozenset)


class TestIsGoldStar:
    def test_legendary_success_is_gold(self):
        assert is_gold_star(_score(400, rarity=Rarity.LEGENDARY)) is True

    def test_rare_success_is_gold(self):
        assert is_gold_star(_score(300, rarity=Rarity.RARE)) is True

    def test_first_ever_success_is_gold(self):
        assert is_gold_star(_score(250, rarity=Rarity.COMMON, first_ever=True)) is True

    def test_common_success_not_gold(self):
        assert is_gold_star(_score(100, rarity=Rarity.COMMON)) is False

    def test_uncommon_success_not_gold(self):
        assert is_gold_star(_score(150, rarity=Rarity.UNCOMMON)) is False

    def test_failure_never_gold(self):
        assert is_gold_star(_score(-10, rarity=Rarity.LEGENDARY, first_ever=True)) is False

    def test_zero_pts_not_gold(self):
        assert is_gold_star(_score(0, rarity=Rarity.RARE)) is False


class TestMaybeEvict:
    def test_skips_preexisting_model(self, tmp_path):
        preexisting = frozenset({"org/model"})
        evicted, freed = maybe_evict("org/model", _score(100), preexisting)
        assert evicted is False
        assert freed == 0

    def test_skips_if_no_hf_cache_entry(self, tmp_path):
        preexisting = frozenset()
        with patch("lib.expedition.cache_janitor._hf_repo_dir",
                   return_value=tmp_path / "nonexistent"):
            evicted, freed = maybe_evict("org/model", _score(100), preexisting)
        assert evicted is False
        assert freed == 0

    def test_evicts_successful_common_model(self, tmp_path):
        repo_dir = tmp_path / "models--org--model"
        repo_dir.mkdir()
        (repo_dir / "weights.bin").write_bytes(b"x" * 1024)

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir), \
             patch("lib.expedition.cache_janitor.scan_cache_dir") as mock_scan:
            mock_repo = MagicMock()
            mock_repo.repo_id = "org/model"
            mock_repo.size_on_disk = 1024
            mock_scan.return_value.repos = [mock_repo]

            evicted, freed = maybe_evict("org/model", _score(100, rarity=Rarity.COMMON),
                                         frozenset())
        assert evicted is True
        assert freed == 1024
        assert not repo_dir.exists()

    def test_preserves_gold_star_model(self, tmp_path):
        repo_dir = tmp_path / "models--org--bigmodel"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir):
            evicted, freed = maybe_evict(
                "org/bigmodel",
                _score(400, rarity=Rarity.LEGENDARY),
                frozenset(),
            )
        assert evicted is False
        assert repo_dir.exists()

    def test_keeps_failure_without_evict_failures_flag(self, tmp_path):
        repo_dir = tmp_path / "models--org--failmodel"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir):
            evicted, freed = maybe_evict(
                "org/failmodel", _score(-10), frozenset(), evict_failures=False
            )
        assert evicted is False
        assert repo_dir.exists()

    def test_evicts_failure_with_evict_failures_flag(self, tmp_path):
        repo_dir = tmp_path / "models--org--failmodel"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir), \
             patch("lib.expedition.cache_janitor.scan_cache_dir") as mock_scan:
            mock_scan.return_value.repos = []

            evicted, freed = maybe_evict(
                "org/failmodel", _score(-10), frozenset(), evict_failures=True
            )
        assert evicted is True
        assert not repo_dir.exists()
        assert freed == 0

    def test_survives_rmtree_error(self, tmp_path):
        repo_dir = tmp_path / "models--org--model"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir), \
             patch("lib.expedition.cache_janitor.scan_cache_dir") as mock_scan, \
             patch("lib.expedition.cache_janitor.shutil.rmtree", side_effect=OSError("permission denied")):
            mock_scan.return_value.repos = []
            evicted, freed = maybe_evict("org/model", _score(100), frozenset())
        assert evicted is False
        assert freed == 0

    def test_evicts_and_returns_zero_bytes_when_scan_raises(self, tmp_path):
        repo_dir = tmp_path / "models--org--model"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir), \
             patch("lib.expedition.cache_janitor.scan_cache_dir", side_effect=Exception("scan failed")):
            evicted, freed = maybe_evict("org/model", _score(100), frozenset())
        assert evicted is True
        assert freed == 0
        assert not repo_dir.exists()
