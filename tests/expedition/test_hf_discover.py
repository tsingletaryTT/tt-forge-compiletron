# tests/expedition/test_hf_discover.py
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from lib.expedition.hf_discover import (
    discover_frontier, build_dynamic_loader, FrontierModel,
    _model_to_frontier,
)
from lib.expedition.scorer import Rarity, Newness


def _mock_model(model_id="org/model", pipeline_tag="text-generation",
                downloads=500_000, days_ago=60):
    m = MagicMock()
    m.id = model_id
    m.pipeline_tag = pipeline_tag
    m.downloads = downloads
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    m.created_at = created
    return m


class TestModelToFrontier:
    def test_basic_conversion(self):
        mock = _mock_model("org/bert", "text-classification", downloads=200_000, days_ago=45)
        result = _model_to_frontier(mock)
        assert result.model_id == "org/bert"
        assert result.pipeline_tag == "text-classification"
        assert result.rarity == Rarity.UNCOMMON
        assert result.newness == Newness.RECENT

    def test_legendary_rarity(self):
        mock = _mock_model(downloads=20_000_000)
        result = _model_to_frontier(mock)
        assert result.rarity == Rarity.LEGENDARY

    def test_zero_day_newness(self):
        mock = _mock_model(days_ago=0.25)
        result = _model_to_frontier(mock)
        assert result.newness == Newness.ZERO_DAY


class TestDiscoverFrontier:
    def test_filters_already_compiled(self):
        models = [
            _mock_model("org/bert"),
            _mock_model("org/gpt2"),
        ]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(
                compiled_ids={"org/bert"},
                known_model_ids=set(),
            )
        ids = [m.model_id for m in result]
        assert "org/bert" not in ids
        assert "org/gpt2" in ids

    def test_filters_known_forge_models(self):
        models = [_mock_model("org/resnet50")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(
                compiled_ids=set(),
                known_model_ids={"org/resnet50"},
            )
        assert len(result) == 0

    def test_skips_unsupported_pipeline_tag(self):
        models = [_mock_model("org/weird", pipeline_tag="reinforcement-learning")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0

    def test_skips_none_pipeline_tag(self):
        models = [_mock_model("org/notag", pipeline_tag=None)]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0


class TestBuildDynamicLoader:
    def test_returns_callable_for_text_generation(self):
        model = FrontierModel(
            model_id="gpt2",
            pipeline_tag="text-generation",
            downloads=1_000_000,
            created_at=None,
            rarity=Rarity.RARE,
            newness=Newness.ESTABLISHED,
        )
        loader = build_dynamic_loader(model)
        assert loader is not None
        assert callable(loader)

    def test_returns_none_for_unsupported_tag(self):
        model = FrontierModel(
            model_id="org/rl",
            pipeline_tag="reinforcement-learning",
            downloads=100,
            created_at=None,
            rarity=Rarity.COMMON,
            newness=Newness.ESTABLISHED,
        )
        result = build_dynamic_loader(model)
        assert result is None
