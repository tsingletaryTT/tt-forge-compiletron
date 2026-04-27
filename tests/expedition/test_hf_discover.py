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

    def test_loader_has_input_type_and_model_id(self):
        model = FrontierModel(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            downloads=10_000_000,
            created_at=None,
            rarity=Rarity.LEGENDARY,
            newness=Newness.ESTABLISHED,
        )
        loader = build_dynamic_loader(model)
        assert loader._input_type == "audio"
        assert loader._model_id == "openai/whisper-large-v3"


class TestDiscoverFrontierAPIFallback:
    def test_returns_empty_when_api_raises(self):
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.side_effect = Exception("network error")
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert result == []

    def test_returns_empty_when_hfapi_none(self):
        import lib.expedition.hf_discover as mod
        original = mod.HfApi
        try:
            mod.HfApi = None
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
            assert result == []
        finally:
            mod.HfApi = original


class TestMeshChipsDetection:
    def test_deepseek_model_gets_mesh_4(self):
        mock = _mock_model("deepseek-ai/DeepSeek-R1", "text-generation", downloads=1_000_000, days_ago=30)
        mock.safetensors = None
        result = _model_to_frontier(mock)
        assert result.mesh_chips == 4

    def test_regular_model_gets_mesh_1(self):
        mock = _mock_model("google/bert-base", "fill-mask", downloads=5_000_000, days_ago=200)
        mock.safetensors = None
        result = _model_to_frontier(mock)
        assert result.mesh_chips == 1

    def test_large_param_count_gets_mesh_4(self):
        mock = _mock_model("org/some-giant-model", "text-generation", downloads=1_000, days_ago=5)
        # Simulate safetensors with 45B params
        mock.safetensors = MagicMock()
        mock.safetensors.total = int(45e9)
        result = _model_to_frontier(mock)
        assert result.mesh_chips == 4
