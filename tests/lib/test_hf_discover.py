# tests/expedition/test_hf_discover.py
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from lib.expedition.hf_discover import (
    discover_frontier, build_dynamic_loader, FrontierModel,
    _model_to_frontier, _parse_params_from_name,
)
from lib.expedition.scorer import Rarity, Newness


def _mock_model(model_id="org/model", pipeline_tag="text-generation",
                downloads=500_000, days_ago=60, likes=100):
    m = MagicMock()
    m.id = model_id
    m.pipeline_tag = pipeline_tag
    m.downloads = downloads
    m.likes = likes
    m.gated = None
    m.disabled = None
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
            likes=500,
            params_b=0.1,
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
            likes=0,
            params_b=0.0,
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
            likes=5634,
            params_b=1.54,
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


class TestParseParamsFromName:
    def test_simple_integer(self):
        assert _parse_params_from_name("org/Model-7B") == 7.0

    def test_decimal(self):
        assert _parse_params_from_name("org/Model-1.3B") == 1.3

    def test_largest_wins(self):
        # "64B" is the actual model size; "12B" is the active-param count
        assert _parse_params_from_name("Nemotron-3-Super-64B-A12B-Math-REAP-GGUF") == 64.0

    def test_lowercase_b(self):
        assert _parse_params_from_name("org/tiny-0.5b-model") == 0.5

    def test_no_match_returns_zero(self):
        assert _parse_params_from_name("org/bert-base-uncased") == 0.0

    def test_does_not_match_standalone_number(self):
        # "patch16" should not parse as 16B
        assert _parse_params_from_name("google/vit-base-patch16-224") == 0.0


class TestUnsupportedFormatFilter:
    def test_gguf_skipped(self):
        models = [_mock_model("org/Model-7B-Q4_K_M-GGUF", pipeline_tag="text-generation")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0

    def test_ggml_skipped(self):
        models = [_mock_model("org/llama-7b-GGML", pipeline_tag="text-generation")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0

    def test_exl2_skipped(self):
        models = [_mock_model("org/Model-EXL2", pipeline_tag="text-generation")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0

    def test_mlx_skipped(self):
        models = [_mock_model("org/Model-MLX", pipeline_tag="text-generation")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0

    def test_standard_safetensors_model_passes(self):
        models = [_mock_model("org/standard-7b-model", pipeline_tag="text-generation")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 1

    def test_case_insensitive(self):
        models = [_mock_model("org/Model-7B-q4-gguf", pipeline_tag="text-generation")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0


class TestGGUFSizeFilter:
    """GGUF models have no safetensors metadata — the name-based fallback must catch them."""

    def test_gguf_64b_filtered_at_8b_limit(self):
        models = [_mock_model("org/Nemotron-3-Super-64B-A12B-Math-REAP-GGUF",
                              pipeline_tag="text-generation", downloads=5_000)]
        models[0].safetensors = None
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(
                compiled_ids=set(), known_model_ids=set(), max_params_b=8.0
            )
        assert len(result) == 0

    def test_gguf_always_filtered_regardless_of_size(self):
        # GGUF models are rejected by the format filter before size is checked —
        # forge can't compile them, so size is irrelevant.
        models = [_mock_model("org/Llama-3-Groq-8B-Tool-Use-Q4_K_M-GGUF",
                              pipeline_tag="text-generation", downloads=5_000)]
        models[0].safetensors = None
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(
                compiled_ids=set(), known_model_ids=set(), max_params_b=8.0
            )
        assert len(result) == 0

    def test_name_params_stored_on_frontier_model(self):
        mock = _mock_model("org/Model-13B-Instruct", "text-generation", downloads=10_000)
        mock.safetensors = None
        result = _model_to_frontier(mock)
        assert result.params_b == 13.0
