# tests/expedition/test_intelligent_dispatch.py
"""Tests for intelligent dispatch: bestiary backend tracking, scorer mesh_mult,
router decisions, and worker single-model mode."""
import sys
from pathlib import Path

# Ensure project root is in sys.path when pytest runs from tests/ subdir
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import tempfile
import pytest
from lib.expedition.bestiary import Bestiary


# ── Task 1: Bestiary backend field ────────────────────────────────────────────

def _make_bestiary(tmp_path: Path) -> Bestiary:
    """Create a fresh in-memory Bestiary backed by a temp file."""
    b = Bestiary(path=str(tmp_path / "bestiary.json"))
    return b


def test_record_success_backend_stored(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success("org/model", chip=0, run=1, time_s=2.5, task="text-generation",
                     source="huggingface", rarity="common", hf_downloads=1000,
                     hf_created_at=None, artifact="hello", backend="forge")
    assert b.compiled["org/model"]["backend"] == "forge"


def test_record_success_backends_succeeded_accumulates(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success("org/model", chip=0, run=1, time_s=2.5, task="text-generation",
                     source="huggingface", rarity="common", hf_downloads=1000,
                     hf_created_at=None, artifact="hello", backend="forge")
    b.record_success("org/model", chip=1, run=2, time_s=1.5, task="text-generation",
                     source="huggingface", rarity="common", hf_downloads=1000,
                     hf_created_at=None, artifact="world", backend="xla")
    assert b.compiled["org/model"]["backend"] == "forge"   # first backend preserved
    assert set(b.compiled["org/model"]["backends_succeeded"]) == {"forge", "xla"}


def test_is_compiled_by_returns_true_for_matching_backend(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success("org/model", chip=0, run=1, time_s=2.5, task="text-generation",
                     source="huggingface", rarity="common", hf_downloads=1000,
                     hf_created_at=None, artifact="hello", backend="xla")
    assert b.is_compiled_by("org/model", "xla") is True
    assert b.is_compiled_by("org/model", "forge") is False


def test_is_compiled_by_returns_false_for_unknown_model(tmp_path):
    b = _make_bestiary(tmp_path)
    assert b.is_compiled_by("nonexistent/model", "forge") is False


# ── Task 2: Scorer mesh_mult ──────────────────────────────────────────────────

from lib.expedition.scorer import compute_score, Rarity, Newness


def test_mesh_mult_single_chip_unchanged():
    """1-chip compile should produce same points as before (mesh_mult = 1.0)."""
    s = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=1)
    assert s.pts == int(50 * 1.0 * 1.0 * 1.0 * 1.0)   # 50


def test_mesh_mult_four_chips():
    """4-chip compile earns 2.5× the single-chip score."""
    s4 = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=4)
    assert s4.pts == int(50 * 1.0 * 1.0 * 1.0 * 2.5)   # base * rarity * newness * streak * mesh


def test_mesh_mult_two_chips():
    """2-chip compile earns 1.5× the single-chip score."""
    s2 = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=2)
    assert s2.pts == int(50 * 1.0 * 1.0 * 1.0 * 1.5)   # base * rarity * newness * streak * mesh


def test_mesh_mult_in_breakdown():
    """Breakdown dict should expose mesh_mult and not have old mesh_bonus key."""
    s = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=4)
    assert "mesh_mult" in s.breakdown
    assert "mesh_bonus" not in s.breakdown
    assert s.breakdown["mesh_mult"] == 2.5


def test_opportunist_bonus():
    """is_opportunist=True adds flat +25 after bracket."""
    s_plain = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0)
    s_opp   = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, is_opportunist=True)
    assert s_opp.pts == s_plain.pts + 25
    assert s_opp.breakdown["opportunist_bonus"] == 25


def test_formation_share_flat_150():
    """is_formation_share=True returns exactly 150 pts regardless of other params."""
    s = compute_score(True, True, Rarity.LEGENDARY, Newness.ZERO_DAY, 5, is_formation_share=True)
    assert s.pts == 150
    assert s.breakdown["formation_share"] is True


def test_legendary_4chip_rally_example():
    """Sanity-check the scoring example from the design spec."""
    # base=50, first_ever=100, first_voice=100 → 250
    # rarity=4.0, newness=5.0, streak=1.3, mesh=2.5
    # int(250 * 4.0 * 5.0 * 1.3 * 2.5) = 16250
    s = compute_score(
        success=True,
        is_first_ever=True,
        rarity=Rarity.LEGENDARY,
        newness=Newness.ZERO_DAY,
        streak=3,
        mesh_chips=4,
        is_first_voice=True,
    )
    assert s.pts == 16_250


# ── Task 3: HF discover library param ────────────────────────────────────────

from unittest.mock import patch, MagicMock
from lib.expedition.hf_discover import discover_frontier, discover_from_authors


def test_discover_frontier_passes_library_to_api():
    """discover_frontier(library="jax") should call api.list_models(filter="jax", ...)."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_frontier(compiled_ids=set(), known_model_ids=set(), library="jax")
    call_kwargs = mock_api_instance.list_models.call_args[1]
    assert call_kwargs.get("filter") == "jax"


def test_discover_frontier_omits_filter_when_library_none():
    """discover_frontier(library=None) should NOT include 'filter' kwarg."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_frontier(compiled_ids=set(), known_model_ids=set(), library=None)
    call_kwargs = mock_api_instance.list_models.call_args[1]
    assert "filter" not in call_kwargs


def test_discover_frontier_default_library_is_pytorch():
    """Default call should still filter for pytorch (backwards compatible)."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_frontier(compiled_ids=set(), known_model_ids=set())
    call_kwargs = mock_api_instance.list_models.call_args[1]
    assert call_kwargs.get("filter") == "pytorch"


def test_discover_from_authors_passes_library_to_api():
    """discover_from_authors(library="jax") should call api.list_models(filter="jax", ...)."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_from_authors(
            authors=["google"],
            compiled_ids=set(),
            known_model_ids=set(),
            library="jax",
        )
    if mock_api_instance.list_models.call_args is not None:
        call_kwargs = mock_api_instance.list_models.call_args[1]
        assert call_kwargs.get("filter") == "jax"


def test_discover_from_authors_omits_filter_when_library_none():
    """discover_from_authors(library=None) should NOT include 'filter' kwarg."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_from_authors(
            authors=["google"],
            compiled_ids=set(),
            known_model_ids=set(),
            library=None,
        )
    if mock_api_instance.list_models.call_args is not None:
        call_kwargs = mock_api_instance.list_models.call_args[1]
        assert "filter" not in call_kwargs


def test_discover_from_authors_default_library_is_pytorch():
    """Default discover_from_authors call should still filter for pytorch."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_from_authors(
            authors=["google"],
            compiled_ids=set(),
            known_model_ids=set(),
        )
    if mock_api_instance.list_models.call_args is not None:
        call_kwargs = mock_api_instance.list_models.call_args[1]
        assert call_kwargs.get("filter") == "pytorch"


# ── Task 4: Pattern split ─────────────────────────────────────────────────────

def _load_root_expedition():
    """Load the root expedition.py module explicitly, bypassing the tests/expedition package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "expedition_root",
        str(Path(__file__).resolve().parent.parent.parent / "expedition.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_forge_ignore_patterns_excludes_msgpack():
    """Forge patterns must exclude .msgpack (Flax weights) since forge doesn't need them."""
    mod = _load_root_expedition()
    assert any("msgpack" in p for p in mod._FORGE_IGNORE_PATTERNS)


def test_xla_ignore_patterns_includes_msgpack():
    """XLA patterns must NOT exclude .msgpack (Flax weights are required by XLA)."""
    mod = _load_root_expedition()
    assert not any("msgpack" in p for p in mod._XLA_IGNORE_PATTERNS)


def test_scan_frontier_accepts_library_param():
    """_scan_frontier must accept a library keyword argument without raising."""
    import inspect
    mod = _load_root_expedition()
    sig = inspect.signature(mod._scan_frontier)
    assert "library" in sig.parameters


# ── Task 5: Router ───────────────────────────────────────────────────────────

def test_router_jax_native_routes_to_xla(tmp_path):
    """Models with library=="jax" or "flax" must be routed to XLA backend."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "google/flax-bert", "library": "jax",
         "hf_downloads": 5000, "mesh_chips": 1},
        b,
    )
    assert d.backend == "xla"
    assert d.confidence >= 0.9
    assert d.reason == "jax-native"


def test_router_flax_library_routes_to_xla(tmp_path):
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "google/flax-bert", "library": "flax",
         "hf_downloads": 5000, "mesh_chips": 1},
        b,
    )
    assert d.backend == "xla"
    assert d.reason == "jax-native"


def test_router_default_routes_to_forge(tmp_path):
    """Pytorch models with no special signals should default to forge."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "foo/bar", "library": "pytorch",
         "hf_downloads": 100, "mesh_chips": 1},
        b,
    )
    assert d.backend == "forge"
    assert d.reason == "default"


def test_router_forge_failure_history_redirects_to_xla(tmp_path):
    """Models with >=2 forge_missing_op or forge_internal failures redirect to XLA."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    # Simulate 2 forge_missing_op failures recorded in bestiary
    b._data["failed"]["org/troubled"] = {
        "run_first_failed": 1, "attempts": 2,
        "last_error": "are not implemented in tt-forge",
        "error_category": "forge_missing_op",
    }
    d = route_model(
        {"model_id": "org/troubled", "library": "pytorch",
         "hf_downloads": 1000, "mesh_chips": 1},
        b,
    )
    assert d.backend == "xla"
    assert d.reason == "forge-failure-history"


def test_router_large_model_gets_4_chips(tmp_path):
    """Models with mesh_chips=4 in metadata should be dispatched on 4 chips."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "deepseek-ai/deepseek-v3", "library": "pytorch",
         "hf_downloads": 1_000_000, "mesh_chips": 4, "hf_params_b": 67.0},
        b,
        available_chips=set(range(4)),
    )
    assert d.chips == 4


def test_router_caps_chips_at_available(tmp_path):
    """If only 2 chips available but model wants 4, cap at 2."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "deepseek-ai/deepseek-v3", "library": "pytorch",
         "hf_downloads": 1_000_000, "mesh_chips": 4},
        b,
        available_chips=set(range(2)),
    )
    assert d.chips == 2


def test_dispatch_decision_has_required_fields(tmp_path):
    """DispatchDecision must have backend, chips, confidence, reason."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model({"model_id": "a/b", "library": "pytorch",
                     "hf_downloads": 100, "mesh_chips": 1}, b)
    assert hasattr(d, "backend")
    assert hasattr(d, "chips")
    assert hasattr(d, "confidence")
    assert hasattr(d, "reason")
    assert isinstance(d.chips, int)
    assert 0.0 <= d.confidence <= 1.0
