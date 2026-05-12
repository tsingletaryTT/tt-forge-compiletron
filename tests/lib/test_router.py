# tests/lib/test_router.py
"""Unit tests for lib/expedition/router.py — backend routing decisions."""
import tempfile
from pathlib import Path

import pytest

from lib.expedition.bestiary import Bestiary
from lib.expedition.router import DispatchDecision, route_model


def _fresh_bestiary(tmp_path: Path) -> Bestiary:
    return Bestiary(path=tmp_path / "bestiary.json")


def _item(**kwargs) -> dict:
    base = {"model_id": "org/model", "library": "", "model_type": "", "mesh_chips": 1}
    base.update(kwargs)
    return base


# ── Backend selection ─────────────────────────────────────────────────────────

class TestBackendRouting:
    def test_jax_library_routes_xla(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(library="jax"), b)
        assert d.backend == "xla"
        assert d.reason == "jax-native"

    def test_flax_library_routes_xla(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(library="flax"), b)
        assert d.backend == "xla"
        assert d.reason == "jax-native"

    def test_default_routes_forge(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(library="pytorch"), b)
        assert d.backend == "forge"
        assert d.reason == "default"

    def test_empty_library_routes_forge(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(library=""), b)
        assert d.backend == "forge"

    def test_xla_affinity_model_type_routes_xla(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(library="pytorch", model_type="bert"), b)
        assert d.backend == "xla"
        assert d.reason == "arch-xla-affinity"

    def test_gpt2_affinity_routes_xla(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(library="pytorch", model_type="gpt2"), b)
        assert d.backend == "xla"
        assert d.reason == "arch-xla-affinity"

    def test_unknown_model_type_routes_forge(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(model_type="llama"), b)
        assert d.backend == "forge"

    def test_jax_library_beats_affinity(self, tmp_path):
        """jax library tag overrides arch-affinity and produces higher confidence."""
        b = _fresh_bestiary(tmp_path)
        jax = route_model(_item(library="jax", model_type="bert"), b)
        aff = route_model(_item(library="pytorch", model_type="bert"), b)
        assert jax.confidence > aff.confidence
        assert jax.reason == "jax-native"


# ── Failure history routing ───────────────────────────────────────────────────

class TestFailureHistoryRouting:
    def test_forge_fatal_two_attempts_routes_xla(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        b.failed["org/model"] = {"error_category": "forge_internal", "attempts": 2}
        d = route_model(_item(model_id="org/model", library="pytorch"), b)
        assert d.backend == "xla"
        assert d.reason == "forge-failure-history"

    def test_forge_fatal_one_attempt_does_not_reroute(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        b.failed["org/model"] = {"error_category": "forge_internal", "attempts": 1}
        d = route_model(_item(model_id="org/model", library="pytorch"), b)
        assert d.backend == "forge"

    def test_xla_fatal_two_attempts_routes_forge(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        b.failed["org/model"] = {"error_category": "xla_runtime_error", "attempts": 2}
        d = route_model(_item(model_id="org/model", library="jax"), b)
        # xla failure history overrides even jax library tag? No — jax is priority 1.
        # But for non-jax: xla failure history should redirect to forge.
        d2 = route_model(_item(model_id="org/model", library="pytorch", model_type="bert"), b)
        assert d2.backend == "forge"
        assert d2.reason == "xla-failure-history"

    def test_non_fatal_forge_error_does_not_reroute(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        b.failed["org/model"] = {"error_category": "oom", "attempts": 5}
        d = route_model(_item(model_id="org/model", library="pytorch"), b)
        assert d.backend == "forge"

    def test_no_failure_history_uses_default(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(model_id="never/seen"), b)
        assert d.backend == "forge"
        assert d.reason == "default"


# ── Chip count ────────────────────────────────────────────────────────────────

class TestChipCount:
    def test_mesh_chips_respected(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(mesh_chips=4), b)
        assert d.chips == 4

    def test_missing_mesh_chips_defaults_to_one(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        item = _item()
        del item["mesh_chips"]
        d = route_model(item, b)
        assert d.chips == 1

    def test_chips_capped_by_available(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(mesh_chips=4), b, available_chips={0, 1})
        assert d.chips == 2

    def test_chips_not_zero_when_available_is_empty(self, tmp_path):
        b = _fresh_bestiary(tmp_path)
        d = route_model(_item(mesh_chips=4), b, available_chips=set())
        assert d.chips >= 1
