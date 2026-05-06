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
