# tests/test_bestiary_envfix.py
import json, tempfile, pathlib, pytest
from lib.expedition.bestiary import Bestiary


def _make_bestiary(tmp_path, failed: dict) -> Bestiary:
    """Write a minimal bestiary JSON and return a loaded Bestiary instance."""
    data = {"compiled": {}, "failed": failed, "chip_totals": {}}
    p = tmp_path / "bestiary.json"
    p.write_text(json.dumps(data))
    return Bestiary(path=p)


def test_clear_entries_matching_removes_matching(tmp_path):
    b = _make_bestiary(tmp_path, {
        "model/a": {"last_error": "FileNotFoundError: cats_image.jpeg", "attempts": 2},
        "model/b": {"last_error": "RuntimeError: segfault", "attempts": 1},
    })
    removed = b.clear_entries_matching(error_contains="cats_image.jpeg")
    assert removed == ["model/a"]
    assert "model/a" not in b.failed
    assert "model/b" in b.failed


def test_clear_entries_matching_leaves_no_match_untouched(tmp_path):
    b = _make_bestiary(tmp_path, {
        "model/c": {"last_error": "Something else entirely", "attempts": 1},
    })
    removed = b.clear_entries_matching(error_contains="cats_image.jpeg")
    assert removed == []
    assert "model/c" in b.failed
