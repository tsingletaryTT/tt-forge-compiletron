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


def test_record_failure_stores_env_fingerprint(tmp_path):
    b = _make_bestiary(tmp_path, {})
    fingerprint = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "0.36.2"}
    b.record_failure("model/x", run=1, error="ImportError: some version error",
                     env_fingerprint=fingerprint)
    assert b.failed["model/x"]["env_fingerprint"] == fingerprint


def test_clear_stale_env_failures_clears_on_version_change(tmp_path):
    b = _make_bestiary(tmp_path, {
        "model/hub_old": {
            "last_error": "ImportError: huggingface-hub>=0.30.0,<1.0 required but found 1.15.0",
            "error_category": "other",
            "attempts": 3,
            "env_fingerprint": {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "1.15.0"},
        },
        "model/segfault": {
            "last_error": "SIGSEGV: forge.compile() killed by signal 11",
            "error_category": "forge_internal",
            "attempts": 3,
            "env_fingerprint": {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "1.15.0"},
        },
        "model/no_fp": {
            "last_error": "ImportError: version mismatch >= something",
            "error_category": "other",
            "attempts": 2,
            # no env_fingerprint — should be left alone
        },
    })
    current = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "0.36.2"}
    cleared = b.clear_stale_env_failures(current)

    # hub_old: matches — version changed + category + version-signal in error
    assert "model/hub_old" not in b.failed
    assert "model/hub_old" in cleared

    # segfault: forge_internal not in eligible categories — must NOT be cleared
    assert "model/segfault" in b.failed

    # no_fp: no stored fingerprint — must NOT be cleared
    assert "model/no_fp" in b.failed


def test_clear_stale_env_failures_no_change_when_env_same(tmp_path):
    fp = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "0.36.2"}
    b = _make_bestiary(tmp_path, {
        "model/y": {
            "last_error": "ImportError: version >= 1.0 required",
            "error_category": "api_mismatch",
            "attempts": 2,
            "env_fingerprint": fp,
        },
    })
    cleared = b.clear_stale_env_failures(fp)  # same fingerprint
    assert "model/y" in b.failed
    assert cleared == []


def test_current_env_fingerprint_returns_dict():
    from lib.expedition.bestiary import _current_env_fingerprint
    fp = _current_env_fingerprint()
    assert isinstance(fp, dict)
    assert "torch" in fp
    assert "transformers" in fp
    assert "huggingface_hub" in fp


def test_clear_stale_env_failures_does_not_clear_non_version_other(tmp_path):
    """An 'other' entry with > in error text must NOT be cleared on env change."""
    old_fp = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "1.15.0"}
    new_fp = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "0.36.2"}
    b = _make_bestiary(tmp_path, {
        "model/dim_err": {
            "last_error": "RuntimeError: dimension 5 > max 3",
            "error_category": "other",
            "attempts": 2,
            "env_fingerprint": old_fp,
        },
    })
    cleared = b.clear_stale_env_failures(new_fp)
    assert cleared == []
    assert "model/dim_err" in b.failed


def test_warm_hf_datasets_is_importable():
    """_warm_hf_datasets must be importable without running forge or hardware."""
    from lib.expedition.expedition_worker import _warm_hf_datasets
    assert callable(_warm_hf_datasets)


def test_wrong_backend_not_in_perm_fail_cats():
    """wrong_backend must not be in _RUNTIME_PERM_FAIL_CATS so JAX models get retried."""
    import ast, pathlib
    src = pathlib.Path("lib/expedition/expedition_worker.py").read_text()
    # Find the set literal assigned to _RUNTIME_PERM_FAIL_CATS
    # Quick textual check is reliable enough here
    assert "wrong_backend" not in src.split("_RUNTIME_PERM_FAIL_CATS")[1].split("}")[0], \
        "wrong_backend must not appear inside _RUNTIME_PERM_FAIL_CATS"


def test_ird_preflight_symbols_exist():
    """_IRD_DEPENDENT_PREFIXES must exist and contain known IRD models."""
    from lib.expedition.expedition_worker import _IRD_DEPENDENT_PREFIXES
    assert "bevformer" in _IRD_DEPENDENT_PREFIXES
    assert "centernet" in _IRD_DEPENDENT_PREFIXES
    assert "yolov3" in _IRD_DEPENDENT_PREFIXES
    # All four spec-listed prefixes that were initially missing must be present.
    assert "bevdepth" in _IRD_DEPENDENT_PREFIXES
    assert "detr3d" in _IRD_DEPENDENT_PREFIXES
    assert "arnold" in _IRD_DEPENDENT_PREFIXES
    assert "fuyu" in _IRD_DEPENDENT_PREFIXES


def test_ird_preflight_guard_fires_without_env(monkeypatch):
    """Guard returns (True, error) for IRD model when IRD_LF_CACHE is unset."""
    import os
    from unittest.mock import MagicMock
    from lib.expedition.expedition_worker import _preflight_arch_check

    monkeypatch.delenv("IRD_LF_CACHE", raising=False)

    item = MagicMock()
    item.model_id = "bevformer/pytorch"
    item.is_frontier = False

    skip, reason = _preflight_arch_check(item)
    assert skip is True
    assert "missing_dependency" in reason
    assert "IRD_LF_CACHE" in reason


def test_ird_preflight_guard_passes_with_env(monkeypatch):
    """Guard does not fire for IRD model when IRD_LF_CACHE is set."""
    import os
    from unittest.mock import MagicMock
    from lib.expedition.expedition_worker import _preflight_arch_check

    monkeypatch.setenv("IRD_LF_CACHE", "http://ird.internal/cache")

    item = MagicMock()
    item.model_id = "bevformer/pytorch"
    item.is_frontier = False

    skip, reason = _preflight_arch_check(item)
    # With IRD_LF_CACHE set the IRD guard doesn't fire; seed (non-frontier) model
    # then hits the is_frontier gate and returns (False, "").
    assert skip is False
