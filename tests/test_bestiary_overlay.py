# tests/test_bestiary_overlay.py
#
# Tests for the pip_deps / missing_packages fields and missing_dep_report()
# method added to Bestiary as part of the isolated-overlay-deps feature.
import json, pathlib, pytest
from lib.expedition.bestiary import Bestiary


def _make_bestiary(tmp_path, compiled=None, failed=None) -> Bestiary:
    data = {
        "compiled": compiled or {},
        "failed": failed or {},
        "chip_totals": {},
    }
    p = tmp_path / "bestiary.json"
    p.write_text(json.dumps(data))
    return Bestiary(path=p)


def test_record_success_stores_pip_deps(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success(
        model_id="gliner/pytorch", chip=0, run=1, time_s=10.0,
        task="token-classification", source="seed", rarity="uncommon",
        hf_downloads=0, hf_created_at="", artifact="", backend="forge",
        pip_deps=["gliner"],
    )
    assert b.compiled["gliner/pytorch"]["pip_deps"] == ["gliner"]


def test_record_success_omits_pip_deps_when_empty(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success(
        model_id="alexnet/pytorch", chip=0, run=1, time_s=5.0,
        task="image-classification", source="seed", rarity="common",
        hf_downloads=0, hf_created_at="", artifact="", backend="forge",
    )
    assert "pip_deps" not in b.compiled["alexnet/pytorch"]


def test_record_failure_stores_missing_packages(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_failure("surya/pytorch", run=1, error="No module named 'surya'",
                     missing_packages=["surya-ocr"])
    assert b.failed["surya/pytorch"]["missing_packages"] == ["surya-ocr"]


def test_record_failure_merges_missing_packages(tmp_path):
    """Second failure accumulates packages rather than overwriting."""
    b = _make_bestiary(tmp_path, failed={
        "model/x": {
            "run_first_failed": 1,
            "attempts": 1,
            "last_error": "No module named 'alpha'",
            "error_category": "missing_dependency",
            "missing_packages": ["alpha"],
        }
    })
    b.record_failure("model/x", run=2, error="No module named 'beta'",
                     missing_packages=["beta"])
    assert set(b.failed["model/x"]["missing_packages"]) == {"alpha", "beta"}


def test_record_failure_omits_missing_packages_when_empty(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_failure("model/y", run=1, error="SIGSEGV")
    assert "missing_packages" not in b.failed["model/y"]


def test_missing_dep_report_ranks_by_count(tmp_path):
    b = _make_bestiary(tmp_path, failed={
        "model/a": {"last_error": "", "error_category": "missing_dependency",
                    "attempts": 1, "missing_packages": ["surya-ocr"]},
        "model/b": {"last_error": "", "error_category": "missing_dependency",
                    "attempts": 1, "missing_packages": ["surya-ocr", "torchaudio"]},
        "model/c": {"last_error": "", "error_category": "missing_dependency",
                    "attempts": 1, "missing_packages": ["torchaudio"]},
        "model/d": {"last_error": "", "error_category": "missing_dependency",
                    "attempts": 1, "missing_packages": ["gliner"]},
    })
    report = b.missing_dep_report()
    assert report[0]["package"] == "surya-ocr"
    assert report[0]["count"] == 2
    assert report[1]["package"] == "torchaudio"
    assert report[1]["count"] == 2
    assert report[2]["package"] == "gliner"
    assert report[2]["count"] == 1
    assert "models" in report[0]
    assert "model/a" in report[0]["models"]


def test_missing_dep_report_empty_bestiary(tmp_path):
    b = _make_bestiary(tmp_path)
    assert b.missing_dep_report() == []
