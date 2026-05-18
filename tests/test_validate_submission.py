import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.validate_submission import validate, ValidationResult

GOOD_HW = {
    "hardware_system": "QB2",
    "chips_used": 4,
    "chips_in_system": 4,
    "firmware_version": "80.14.0.0",
    "backend_version": "0.1.0",
    "tt_kmd_version": "1.29",
    "submitter": "testuser",
    "submission_issue": 1,
}

GOOD_RECORD = {
    "model_id": "alexnet/pytorch",
    "backend": "forge",
    "compile_s": 2.98,
    "bench_passes": 5,
    "infer_p50_s": 0.169,
    "throughput_p50": 169.3,
    "throughput_unit": "ms/sample",
    "timestamp": "2026-05-18T10:35:03Z",
}

def _jsonl(records):
    return "\n".join(json.dumps(r) for r in records)

def test_valid_submission():
    result = validate(_jsonl([GOOD_RECORD]), GOOD_HW)
    assert result.valid
    assert not result.errors
    assert len(result.records) == 1

def test_enriched_with_hardware_fields():
    result = validate(_jsonl([GOOD_RECORD]), GOOD_HW)
    r = result.records[0]
    assert r["hardware_system"] == "QB2"
    assert r["chips_used"] == 4
    assert r["submitter"] == "testuser"
    assert r["submission_issue"] == 1

def test_missing_required_field():
    bad = {k: v for k, v in GOOD_RECORD.items() if k != "compile_s"}
    result = validate(_jsonl([bad]), GOOD_HW)
    assert not result.valid
    assert any("compile_s" in e for e in result.errors)

def test_bench_passes_too_low():
    bad = {**GOOD_RECORD, "bench_passes": 3}
    result = validate(_jsonl([bad]), GOOD_HW)
    assert not result.valid
    assert any("bench_passes" in e for e in result.errors)

def test_chips_used_exceeds_chips_in_system():
    hw = {**GOOD_HW, "chips_used": 8, "chips_in_system": 4}
    result = validate(_jsonl([GOOD_RECORD]), hw)
    assert not result.valid
    assert any("chips_used" in e for e in result.errors)

def test_invalid_hardware_system():
    hw = {**GOOD_HW, "hardware_system": "SuperBox"}
    result = validate(_jsonl([GOOD_RECORD]), hw)
    assert not result.valid
    assert any("hardware_system" in e for e in result.errors)

def test_invalid_throughput_unit():
    bad = {**GOOD_RECORD, "throughput_unit": "fps"}
    result = validate(_jsonl([bad]), GOOD_HW)
    assert not result.valid
    assert any("throughput_unit" in e for e in result.errors)

def test_empty_firmware_version():
    hw = {**GOOD_HW, "firmware_version": ""}
    result = validate(_jsonl([GOOD_RECORD]), hw)
    assert not result.valid
    assert any("firmware_version" in e for e in result.errors)

def test_invalid_json_line():
    result = validate("not json at all", GOOD_HW)
    assert not result.valid
    assert any("invalid JSON" in e for e in result.errors)

def test_multiple_records_all_valid():
    r2 = {**GOOD_RECORD, "model_id": "mobilenetv2/pytorch"}
    result = validate(_jsonl([GOOD_RECORD, r2]), GOOD_HW)
    assert result.valid
    assert len(result.records) == 2

def test_multiple_records_one_invalid():
    bad = {**GOOD_RECORD, "bench_passes": 2}
    result = validate(_jsonl([GOOD_RECORD, bad]), GOOD_HW)
    assert not result.valid
    assert any("Line 2" in e for e in result.errors)

def test_invalid_backend():
    bad = {**GOOD_RECORD, "backend": "tensorflow"}
    result = validate(_jsonl([bad]), GOOD_HW)
    assert not result.valid
    assert any("backend" in e for e in result.errors)
