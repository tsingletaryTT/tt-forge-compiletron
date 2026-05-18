import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / ".github" / "scripts"))
from process_submission import parse_issue_body

FULL_BODY = """### Tenstorrent system

QB2

### Chips used for this run

4

### Total chips in system

4

### Firmware version

80.14.0.0

### Backend version

0.1.0

### tt-kmd version (optional)

1.29

### Bench JSON

{"model_id": "alexnet/pytorch", "backend": "forge"}
{"model_id": "mobilenetv2/pytorch", "backend": "forge"}

### Notes (optional)

Ran overnight, room temp 22C."""


def test_parse_all_fields():
    p = parse_issue_body(FULL_BODY)
    assert p["hardware_system"] == "QB2"
    assert p["chips_used"] == 4
    assert p["chips_in_system"] == 4
    assert p["firmware_version"] == "80.14.0.0"
    assert p["backend_version"] == "0.1.0"
    assert p["tt_kmd_version"] == "1.29"
    assert '{"model_id": "alexnet/pytorch"' in p["bench_json"]
    assert '{"model_id": "mobilenetv2/pytorch"' in p["bench_json"]
    assert p["notes"] == "Ran overnight, room temp 22C."


def test_parse_empty_optional_fields():
    body = FULL_BODY.replace("1.29", "_No response_").replace(
        "Ran overnight, room temp 22C.", "_No response_"
    )
    p = parse_issue_body(body)
    assert p["tt_kmd_version"] == ""
    assert p["notes"] == ""


def test_parse_chips_are_integers():
    p = parse_issue_body(FULL_BODY)
    assert isinstance(p["chips_used"], int)
    assert isinstance(p["chips_in_system"], int)
