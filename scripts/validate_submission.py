"""
Community bench submission validator.

Validates and enriches JSONL bench data from community submissions.
Importable by the GitHub Action entry point and runnable as a CLI.
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import dataclass, field

REQUIRED_FIELDS: dict[str, type | tuple] = {
    "model_id":        str,
    "backend":         str,
    "compile_s":       (int, float),
    "bench_passes":    int,
    "infer_p50_s":     (int, float),
    "throughput_p50":  (int, float),
    "throughput_unit": str,
    "timestamp":       str,
    "hardware_system": str,
    "chips_used":      int,
    "chips_in_system": int,
    "firmware_version": str,
    "backend_version": str,
}

VALID_HARDWARE  = {"N150", "N300", "QB", "QB2", "LoudBox", "custom"}
VALID_TPUT_UNITS = {"tokens/sec", "ms/sample"}
VALID_BACKENDS  = {"forge", "xla", "onnx"}


@dataclass
class ValidationResult:
    valid:   bool
    errors:  list[str] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)


def validate(jsonl_text: str, hardware: dict) -> ValidationResult:
    """Validate and enrich JSONL bench submission.

    hardware keys (required): hardware_system, chips_used, chips_in_system,
                              firmware_version, backend_version, submitter,
                              submission_issue
    hardware keys (optional): tt_kmd_version
    """
    errors: list[str] = []
    records: list[dict] = []

    lines = [l.strip() for l in jsonl_text.strip().splitlines() if l.strip()]
    if not lines:
        return ValidationResult(False, ["No JSON records found"])

    for i, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: invalid JSON: {e}")
            continue

        record.update(hardware)
        line_errors = _validate_record(record, i)
        errors.extend(line_errors)
        if not line_errors:
            records.append(record)

    valid = not errors
    return ValidationResult(valid, errors, records if valid else [])


def _validate_record(record: dict, line_num: int) -> list[str]:
    errors: list[str] = []
    p = f"Line {line_num}"

    for fname, expected in REQUIRED_FIELDS.items():
        if fname not in record:
            errors.append(f"{p}: missing required field '{fname}'")
            continue
        if not isinstance(record[fname], expected):
            exp_name = expected.__name__ if isinstance(expected, type) else "/".join(t.__name__ for t in expected)
            errors.append(f"{p}: '{fname}' must be {exp_name}, got {type(record[fname]).__name__}")

    if errors:
        return errors

    if record["bench_passes"] < 5:
        errors.append(f"{p}: bench_passes must be >= 5, got {record['bench_passes']}")
    if record["chips_used"] < 1:
        errors.append(f"{p}: chips_used must be >= 1, got {record['chips_used']}")
    if record["chips_in_system"] < 1:
        errors.append(f"{p}: chips_in_system must be >= 1, got {record['chips_in_system']}")
    for perf_field in ("compile_s", "infer_p50_s", "throughput_p50"):
        if record[perf_field] < 0:
            errors.append(f"{p}: {perf_field} must be >= 0, got {record[perf_field]}")
    if record["chips_used"] > record["chips_in_system"]:
        errors.append(f"{p}: chips_used ({record['chips_used']}) > chips_in_system ({record['chips_in_system']})")
    if record["hardware_system"] not in VALID_HARDWARE:
        errors.append(f"{p}: hardware_system must be one of {sorted(VALID_HARDWARE)}, got '{record['hardware_system']}'")
    if record["throughput_unit"] not in VALID_TPUT_UNITS:
        errors.append(f"{p}: throughput_unit must be one of {sorted(VALID_TPUT_UNITS)}, got '{record['throughput_unit']}'")
    if record["backend"] not in VALID_BACKENDS:
        errors.append(f"{p}: backend must be one of {sorted(VALID_BACKENDS)}, got '{record['backend']}'")
    if not record["firmware_version"].strip():
        errors.append(f"{p}: firmware_version must not be empty")
    if not record["backend_version"].strip():
        errors.append(f"{p}: backend_version must not be empty")

    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a community bench submission")
    ap.add_argument("jsonl_file", help="Path to JSONL file, or '-' to read stdin")
    ap.add_argument("--system",            required=True, choices=sorted(VALID_HARDWARE))
    ap.add_argument("--chips-used",        required=True, type=int)
    ap.add_argument("--chips-in-system",   required=True, type=int)
    ap.add_argument("--firmware-version",  required=True)
    ap.add_argument("--backend-version",   required=True)
    ap.add_argument("--tt-kmd-version",    default="")
    ap.add_argument("--submitter",         default="local-test")
    ap.add_argument("--issue",             default=0, type=int)
    args = ap.parse_args()

    if args.jsonl_file == "-":
        text = sys.stdin.read()
    else:
        try:
            with open(args.jsonl_file) as f:
                text = f.read()
        except FileNotFoundError:
            print(f"✗ File not found: {args.jsonl_file}", file=sys.stderr)
            sys.exit(1)

    hardware = {
        "hardware_system":  args.system,
        "chips_used":       args.chips_used,
        "chips_in_system":  args.chips_in_system,
        "firmware_version": args.firmware_version,
        "backend_version":  args.backend_version,
        "tt_kmd_version":   args.tt_kmd_version,
        "submitter":        args.submitter,
        "submission_issue": args.issue,
    }

    result = validate(text, hardware)
    if result.valid:
        print(f"✓ Valid — {len(result.records)} record(s)")
        for r in result.records:
            print(f"  {r['model_id']} · {r['throughput_p50']} {r['throughput_unit']}")
        sys.exit(0)
    else:
        print(f"✗ Invalid — {len(result.errors)} error(s):")
        for e in result.errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
