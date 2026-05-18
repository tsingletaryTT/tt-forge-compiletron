# Community Bench Submissions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let players at home submit verified 5-pass bench results from their Tenstorrent hardware via a GitHub Issue form, with a validation Action that auto-creates a PR for maintainer review, and a website community section that renders cross-hardware comparisons.

**Architecture:** Issue form (YAML) → GitHub Action parses body, calls validator, creates PR in `data/community/`; website JS fetches community files from GitHub raw API and renders a use-case → model → hardware table alongside the maintainer baseline.

**Tech Stack:** Python 3.11 (validator, Action entry point), GitHub Actions YAML, GitHub Issue Forms YAML, vanilla JS (no build step), pytest.

**Repo:** `tsingletaryTT/tt-forge-compiletron`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/validate_submission.py` | Create | Schema validation + CLI; importable by Action and tests |
| `tests/test_validate_submission.py` | Create | Unit tests for validator |
| `.github/ISSUE_TEMPLATE/bench-submission.yml` | Create | Structured issue form with dropdowns |
| `.github/scripts/process_submission.py` | Create | Action entry point: parse issue body → validate → create PR |
| `tests/test_process_submission.py` | Create | Unit tests for issue body parser |
| `.github/workflows/community-submission.yml` | Create | Workflow: trigger on bench-submission issues |
| `data/community/.gitkeep` | Create | Ensure directory exists in git |
| `docs/index.html` | Modify | Add `#community` section with JS fetching |

---

## Task 1: Validator — `scripts/validate_submission.py`

**Files:**
- Create: `scripts/validate_submission.py`
- Test: `tests/test_validate_submission.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_validate_submission.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_validate_submission.py -v
```
Expected: `ModuleNotFoundError: No module named 'scripts.validate_submission'`

- [ ] **Step 3: Implement the validator**

Create `scripts/validate_submission.py`:

```python
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

    text = sys.stdin.read() if args.jsonl_file == "-" else open(args.jsonl_file).read()

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_validate_submission.py -v
```
Expected: all 13 tests pass.

- [ ] **Step 5: Smoke-test the CLI**

```bash
echo '{"model_id":"alexnet/pytorch","backend":"forge","compile_s":2.98,"bench_passes":5,"infer_p50_s":0.169,"throughput_p50":169.3,"throughput_unit":"ms/sample","timestamp":"2026-05-18T10:35:03Z"}' \
  | python3 scripts/validate_submission.py - \
    --system QB2 --chips-used 4 --chips-in-system 4 \
    --firmware-version "80.14.0.0" --backend-version "0.1.0"
```
Expected output:
```
✓ Valid — 1 record(s)
  alexnet/pytorch · 169.3 ms/sample
```

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_submission.py tests/test_validate_submission.py
git commit -m "feat: add community submission validator"
```

---

## Task 2: Issue Template + Community Directory

**Files:**
- Create: `.github/ISSUE_TEMPLATE/bench-submission.yml`
- Create: `data/community/.gitkeep`

No tests needed — GitHub validates the template schema at CI time; we'll verify locally by checking YAML syntax.

- [ ] **Step 1: Create community data directory**

```bash
mkdir -p data/community
touch data/community/.gitkeep
```

- [ ] **Step 2: Create the issue template**

Create `.github/ISSUE_TEMPLATE/bench-submission.yml`:

```yaml
name: "Bench Submission"
description: "Submit your verified 5-pass bench results from Tenstorrent hardware"
title: "[Bench] <model> · <system> · <N>-chip"
labels: ["bench-submission"]
body:
  - type: markdown
    attributes:
      value: |
        Submit results from a `--bench-passes 5` run. The Action will validate
        your JSON and open a PR for maintainer review. Results appear on the
        [community benchmarks table](https://tsingletarytt.github.io/tt-forge-compiletron/#community)
        once merged.

  - type: dropdown
    id: hardware_system
    attributes:
      label: "Tenstorrent system"
      options: ["N150", "N300", "QB", "QB2", "LoudBox", "custom"]
    validations:
      required: true

  - type: dropdown
    id: chips_used
    attributes:
      label: "Chips used for this run"
      description: "Number of chips active during the bench run (e.g. 1 for single-chip, 4 for RALLY)"
      options: ["1", "2", "4", "8", "16", "32"]
    validations:
      required: true

  - type: dropdown
    id: chips_in_system
    attributes:
      label: "Total chips in system"
      description: "Total chips installed in the box or cluster"
      options: ["1", "2", "4", "8", "16", "32"]
    validations:
      required: true

  - type: input
    id: firmware_version
    attributes:
      label: "Firmware version"
      description: "Run: `tt-smi -s | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d['device_info'][0]['firmware_version'])\"`"
      placeholder: "80.14.0.0"
    validations:
      required: true

  - type: input
    id: backend_version
    attributes:
      label: "Backend version"
      description: |
        forge → `pip show tt-forge | grep Version`
        xla   → `pip show pjrt-plugin-tt | grep Version`
        onnx  → `pip show tt-forge-onnx | grep Version`
      placeholder: "0.1.0"
    validations:
      required: true

  - type: input
    id: tt_kmd_version
    attributes:
      label: "tt-kmd version (optional)"
      description: "`modinfo tenstorrent | grep ^version`"
      placeholder: "1.29"
    validations:
      required: false

  - type: textarea
    id: bench_json
    attributes:
      label: "Bench JSON"
      description: |
        Paste your `perf_history.jsonl` lines from a `--bench-passes 5` run,
        or attach the file (rename to `.txt` if GitHub rejects `.jsonl`).
        One JSON object per line.

        Generate with:
        ```
        bash scripts/record_demo.sh --curated --bench-passes 5
        # then copy the relevant lines from data/perf_history.jsonl
        ```
      placeholder: '{"model_id": "alexnet/pytorch", "backend": "forge", "compile_s": 2.98, "bench_passes": 5, ...}'
    validations:
      required: true

  - type: textarea
    id: notes
    attributes:
      label: "Notes (optional)"
      description: "Anything unusual: cooling setup, ambient temperature, other workloads running, firmware flashed mid-run, etc."
    validations:
      required: false
```

- [ ] **Step 3: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/ISSUE_TEMPLATE/bench-submission.yml'))" && echo "YAML OK"
```
Expected: `YAML OK`

- [ ] **Step 4: Commit**

```bash
git add .github/ISSUE_TEMPLATE/bench-submission.yml data/community/.gitkeep
git commit -m "feat: add bench-submission issue template and community data directory"
```

---

## Task 3: GitHub Action + Issue Parser

**Files:**
- Create: `.github/scripts/process_submission.py`
- Create: `.github/workflows/community-submission.yml`
- Test: `tests/test_process_submission.py`

- [ ] **Step 1: Write failing tests for the issue body parser**

Create `tests/test_process_submission.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_process_submission.py -v
```
Expected: `ModuleNotFoundError: No module named 'process_submission'`

- [ ] **Step 3: Create the Action entry point**

Create `.github/scripts/process_submission.py`:

```python
"""
GitHub Actions entry point for community bench submission intake.

Reads env vars:  ISSUE_BODY, ISSUE_NUMBER, SUBMITTER
Writes outputs:  valid, errors, pr_url, model_count (via $GITHUB_OUTPUT)
Side effects:    creates enriched JSONL file, git branch, PR (when valid)
"""
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.validate_submission import validate


def parse_issue_body(body: str) -> dict:
    """Extract form fields from a GitHub issue form body.

    GitHub issue forms render as markdown with '### Section' headers.
    Each section's content is everything until the next header.
    '_No response_' means the optional field was left blank.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in body.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    def get(key: str, default: str = "") -> str:
        lines = sections.get(key, [])
        content = "\n".join(lines).strip()
        return default if content in ("", "_No response_") else content

    return {
        "hardware_system":  get("Tenstorrent system"),
        "chips_used":       int(get("Chips used for this run", "0") or "0"),
        "chips_in_system":  int(get("Total chips in system", "0") or "0"),
        "firmware_version": get("Firmware version"),
        "backend_version":  get("Backend version"),
        "tt_kmd_version":   get("tt-kmd version (optional)"),
        "bench_json":       get("Bench JSON"),
        "notes":            get("Notes (optional)"),
    }


def set_output(name: str, value: str) -> None:
    gho = os.environ.get("GITHUB_OUTPUT")
    if gho:
        with open(gho, "a") as f:
            f.write(f"{name}={value}\n")
    else:
        print(f"::set-output name={name}::{value}")


def main() -> None:
    issue_body   = os.environ["ISSUE_BODY"]
    issue_number = int(os.environ["ISSUE_NUMBER"])
    submitter    = os.environ["SUBMITTER"]

    parsed = parse_issue_body(issue_body)
    hardware = {
        "hardware_system":  parsed["hardware_system"],
        "chips_used":       parsed["chips_used"],
        "chips_in_system":  parsed["chips_in_system"],
        "firmware_version": parsed["firmware_version"],
        "backend_version":  parsed["backend_version"],
        "tt_kmd_version":   parsed.get("tt_kmd_version", ""),
        "submitter":        submitter,
        "submission_issue": issue_number,
    }

    result = validate(parsed["bench_json"], hardware)

    if not result.valid:
        set_output("valid", "false")
        set_output("errors", "\n".join(result.errors))
        sys.exit(0)  # exit 0 so the step succeeds; downstream step handles the failure

    # Write enriched JSONL
    date     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"{submitter}-{hardware['hardware_system']}-{hardware['chips_used']}chip-{date}.jsonl"
    outpath  = Path("data/community") / filename
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for record in result.records:
            f.write(json.dumps(record) + "\n")

    # Create branch → commit → push → open PR
    branch     = f"community/{submitter}-{issue_number}"
    n          = len(result.records)
    model_list = ", ".join(sorted({r["model_id"] for r in result.records}))
    pr_title   = f"[Community] {submitter} · {hardware['hardware_system']} · {hardware['chips_used']}-chip · {n} model(s)"

    pr_body = (
        f"Community bench submission from @{submitter}.\n\n"
        f"| Model | System | Chips | Compile | p50 Infer | Throughput |\n"
        f"|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r['model_id']} | {r['hardware_system']} | {r['chips_used']} "
            f"| {r['compile_s']}s | {r['infer_p50_s']*1000:.0f}ms "
            f"| {r['throughput_p50']} {r['throughput_unit']} |"
            for r in result.records
        )
        + f"\n\nCloses #{issue_number}\n"
    )

    subprocess.run(["git", "config", "user.email", "action@github.com"], check=True)
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", str(outpath)], check=True)
    subprocess.run(["git", "commit", "-m", f"community: {submitter} · {hardware['hardware_system']} · {hardware['chips_used']}-chip · {n} model(s)"], check=True)
    subprocess.run(["git", "push", "origin", branch], check=True)

    pr = subprocess.run(
        ["gh", "pr", "create",
         "--title", pr_title,
         "--body", pr_body,
         "--head", branch,
         "--base", "main"],
        capture_output=True, text=True, check=True,
    )
    pr_url = pr.stdout.strip()

    set_output("valid", "true")
    set_output("pr_url", pr_url)
    set_output("model_count", str(n))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_process_submission.py -v
```
Expected: all 3 tests pass.

- [ ] **Step 5: Create the workflow**

Create `.github/workflows/community-submission.yml`:

```yaml
name: Community Bench Submission

on:
  issues:
    types: [opened, edited]

jobs:
  process:
    if: contains(github.event.issue.labels.*.name, 'bench-submission')
    runs-on: ubuntu-latest
    permissions:
      issues: write
      contents: write
      pull-requests: write

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install pyyaml (for YAML issue template validation)
        run: pip install pyyaml --quiet

      - name: Process submission
        id: process
        env:
          ISSUE_BODY:   ${{ github.event.issue.body }}
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          SUBMITTER:    ${{ github.event.issue.user.login }}
          GH_TOKEN:     ${{ secrets.GITHUB_TOKEN }}
        run: python3 .github/scripts/process_submission.py

      - name: Comment — validation failed
        if: steps.process.outputs.valid == 'false'
        uses: actions/github-script@v7
        env:
          ERRORS: ${{ steps.process.outputs.errors }}
        with:
          script: |
            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `❌ **Validation failed** — please fix the following and edit your issue to re-trigger:\n\n\`\`\`\n${process.env.ERRORS}\n\`\`\``
            });
            await github.rest.issues.addLabels({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              labels: ['submission-invalid']
            });

      - name: Comment — PR opened
        if: steps.process.outputs.valid == 'true'
        uses: actions/github-script@v7
        env:
          PR_URL:      ${{ steps.process.outputs.pr_url }}
          MODEL_COUNT: ${{ steps.process.outputs.model_count }}
        with:
          script: |
            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `✓ **Validated** — ${process.env.MODEL_COUNT} model(s) · PR opened for maintainer review: ${process.env.PR_URL}`
            });
            await github.rest.issues.addLabels({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              labels: ['submission-pending']
            });
```

- [ ] **Step 6: Validate YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/community-submission.yml'))" && echo "YAML OK"
```
Expected: `YAML OK`

- [ ] **Step 7: Run the full test suite to confirm nothing regressed**

```bash
python3 -m pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add .github/scripts/process_submission.py .github/workflows/community-submission.yml tests/test_process_submission.py
git commit -m "feat: add community submission GitHub Action and issue parser"
```

---

## Task 4: Website Community Section

**Files:**
- Modify: `docs/index.html` (add `#community` section after `#perf`, around line 1347)

No automated tests — verify by opening `docs/index.html` in a browser and checking the section renders and fetches data.

- [ ] **Step 1: Add the community section HTML**

In `docs/index.html`, find the `<div class="section-divider"></div>` immediately after `</section>` that closes `#perf` (around line 1344) and insert after it:

```html
<div class="section-divider"></div>

<!-- ── Community ─────────────────────────────────────────────────────── -->
<section id="community" class="wide">
  <h2>Community Benchmarks &#8212; Players at Home</h2>

  <p>
    Verified 5-pass bench results from the community on their own Tenstorrent
    hardware. Each entry was reviewed and merged by a maintainer.
    <a href="https://github.com/tsingletaryTT/tt-forge-compiletron/issues/new?template=bench-submission.yml"
       style="color:var(--teal)">Submit your results &rarr;</a>
  </p>

  <div id="community-summary"
       style="color:var(--muted);font-size:0.88rem;margin-bottom:1.25rem">
    Loading community data&hellip;
  </div>

  <div id="community-table"></div>
</section>
```

- [ ] **Step 2: Add the community JS**

In `docs/index.html`, add the following `<script>` block immediately before the closing `</body>` tag:

```html
<script>
(async function loadCommunity() {
  var OWNER  = 'tsingletaryTT';
  var REPO   = 'tt-forge-compiletron';
  var BRANCH = 'main';
  var RAW    = 'https://raw.githubusercontent.com/' + OWNER + '/' + REPO + '/' + BRANCH;
  var API    = 'https://api.github.com/repos/' + OWNER + '/' + REPO;

  async function fetchJsonl(url) {
    try {
      var text = await fetch(url).then(function(r) { return r.ok ? r.text() : ''; });
      return text.trim().split('\n').filter(function(l) { return l.trim(); }).map(function(l) {
        try { return JSON.parse(l); } catch(e) { return null; }
      }).filter(Boolean);
    } catch(e) { return []; }
  }

  // Maintainer baseline — inject QB2 4-chip hardware fields
  var maintainerRaw = await fetchJsonl(RAW + '/data/perf_history.jsonl');
  var maintainerMap = {};
  maintainerRaw.forEach(function(r) {
    if (!maintainerMap[r.model_id] || r.timestamp > maintainerMap[r.model_id].timestamp)
      maintainerMap[r.model_id] = r;
  });
  var maintainerRows = Object.values(maintainerMap).map(function(r) {
    return Object.assign({}, r, {
      hardware_system: 'QB2', chips_used: 4, chips_in_system: 4,
      submitter: 'maintainer', submission_issue: null,
      firmware_version: '', backend_version: ''
    });
  });

  // Community files
  var communityRows = [];
  var files = await fetch(API + '/contents/data/community')
    .then(function(r) { return r.ok ? r.json() : []; }).catch(function() { return []; });
  if (Array.isArray(files)) {
    for (var i = 0; i < files.length; i++) {
      if (!files[i].name.endsWith('.jsonl')) continue;
      var records = await fetchJsonl(files[i].download_url);
      communityRows = communityRows.concat(records);
    }
  }

  // Summary line
  var summaryEl = document.getElementById('community-summary');
  if (communityRows.length === 0) {
    summaryEl.innerHTML = 'No community submissions yet &mdash; <a href="https://github.com/tsingletaryTT/tt-forge-compiletron/issues/new?template=bench-submission.yml" style="color:var(--teal)">be the first!</a>';
  } else {
    var submitters  = new Set(communityRows.map(function(r) { return r.submitter; }));
    var platforms   = new Set(communityRows.map(function(r) { return r.hardware_system; }));
    var modelIds    = new Set([].concat(maintainerRows, communityRows).map(function(r) { return r.model_id; }));
    summaryEl.textContent = communityRows.length + ' submission(s) · ' +
      submitters.size + ' contributor(s) · ' +
      platforms.size + ' hardware platform(s) · ' +
      modelIds.size + ' unique model(s)';
  }

  // Render: group by throughput_unit (language vs vision), then model, then hardware
  var allRows = maintainerRows.concat(communityRows);
  var GROUPS = [
    { label: 'Language Models',              unit: 'tokens/sec',  bestFirst: true  },
    { label: 'Vision / Embedding Models',    unit: 'ms/sample',   bestFirst: false },
  ];

  var html = '';
  GROUPS.forEach(function(group) {
    var rows = allRows.filter(function(r) { return r.throughput_unit === group.unit; });
    if (!rows.length) return;

    var byModel = {};
    rows.forEach(function(r) {
      if (!byModel[r.model_id]) byModel[r.model_id] = [];
      byModel[r.model_id].push(r);
    });

    // Sort models: most submissions first
    var sortedModels = Object.entries(byModel).sort(function(a, b) {
      return b[1].length - a[1].length;
    });

    html += '<h3 style="margin-top:2rem;color:var(--teal2)">' + group.label + '</h3>';

    sortedModels.forEach(function(entry) {
      var modelId   = entry[0];
      var modelRows = entry[1];

      // Sort by throughput
      var sorted = modelRows.slice().sort(function(a, b) {
        return group.bestFirst
          ? b.throughput_p50 - a.throughput_p50
          : a.throughput_p50 - b.throughput_p50;
      });

      html += '<div style="margin:1.25rem 0 0.35rem;font-weight:600;font-family:var(--mono);color:var(--fg)">' + modelId + '</div>';
      html += '<table class="perf-table"><thead><tr>' +
        '<th>System</th><th>Chips</th><th>Compile</th>' +
        '<th>p50 Infer</th><th>Throughput</th><th>Backend</th><th>By</th>' +
        '</tr></thead><tbody>';

      sorted.forEach(function(r) {
        var byCell = r.submitter === 'maintainer'
          ? 'maintainer'
          : '<a href="https://github.com/tsingletaryTT/tt-forge-compiletron/issues/' +
            r.submission_issue + '" style="color:var(--teal)">@' + r.submitter + ' &#8599;</a>';
        var infer = (r.infer_p50_s * 1000).toFixed(0) + 'ms';
        var tput  = r.throughput_unit === 'tokens/sec'
          ? r.throughput_p50.toLocaleString() + ' tok/s'
          : r.throughput_p50.toFixed(1) + ' ms/smp';
        var title = r.firmware_version
          ? ' title="fw: ' + r.firmware_version + ' · backend: ' + r.backend_version + '"'
          : '';
        html += '<tr' + title + '>' +
          '<td class="pt-chip">'    + r.hardware_system + '</td>' +
          '<td>'                    + r.chips_used + '-chip</td>' +
          '<td class="pt-compile">' + r.compile_s.toFixed(2) + 's</td>' +
          '<td class="pt-infer">'   + infer + '</td>' +
          '<td class="pt-tput">'    + tput + '</td>' +
          '<td class="pt-chip">'    + r.backend + '</td>' +
          '<td>'                    + byCell + '</td>' +
          '</tr>';
      });

      html += '</tbody></table>';
    });
  });

  document.getElementById('community-table').innerHTML = html;
})();
</script>
```

- [ ] **Step 3: Verify the section renders**

Open `docs/index.html` in a browser (or use a local server):

```bash
python3 -m http.server 8080 --directory docs
# open http://localhost:8080 in a browser
```

Check:
- `#community` section appears below `#perf`
- Summary line shows "Loading community data…" then resolves to either "No community submissions yet" or counts
- If the API is rate-limited (60 req/hr unauthenticated), the summary line disappears silently — that's correct behavior
- Maintainer baseline rows appear in the table grouped under Language Models / Vision Models

- [ ] **Step 4: Add `#community` to the site nav**

In `docs/index.html`, find the `<nav>` element and add a community link. The nav currently has entries like:

```html
<a href="#demo">Demo</a>
```

Add after the `#perf` entry:

```html
<a href="#community">Community</a>
```

- [ ] **Step 5: Commit**

```bash
git add docs/index.html
git commit -m "feat: add community benchmarks section to website"
```

---

## Task 5: GitHub Issue Labels Setup

**Files:**
- Create: `.github/labels.yml` (declarative label definitions)

GitHub Actions that add labels (`bench-submission`, `submission-invalid`, `submission-pending`, `submission-accepted`) require those labels to exist in the repo. This task creates them.

- [ ] **Step 1: Create label definitions**

Create `.github/labels.yml`:

```yaml
- name: bench-submission
  color: "0075ca"
  description: "Community bench result submission"

- name: submission-invalid
  color: "e4e669"
  description: "Submission failed schema validation"

- name: submission-pending
  color: "fbca04"
  description: "PR opened, awaiting maintainer review"

- name: submission-accepted
  color: "0e8a16"
  description: "Submission merged and live on website"
```

- [ ] **Step 2: Create labels in the repo using gh CLI**

```bash
gh label create "bench-submission"   --color "0075ca" --description "Community bench result submission"
gh label create "submission-invalid" --color "e4e669" --description "Submission failed schema validation"
gh label create "submission-pending" --color "fbca04" --description "PR opened, awaiting maintainer review"
gh label create "submission-accepted" --color "0e8a16" --description "Submission merged and live on website"
```

Each command prints: `✓ Label "bench-submission" created` (or similar).

- [ ] **Step 3: Commit label definitions**

```bash
git add .github/labels.yml
git commit -m "chore: add GitHub issue labels for community submissions"
```

---

## Task 6: Push and Smoke-Test

- [ ] **Step 1: Run the full test suite one last time**

```bash
python3 -m pytest tests/ -q
```
Expected: all tests pass, no failures.

- [ ] **Step 2: Push to origin**

```bash
git push origin xla-multichip
```

- [ ] **Step 3: Verify issue template appears on GitHub**

Open: `https://github.com/tsingletaryTT/tt-forge-compiletron/issues/new/choose`

Expected: "Bench Submission" option appears in the issue template chooser.

- [ ] **Step 4: Verify Actions tab**

Open: `https://github.com/tsingletaryTT/tt-forge-compiletron/actions`

Expected: "Community Bench Submission" workflow appears in the list (even though it hasn't fired yet).
