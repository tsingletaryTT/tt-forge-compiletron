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

# Use importlib to load validate_submission by absolute path, which is robust against
# namespace-package pollution from tt-forge-fe being inserted into sys.path by
# lib/expedition/expedition_worker.py at import time (which can corrupt the 'scripts'
# namespace package cache before our repo root is in sys.path).
import importlib.util as _ilutil

_repo_root = Path(__file__).parent.parent.parent
_vs_path = _repo_root / "scripts" / "validate_submission.py"
_vs_spec = _ilutil.spec_from_file_location("scripts.validate_submission", _vs_path)
_vs_mod = _ilutil.module_from_spec(_vs_spec)
# Register in sys.modules BEFORE exec so that dataclass field resolution works correctly
# (dataclasses uses sys.modules[cls.__module__] to look up the class namespace).
sys.modules.setdefault("scripts.validate_submission", _vs_mod)
_vs_spec.loader.exec_module(_vs_mod)
validate = _vs_mod.validate


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
        "chips_used":       _parse_int(get("Chips used for this run")),
        "chips_in_system":  _parse_int(get("Total chips in system")),
        "firmware_version": get("Firmware version"),
        "backend_version":  get("Backend version"),
        "tt_kmd_version":   get("tt-kmd version (optional)"),
        "bench_json":       get("Bench JSON"),
        "notes":            get("Notes (optional)"),
    }


def _parse_int(s: str) -> int:
    """Parse an integer from a form field value; returns 0 on non-numeric input."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def set_output(name: str, value: str) -> None:
    gho = os.environ.get("GITHUB_OUTPUT")
    if gho:
        delimiter = f"GHADELIM_{name}"
        with open(gho, "a") as f:
            f.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")
    else:
        # Legacy fallback — GITHUB_OUTPUT is always set on modern runners
        print(f"::set-output name={name}::{value}", file=sys.stderr)


def main() -> None:
    issue_body   = os.environ.get("ISSUE_BODY", "")
    issue_number_str = os.environ.get("ISSUE_NUMBER", "")
    submitter    = os.environ.get("SUBMITTER", "")
    if not issue_body or not issue_number_str or not submitter:
        print("✗ Required env vars missing: ISSUE_BODY, ISSUE_NUMBER, SUBMITTER", file=sys.stderr)
        sys.exit(1)
    issue_number = int(issue_number_str)

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

    subprocess.run(["git", "config", "--local", "user.email", "action@github.com"], check=True)
    subprocess.run(["git", "config", "--local", "user.name", "github-actions[bot]"], check=True)
    # If branch already exists (re-trigger on issue edit), reset it
    existing = subprocess.run(["git", "branch", "--list", branch], capture_output=True, text=True)
    if existing.stdout.strip():
        subprocess.run(["git", "checkout", branch], check=True)
        subprocess.run(["git", "reset", "--hard", "HEAD"], check=True)
    else:
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
