#!/usr/bin/env python3
"""Print a formatted bench-pass summary from data/perf_history.jsonl.

Usage:
    python3 scripts/show_perf_stats.py              # latest run
    python3 scripts/show_perf_stats.py --run 67     # specific run number
    python3 scripts/show_perf_stats.py --all        # every entry (long)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# ── ANSI colours ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
TEAL   = "\033[38;2;79;209;197m"
GOLD   = "\033[38;2;244;196;113m"
PINK   = "\033[38;2;236;150;184m"
GREEN  = "\033[38;2;39;174;96m"
RED    = "\033[38;2;255;107;107m"

NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")
if NO_COLOR:
    RESET = BOLD = DIM = TEAL = GOLD = PINK = GREEN = RED = ""


def _col(text: str, color: str, width: int = 0) -> str:
    padded = text.ljust(width) if width else text
    return f"{color}{padded}{RESET}" if color else padded


def _fmt_model(model_id: str, max_len: int = 36) -> str:
    """Trim long model IDs to fit the column."""
    if len(model_id) <= max_len:
        return model_id
    # keep the last segment (model name) + ellipsis
    parts = model_id.split("/")
    short = parts[-1]
    if len(short) <= max_len - 1:
        return "…" + short[-max_len + 1:]
    return short[:max_len - 1] + "…"


def _fmt_s(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}s"


def _fmt_throughput(tput: float, unit: str) -> str:
    if tput <= 0.0 or not unit:
        return "—"
    if unit == "tokens/sec":
        return f"{tput:.1f} tok/s"
    return f"{tput:.1f} ms/smp"


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        print(f"{RED}No perf_history.jsonl found at {path}{RESET}")
        sys.exit(1)
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def print_table(records: list[dict], run_label: str) -> None:
    if not records:
        print(f"{RED}No records found.{RESET}")
        return

    bench_records = [r for r in records if r.get("bench_passes", 0) > 0]
    passes = bench_records[0]["bench_passes"] if bench_records else 0

    sep = "═" * 88
    print(f"\n{TEAL}╔{sep}{RESET}")
    bench_note = f"  ({GOLD}{passes} bench passes each{RESET})" if passes else ""
    print(f"{TEAL}║{RESET}  {BOLD}Bench Results — {run_label}{RESET}{bench_note}")
    print(f"{TEAL}╠{sep}{RESET}")

    # Header
    hdr = (
        f"{'Model':<36}  "
        f"{'compile':>9}  "
        f"{'infer':>7}  "
        f"{'throughput':>12}  "
        f"{'p50 infer':>10}  "
        f"{'p95 infer':>10}  "
        f"{'p50 tput':>10}"
    )
    print(f"{TEAL}║{RESET}  {DIM}{hdr}{RESET}")
    print(f"{TEAL}║{RESET}  {DIM}{'─'*36}  {'─'*9}  {'─'*7}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}{RESET}")

    for r in records:
        model   = _fmt_model(r.get("model_id", "?"), 36)
        compile_s = r.get("compile_s", 0.0)
        infer_s   = r.get("infer_s", 0.0)
        tput      = r.get("throughput", 0.0)
        unit      = r.get("throughput_unit", "")
        p50_s     = r.get("infer_p50_s", 0.0)
        p95_s     = r.get("infer_p95_s", 0.0)
        p50_tput  = r.get("throughput_p50", 0.0)
        backend   = r.get("backend", "forge")
        chip      = r.get("chip", 0)

        backend_tag = f"{DIM}[{backend[0].upper()} c{chip}]{RESET}"

        col_model   = f"{TEAL}{model:<36}{RESET}"
        col_compile = f"{GOLD}{_fmt_s(compile_s):>9}{RESET}"
        col_infer   = f"{_fmt_s(infer_s):>7}"
        col_tput    = f"{GREEN}{_fmt_throughput(tput, unit):>12}{RESET}"
        col_p50     = f"{_fmt_s(p50_s) if p50_s > 0 else '—':>10}"
        col_p95     = f"{_fmt_s(p95_s) if p95_s > 0 else '—':>10}"
        col_p50t    = f"{_fmt_throughput(p50_tput, unit):>10}"

        print(f"{TEAL}║{RESET}  {col_model}  {col_compile}  {col_infer}  {col_tput}  {col_p50}  {col_p95}  {col_p50t}  {backend_tag}")

    print(f"{TEAL}╚{sep}{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show bench stats from perf_history.jsonl")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", type=int, metavar="N", help="Show a specific run number")
    group.add_argument("--all", action="store_true", help="Show all records")
    parser.add_argument("--data", default="data/perf_history.jsonl",
                        help="Path to perf_history.jsonl (default: data/perf_history.jsonl)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    path = root / args.data

    records = load_records(path)
    if not records:
        print("perf_history.jsonl is empty.")
        return

    if args.all:
        print_table(records, f"all runs ({len(records)} records)")
        return

    if args.run is not None:
        filtered = [r for r in records if r.get("run") == args.run]
        if not filtered:
            print(f"{RED}No records for run {args.run}.{RESET}")
            sys.exit(1)
        print_table(filtered, f"Expedition #{args.run:03d}")
        return

    # Default: latest run
    latest_run = max(r.get("run", 0) for r in records)
    filtered = [r for r in records if r.get("run") == latest_run]
    print_table(filtered, f"Expedition #{latest_run:03d}")


if __name__ == "__main__":
    main()
