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

    # Pre-format all cell values so we can measure actual widths before rendering.
    HEADERS = ("model", "compile", "infer", "throughput", "p50 infer", "p95 infer", "p50 tput", "chip")
    rows: list[tuple[str, ...]] = []
    for r in records:
        unit    = r.get("throughput_unit", "")
        p50_s   = r.get("infer_p50_s", 0.0)
        p95_s   = r.get("infer_p95_s", 0.0)
        p50t    = r.get("throughput_p50", 0.0)
        backend = r.get("backend", "forge")
        chip    = r.get("chip", 0)
        rows.append((
            _fmt_model(r.get("model_id", "?"), 40),
            _fmt_s(r.get("compile_s", 0.0)),
            _fmt_s(r.get("infer_s",   0.0)),
            _fmt_throughput(r.get("throughput", 0.0), unit),
            _fmt_s(p50_s) if p50_s > 0 else "—",
            _fmt_s(p95_s) if p95_s > 0 else "—",
            _fmt_throughput(p50t, unit) if p50t > 0 else "—",
            f"[{backend[0].upper()} c{chip}]",
        ))

    # Column widths = max of header width and widest data cell.
    col_w = [max(len(h), max(len(row[i]) for row in rows))
             for i, h in enumerate(HEADERS)]

    # Row template: 2-space margin on left, 2-space gap between every column.
    def _render_row(cells: tuple[str, ...], colors: tuple[str, ...]) -> str:
        parts = []
        for i, (cell, color) in enumerate(zip(cells, colors)):
            # model and chip are left-aligned; all others right-aligned.
            if i in (0, 7):
                padded = cell.ljust(col_w[i])
            else:
                padded = cell.rjust(col_w[i])
            parts.append(f"{color}{padded}{RESET}" if color else padded)
        return "  " + "  ".join(parts)

    ROW_COLORS = (TEAL, GOLD, "", GREEN, "", "", GREEN, DIM)
    HDR_COLORS = tuple("" for _ in HEADERS)

    # Total visual width of a content row (used for the ═ border lines).
    row_visual_w = 2 + sum(col_w) + 2 * (len(col_w) - 1)
    sep = "═" * row_visual_w

    print(f"\n{TEAL}╔{sep}{RESET}")
    bench_note = f"  ({GOLD}{passes} bench passes each{RESET})" if passes else ""
    print(f"{TEAL}║{RESET}  {BOLD}Bench Results — {run_label}{RESET}{bench_note}")
    print(f"{TEAL}╠{sep}{RESET}")

    hdr_row = _render_row(HEADERS, HDR_COLORS)
    dash_row = "  " + "  ".join("─" * w for w in col_w)
    print(f"{TEAL}║{RESET}{DIM}{hdr_row}{RESET}")
    print(f"{TEAL}║{RESET}{DIM}{dash_row}{RESET}")

    for row in rows:
        print(f"{TEAL}║{RESET}{_render_row(row, ROW_COLORS)}")

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
