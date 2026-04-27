#!/usr/bin/env python3
# expedition.py
"""
Expedition Mode entry point.

Usage:
  python3 expedition.py                        # auto-detect chips, full run
  python3 expedition.py --chips 2              # limit to 2 chips
  python3 expedition.py --seed-only            # skip HF discovery
  python3 expedition.py --frontier-only        # skip forge-models seed
  python3 expedition.py --limit 20             # cap models per chip
  python3 expedition.py summary                # print bestiary summary
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
BESTIARY_PATH = DATA_DIR / "bestiary.json"
RUNS_DIR = DATA_DIR / "runs"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

# HuggingFace model cache — respects env override used by huggingface_hub.
HF_CACHE_DIR = Path(
    os.environ.get("HUGGINGFACE_HUB_CACHE",
    os.environ.get("HF_HOME",
    str(Path.home() / ".cache" / "huggingface" / "hub")))
)


def _hf_cache_gb() -> float:
    """Return current HF cache size in GB using `du -sb` (fast, OS-level)."""
    if not HF_CACHE_DIR.exists():
        return 0.0
    try:
        r = subprocess.run(
            ["du", "-sb", str(HF_CACHE_DIR)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            return int(r.stdout.split()[0]) / 1e9
    except Exception:
        pass
    return 0.0


# ── Queue building ────────────────────────────────────────────────────────────

def _scan_forge_models(bestiary_compiled_ids: set[str]) -> list[dict]:
    """
    Walk ~/code/tt-forge-models and return QueueItem dicts for loaders
    not yet in the bestiary.

    Each loader.py under the forge-models tree is expected to define a
    ForgeModel subclass.  We reflect into each module to extract the class
    name and the model's task/source metadata, then build a minimal queue
    item dict that expedition_worker.py can consume.

    Models already in compiled_ids are skipped — they have been tamed and
    no longer count as expedition targets.
    """
    forge_models_root = Path.home() / "code" / "tt-forge-models"
    if not forge_models_root.exists():
        return []

    # Make the forge-models tree importable so we can load loader modules.
    # The path is removed in the finally block below to avoid polluting
    # sys.path for the remainder of the process.
    sys.path.insert(0, str(forge_models_root))
    items = []

    try:
        for loader_py in sorted(forge_models_root.rglob("loader.py")):
            # Skip hidden directories and private packages (leading _ or .).
            # Filter on the *relative* path only — absolute path components such
            # as a home directory that starts with '.' would otherwise cause all
            # models to be silently skipped.
            rel = loader_py.relative_to(forge_models_root)
            if any(p.startswith("_") or p.startswith(".") for p in rel.parts):
                continue

            # model_id is the directory path relative to forge-models root, minus the
            # trailing "loader.py" segment — e.g. "facebook/bart-large-cnn".
            model_id = "/".join(rel.parts[:-1])

            if model_id in bestiary_compiled_ids:
                continue

            # Convert path to dotted module path for importlib.
            module_path = ".".join(rel.parts[:-1]) + ".loader"

            try:
                import importlib
                mod = importlib.import_module(module_path)
                from base import ForgeModel
                cls_name = None
                for name in dir(mod):
                    obj = getattr(mod, name)
                    try:
                        if isinstance(obj, type) and issubclass(obj, ForgeModel) and obj is not ForgeModel:
                            cls_name = name
                            break
                    except Exception:
                        continue
                if cls_name is None:
                    continue

                # Instantiate the class (no args) to call _get_model_info().
                instance = obj()
                info = instance._get_model_info()
                # ModelInfo.task and ModelInfo.source may be enum instances; use .value
                # if available so we get clean strings in the queue JSON.
                task = info.task.value if hasattr(info.task, "value") else str(info.task)
                source = info.source.value if hasattr(info.source, "value") else str(info.source)

            except Exception:
                # Any import/reflection failure is non-fatal: skip this loader.
                continue

            items.append({
                "model_id": model_id,
                # Derive a human-readable display name from the first path component.
                "display_name": model_id.split("/")[0].replace("_", " ").title(),
                "task": task,
                "source": source,
                "rarity": "familiar",       # seed models are known quantities
                "hf_downloads": None,
                "hf_created_at": None,
                "mesh_chips": 1,
                "loader_module": module_path,
                "loader_class": cls_name,
                "is_frontier": False,
            })
    finally:
        # Always restore sys.path even if an unexpected exception occurs mid-scan.
        try:
            sys.path.remove(str(forge_models_root))
        except ValueError:
            pass

    return items


def _scan_frontier(
    bestiary_compiled_ids: set[str],
    forge_model_ids: set[str],
    min_downloads: int = 0,
    min_likes: int = 0,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
) -> list[dict]:
    """
    Query the HuggingFace frontier for models not yet in the bestiary and not
    already covered by the forge-models library.

    Returns a list of queue item dicts, one per discovered FrontierModel.
    The loader_module and loader_class fields are None for frontier models —
    expedition_worker.py will build a dynamic loader at runtime using
    hf_discover.build_dynamic_loader().
    """
    from lib.expedition.hf_discover import discover_frontier
    models = discover_frontier(
        compiled_ids=bestiary_compiled_ids,
        known_model_ids=forge_model_ids,
        min_downloads=min_downloads,
        min_likes=min_likes,
        max_params_b=max_params_b,
        skip_gated=skip_gated,
    )
    return [
        {
            "model_id": m.model_id,
            "display_name": m.model_id.split("/")[-1],
            "task": m.pipeline_tag,
            "source": "huggingface",
            "rarity": m.rarity.value,
            "hf_downloads": m.downloads,
            "hf_likes": m.likes,
            "hf_params_b": m.params_b,
            "hf_created_at": m.created_at.isoformat() if m.created_at else None,
            "mesh_chips": m.mesh_chips,
            "loader_module": None,
            "loader_class": None,
            "is_frontier": True,
        }
        for m in models
    ]


def build_queues(
    num_chips: int,
    seed_only: bool = False,
    frontier_only: bool = False,
    limit_per_chip: int = 0,
    min_downloads: int = 0,
    min_likes: int = 0,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
) -> list[list[dict]]:
    """
    Build per-chip model queues by merging forge-models seed items with HF
    frontier discoveries.

    The final list is interleaved so each chip gets a mix of familiar seed
    models (60 %) and fresh frontier targets (40 %).  Items are then
    distributed round-robin across chips.

    Args:
        num_chips:      Number of Tenstorrent chips to distribute work across.
        seed_only:      If True, skip HuggingFace frontier discovery.
        frontier_only:  If True, skip forge-models seed scan.
        limit_per_chip: If > 0, truncate each chip's queue to this length.

    Returns:
        A list of num_chips lists, each containing queue item dicts.
    """
    from lib.expedition.bestiary import Bestiary
    # Bestiary only accepts a single `path` argument — no `runs_dir` param.
    bestiary = Bestiary(path=str(BESTIARY_PATH))
    # compiled is a public dict property; use .keys() to get the compiled IDs.
    compiled_ids = set(bestiary.compiled.keys())

    seed_items: list[dict] = []
    frontier_items: list[dict] = []

    if not frontier_only:
        print("  Scanning tt-forge-models library...")
        seed_items = _scan_forge_models(compiled_ids)
        print(f"  {len(seed_items)} seed models queued (not yet compiled)")

    forge_ids = {item["model_id"] for item in seed_items}

    if not seed_only:
        print("  Querying HuggingFace frontier...")
        frontier_items = _scan_frontier(
            compiled_ids, forge_ids,
            min_downloads=min_downloads,
            min_likes=min_likes,
            max_params_b=max_params_b,
            skip_gated=skip_gated,
        )
        print(f"  {len(frontier_items)} frontier models discovered")

    # Interleave seed (60 %) and frontier (40 %) for a balanced run.
    all_items = _interleave(seed_items, frontier_items, seed_ratio=0.6)

    # Final deduplication pass — belt-and-suspenders guard against any upstream
    # source (HF pagination, symlinked loader.py paths, etc.) producing the same
    # model_id twice, which would send it to two different chips via round-robin.
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in all_items:
        mid = item["model_id"]
        if mid not in seen:
            seen.add(mid)
            deduped.append(item)
    if len(deduped) < len(all_items):
        print(f"  Deduplicated {len(all_items) - len(deduped)} duplicate model(s)")
    all_items = deduped

    print(f"  Total queue: {len(all_items)} models across {num_chips} chip(s)")

    # Round-robin distribution across chips.
    chip_queues: list[list[dict]] = [[] for _ in range(num_chips)]
    for i, item in enumerate(all_items):
        chip_queues[i % num_chips].append(item)

    if limit_per_chip > 0:
        chip_queues = [q[:limit_per_chip] for q in chip_queues]

    return chip_queues


def _interleave(seed: list, frontier: list, seed_ratio: float) -> list:
    """
    Interleave two lists such that seed items appear at the given ratio.

    seed_ratio=0.6 means roughly 3 seed items for every 2 frontier items.
    Budget accounting ensures the ratio is maintained globally rather than
    just per-pair, so leftover items from the shorter list are appended at
    the end.
    """
    result = []
    si = fi = 0
    seed_budget = 0.0
    while si < len(seed) or fi < len(frontier):
        # Accumulate seed budget; emit seed items while budget >= 1.
        seed_budget += seed_ratio
        while seed_budget >= 1.0 and si < len(seed):
            result.append(seed[si])
            si += 1
            seed_budget -= 1.0
        # Emit one frontier item per outer loop iteration.
        if fi < len(frontier):
            result.append(frontier[fi])
            fi += 1
    return result


# ── Pre-download ─────────────────────────────────────────────────────────────

def _predownload_queues(chip_queues: list[list[dict]], max_cache_gb: float = 0.0,
                        session_download_max_gb: float = 0.0) -> None:
    """Pre-fetch HuggingFace weights for all frontier models before compile starts.

    Collects unique frontier model IDs across all chip queues and calls
    snapshot_download so that weights land in the local HF cache.  This puts
    all chips on equal footing at compile time — none stalls waiting for a
    download that others finished earlier.

    Forge-model seed entries skip this step; their weights are pulled by the
    loader's own from_pretrained call and are typically already cached.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub not available — skipping pre-download")
        return

    # Collect unique frontier model IDs (de-duplicate across chips).
    seen: dict[str, dict] = {}
    for queue in chip_queues:
        for item in queue:
            if item.get("is_frontier") and item["model_id"] not in seen:
                seen[item["model_id"]] = item

    if not seen:
        print("  No frontier models to pre-download.")
        return

    # Large binary formats we can't use — skip to save bandwidth and disk.
    ignore = ["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.ot"]

    # Disk guard: measure cache once up front, re-check per model.
    import shutil as _shutil
    cache_root = HF_CACHE_DIR.parent if HF_CACHE_DIR.exists() else Path.home()
    baseline_gb = _hf_cache_gb()  # snapshot before this session's downloads begin

    limits = []
    if max_cache_gb > 0:
        print(f"  HF cache: {baseline_gb:.1f} GB used  (limit {max_cache_gb:.0f} GB)")
        if baseline_gb >= max_cache_gb:
            print("  Cache already at limit — skipping pre-download.")
            return
        limits.append(f"cache≤{max_cache_gb:.0f} GB")
    if session_download_max_gb > 0:
        limits.append(f"session≤{session_download_max_gb:.0f} GB new")
    if limits:
        print(f"  Download guards: {', '.join(limits)}")

    total = len(seen)
    print(f"\n  Pre-downloading {total} frontier model(s) to HF cache...")
    ok = fail = skipped = 0
    session_gb = 0.0  # cumulative new GB fetched this session
    for i, (model_id, item) in enumerate(seen.items(), 1):
        # Per-model disk checks before attempting the download.
        current_gb = _hf_cache_gb()
        if max_cache_gb > 0 and current_gb >= max_cache_gb:
            skipped = total - i + 1
            print(f"\n  Cache at {current_gb:.1f}/{max_cache_gb:.0f} GB — "
                  f"stopping ({skipped} will download at compile time)")
            break
        session_gb = current_gb - baseline_gb
        if session_download_max_gb > 0 and session_gb >= session_download_max_gb:
            skipped = total - i + 1
            print(f"\n  Session cap reached ({session_gb:.1f}/{session_download_max_gb:.0f} GB) — "
                  f"stopping ({skipped} will download at compile time)")
            break
        free_gb = _shutil.disk_usage(cache_root).free / 1e9
        if free_gb < 5.0:
            skipped = total - i + 1
            print(f"\n  ⚠ Disk critically low ({free_gb:.1f} GB free) — "
                  f"stopping pre-download ({skipped} remaining)")
            break

        task = item.get("task", "")
        params = item.get("hf_params_b", 0) or 0
        size_hint = f" ~{params:.1f}B" if params > 0 else ""
        session_str = f"  +{session_gb:.1f} GB this session" if session_gb > 0.1 else ""
        label = f"{model_id}{size_hint} ({task})" if task else f"{model_id}{size_hint}"
        print(f"  [{i:>{len(str(total))}}/{total}] {label[:60]:<62}{session_str}", end="  ", flush=True)
        try:
            snapshot_download(model_id, ignore_patterns=ignore, local_files_only=False)
            print("✓")
            ok += 1
        except Exception as e:
            print(f"✗  {str(e)[:50]}")
            fail += 1

    final_session_gb = _hf_cache_gb() - baseline_gb
    parts = [f"✓{ok}"]
    if fail:    parts.append(f"✗{fail}")
    if skipped: parts.append(f"⏭{skipped} deferred")
    if final_session_gb > 0.1: parts.append(f"+{final_session_gb:.1f} GB fetched")
    print(f"\n  Pre-download complete — {'  '.join(parts)}\n")


# ── Run summary ──────────────────────────────────────────────────────────────

def _print_run_summary(num_chips: int, run_number: int) -> None:
    """
    Aggregate end-of-run summary printed to the launching terminal after tmux exits.

    Reads per-chip CSV result files written by expedition_worker.py from /tmp,
    ranks chips by total points, lists new bestiary entries (first-evers), and
    prints a failure table.  Writes a compact run JSON to data/runs/.
    """
    import csv
    from lib.expedition.bestiary import Bestiary

    chip_results: list[dict] = []
    for chip_id in range(num_chips):
        path = Path(f"/tmp/expedition_results_chip{chip_id}.csv")
        if not path.exists():
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        successes = [r for r in rows if r.get("status") == "success"]
        failures  = [r for r in rows if r.get("status") == "failed"]
        # pts column may be empty string for failed rows; `or 0` coerces ""
        # to 0 before int() conversion, avoiding a ValueError crash.
        total_pts = sum(int(r.get("pts") or 0) for r in rows)
        first_evers = [r for r in successes if r.get("first_ever") == "True"]
        chip_results.append({
            "chip_id": chip_id,
            "pts": total_pts,
            "successes": successes,
            "failures": failures,
            "first_evers": first_evers,
        })

    # Rank chips by descending points for the leaderboard.
    chip_results.sort(key=lambda x: -x["pts"])

    W = 72

    # Guard: if no CSV files were found (all workers failed to write), print a
    # clear diagnostic message and return early rather than showing a misleading
    # "EXPEDITION COMPLETE" banner with no rows.
    if not chip_results:
        print(f"\n{'═'*W}")
        print(f"  EXPEDITION #{run_number:03d} — NO RESULTS")
        print(f"  No per-chip CSV files found in /tmp.")
        print(f"  Workers may not have completed. Check /tmp/expedition_results_chip*.csv")
        print(f"{'═'*W}\n")
        return

    medals = ["🥇", "🥈", "🥉", "  "]
    print(f"\n{'═'*W}")
    print(f"  EXPEDITION #{run_number:03d} COMPLETE")
    print(f"{'═'*W}")
    for i, c in enumerate(chip_results):
        medal = medals[min(i, 3)]
        fe = len(c["first_evers"])
        print(f"  {medal} CHIP {c['chip_id']}   {c['pts']:,} pts   "
              f"✓{len(c['successes'])} ✗{len(c['failures'])}   ★{fe} first-evers")

    all_first_evers = [
        r for c in chip_results for r in c["first_evers"]
    ]
    if all_first_evers:
        print(f"\n{'─'*W}")
        print("  NEW TO BESTIARY:")
        for r in all_first_evers:
            artifact = (r.get("artifact") or "")[:80]
            rune = "★"
            print(f"  {rune} {r['model']:40s}  {artifact}")

    all_failures = [r for c in chip_results for r in c["failures"]]
    if all_failures:
        print(f"\n{'─'*W}")
        print("  FAILED:")
        for r in all_failures:
            print(f"  ✗ {r['model']:40s}  {(r.get('error') or '')[:40]}")

    # Final bestiary headcount — Bestiary only takes path, no runs_dir.
    b = Bestiary(path=str(BESTIARY_PATH))
    # compiled is a public dict property; len() gives the total count.
    compiled_count = len(b.compiled)
    print(f"\n{'─'*W}")
    print(f"  BESTIARY: {compiled_count} total compiled")
    print(f"{'═'*W}\n")

    # Persist a compact run record for historical lookup.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_file = RUNS_DIR / f"run_{run_number:03d}.json"
    run_file.write_text(json.dumps({
        "run": run_number,
        "timestamp": datetime.now().isoformat(),
        "chips": num_chips,
        # Flatten list fields to counts so the JSON stays compact.
        "chip_results": [
            {k: v if not isinstance(v, list) else len(v) for k, v in c.items()}
            for c in chip_results
        ],
        "new_bestiary_entries": len(all_first_evers),
        "total_failures": len(all_failures),
    }, indent=2))


# ── Summary command ───────────────────────────────────────────────────────────

def cmd_summary():
    """
    Print a human-readable snapshot of the expedition bestiary: total
    compiled, total failed, and a per-chip hall-of-fame table.
    """
    from lib.expedition.bestiary import Bestiary
    # Bestiary only accepts path — no runs_dir parameter.
    b = Bestiary(path=str(BESTIARY_PATH))
    # Use the public dict properties; do NOT access _data directly.
    compiled = b.compiled
    failed = b.failed
    totals = b.chip_totals

    print(f"\n{'═'*60}")
    print(f"  EXPEDITION BESTIARY")
    print(f"{'═'*60}")
    print(f"  Compiled:  {len(compiled)} models")
    print(f"  Failed:    {len(failed)} models")
    if totals:
        print(f"\n  Chip Hall of Fame:")
        for chip_id, data in sorted(totals.items(), key=lambda x: -x[1]["pts"]):
            print(f"    Chip {chip_id}: {data['pts']:,} pts  "
                  f"★{data['first_evers']} first-evers  "
                  f"best streak ×{data['best_streak']}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Expedition Mode — roguelike forge compilation")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Start an expedition run (default)")
    run_p.add_argument("--chips",          type=int, default=0,
                       help="Number of chips (0=auto-detect)")
    run_p.add_argument("--limit",          type=int, default=0,
                       help="Max models per chip (0=unlimited)")
    run_p.add_argument("--seed-only",        action="store_true")
    run_p.add_argument("--frontier-only",    action="store_true")
    run_p.add_argument("--no-predownload",   action="store_true",
                       help="Skip pre-downloading HF weights (faster start, unequal footing)")
    run_p.add_argument("--monitor",          action="store_true",
                       help="Add a tt-smi hardware monitor pane in the center column")
    # ── Quality / reputation bar ──────────────────────────────────────────────
    run_p.add_argument("--min-downloads",    type=int,   default=0, metavar="N",
                       help="Skip frontier models with fewer than N total downloads "
                            "(0=off; try 1000 for proven models, 10000 for popular ones)")
    run_p.add_argument("--min-likes",        type=int,   default=0, metavar="N",
                       help="Skip frontier models with fewer than N HuggingFace likes "
                            "(0=off; try 5 to filter pure experiment dumps)")
    run_p.add_argument("--max-model-params", type=float, default=0.0, metavar="B",
                       help="Skip frontier models larger than B billion parameters "
                            "(0=off; try 7 for single-chip sweet-spot, 13 for upper limit)")
    run_p.add_argument("--allow-gated",      action="store_true",
                       help="Include gated HuggingFace models (requires an approved "
                            "access token — downloads will fail without it)")
    # ── Disk management ───────────────────────────────────────────────────────
    run_p.add_argument("--max-cache-gb",          type=float, default=0.0, metavar="GB",
                       help="Stop pre-downloading when HF cache exceeds GB gigabytes "
                            "(0=off; e.g. 150 to cap at 150 GB)")
    run_p.add_argument("--session-download-max",  type=float, default=0.0, metavar="GB",
                       help="Stop pre-downloading after fetching GB gigabytes this session "
                            "(0=off; e.g. 60 to limit a single run to 60 GB of new downloads)")

    sub.add_parser("summary", help="Print bestiary summary")

    args = parser.parse_args()

    # Handle the summary sub-command immediately and exit.
    if args.cmd == "summary":
        cmd_summary()
        return

    # No subcommand → default run with no-arg defaults.
    if args.cmd is None:
        args.chips = 0
        args.limit = 0
        args.seed_only = False
        args.frontier_only = False
        args.no_predownload = False
        args.monitor = False
        args.min_downloads = 0
        args.min_likes = 0
        args.max_model_params = 0.0
        args.allow_gated = False
        args.max_cache_gb = 0.0
        args.session_download_max = 0.0

    # ── Hardware detection ────────────────────────────────────────────────────
    from lib.hardware import detect_hardware, get_hardware_summary
    hw = detect_hardware()
    num_chips = args.chips if args.chips > 0 else hw.get("num_chips", 1)
    if num_chips == 0:
        print("No chips detected. Check tt-smi.")
        sys.exit(1)
    print(f"\n  Hardware: {get_hardware_summary(hw)}")
    print(f"  Chips for this run: {num_chips}")

    # ── Run numbering ─────────────────────────────────────────────────────────
    # Derive next run number from the count of existing run JSON files.
    # We do NOT call any Bestiary.next_run_number() — that method does not exist.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run_number = len(list(RUNS_DIR.glob("run_*.json"))) + 1
    print(f"  Run #{run_number:03d}")

    # ── Queue building ────────────────────────────────────────────────────────
    chip_queues = build_queues(
        num_chips=num_chips,
        seed_only=args.seed_only,
        frontier_only=args.frontier_only,
        limit_per_chip=args.limit,
        min_downloads=args.min_downloads,
        min_likes=args.min_likes,
        max_params_b=args.max_model_params,
        skip_gated=not args.allow_gated,
    )

    # ── Pre-download weights for fairness ────────────────────────────────────
    if not args.no_predownload:
        _predownload_queues(chip_queues, max_cache_gb=args.max_cache_gb,
                            session_download_max_gb=args.session_download_max)

    # ── Write per-chip queue JSON to /tmp ─────────────────────────────────────
    # expedition_worker.py reads these files at startup.
    for chip_id, queue in enumerate(chip_queues):
        queue_path = f"/tmp/expedition_queue_chip{chip_id}.json"
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)
        print(f"  Chip {chip_id}: {len(queue)} models → {queue_path}")

    # ── Launch tmux runner ────────────────────────────────────────────────────
    # Blocks until the user detaches from the session or the session ends.
    script = PROJECT_DIR / "scripts" / "run_expedition.sh"
    env = {**os.environ, "EXPEDITION_RUN": str(run_number),
           "EXPEDITION_NUM_CHIPS": str(num_chips)}
    cmd = ["bash", str(script), "--chips", str(num_chips), "--run", str(run_number)]
    if args.monitor:
        cmd.append("--monitor")
    subprocess.run(cmd, env=env)

    # ── Post-run aggregate summary ────────────────────────────────────────────
    # After tmux exits, gather per-chip CSV results and print the leaderboard.
    _print_run_summary(num_chips, run_number)


if __name__ == "__main__":
    main()
