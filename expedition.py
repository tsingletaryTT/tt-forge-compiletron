#!/usr/bin/env python3
# expedition.py
"""
Expedition Mode entry point.

Usage:
  python3 expedition.py                        # auto-detect chips, full run
  python3 expedition.py --chips 2              # limit to 2 chips
  python3 expedition.py --seed-only            # skip HF discovery
  python3 expedition.py --frontier-only        # skip forge-models seed
  python3 expedition.py --limit 20             # cap total unique models across all chips
  python3 expedition.py summary                # print bestiary summary
"""
from __future__ import annotations
import warnings as _warnings
_warnings.filterwarnings("ignore")   # silence deprecation/import noise before TUI starts
import argparse
import fnmatch as _fnmatch
import json
import logging as _logging
import os
import re
import shutil as _shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# ── ANSI color palette (Tenstorrent theme) ────────────────────────────────────
# These constants are deliberately short so inline use stays readable.
# All color codes use 24-bit RGB sequences for consistent rendering in modern
# terminals; _GRAY falls back to the 256-color table (color #240, dark gray).

_RST   = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_TEAL  = "\033[38;2;79;209;197m"    # #4FD1C5 — primary accent
_LTEAL = "\033[38;2;129;230;217m"   # #81E6D9 — hover / secondary
_PINK  = "\033[38;2;236;150;184m"   # #EC96B8 — epic rarity
_GOLD  = "\033[38;2;244;196;113m"   # #F4C471 — legendary / warnings
_GRN   = "\033[38;2;39;174;96m"     # #27AE60 — success
_RED   = "\033[38;2;255;107;107m"   # #FF6B6B — failure
_GRAY  = "\033[38;5;240m"           # dark gray (256-color table)
_WHT   = "\033[38;2;232;240;242m"   # #E8F0F2 — main text


# ── UI helper functions ───────────────────────────────────────────────────────

def _tw() -> int:
    """Return current terminal width, clamped to [60, 120]."""
    return max(60, min(120, _shutil.get_terminal_size((100, 24)).columns))


def _bar(ratio: float, width: int = 28) -> str:
    """Render a teal/dim progress bar of the given character width.

    ratio is clamped to [0, 1].  Filled cells use a solid block (█) in the
    primary teal accent color; empty cells use a dim light shade (░) so the
    bar width is always visually consistent even at 0 %.
    """
    filled = round(min(ratio, 1.0) * width)
    return _TEAL + "█" * filled + _DIM + "░" * (width - filled) + _RST


def _fmt_num(n) -> str:
    """Format a raw integer count as a compact human-readable string (e.g. 1.2M, 42k)."""
    if n is None:
        return "?"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}k"
    return str(n)


def _fmt_bytes(b: int) -> str:
    """Format a byte count as a compact human-readable string (GB/MB/KB/B)."""
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    if b >= 1e6:
        return f"{b/1e6:.0f} MB"
    if b >= 1e3:
        return f"{b/1e3:.0f} KB"
    return f"{b} B"


def _fmt_speed(bps: float) -> str:
    """Format a bytes-per-second throughput as a compact human-readable string."""
    if bps >= 1e9:
        return f"{bps/1e9:.1f} GB/s"
    if bps >= 1e6:
        return f"{bps/1e6:.0f} MB/s"
    return f"{bps/1e3:.0f} KB/s"


def _fmt_eta(seconds: float) -> str:
    """Format an ETA in seconds as H:MM:SS or M:SS.  Returns '' when unknown or > 1 day."""
    if seconds <= 0 or seconds > 86400:
        return ""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _section(title: str) -> None:
    """Print a teal section header with a dim rule extending to the terminal edge."""
    w = _tw()
    bar = "─" * max(2, w - len(title) - 6)
    print(f"\n  {_TEAL}{_BOLD}{title}{_RST}  {_DIM}{bar}{_RST}")


def _banner(run_number: int, num_chips: int, hw_summary: str) -> None:
    """Print the expedition launch banner with run number and hardware summary."""
    w = min(_tw(), 64)
    inner = w - 4
    sub = f"Run #{run_number:03d}  ·  {num_chips}× {hw_summary}"
    print()
    print(f"  {_TEAL}╔{'═' * inner}{_RST}")
    print(f"  {_TEAL}║{_RST}  {_BOLD}{_WHT}⚡  EXPEDITION MODE{_RST}")
    print(f"  {_TEAL}║{_RST}  {_DIM}tt-forge-compiletron{_RST}")
    print(f"  {_TEAL}║{_RST}")
    print(f"  {_TEAL}║{_RST}  {_TEAL}{sub}{_RST}")
    print(f"  {_TEAL}╚{'═' * inner}{_RST}")


# Rarity display: maps rarity string → (ANSI prefix, fixed-width glyph+label).
# "familiar" is the seed-model rarity — visually quieter than even "common".
_RARITY_FMT = {
    "legendary": (_GOLD + _BOLD, "★ LEGENDARY"),
    "epic":      (_PINK,         "◆ EPIC     "),
    "rare":      (_TEAL,         "◆ RARE     "),
    "uncommon":  (_LTEAL,        "▸ UNCOMMON "),
    "common":    (_GRAY,         "· COMMON   "),
    "familiar":  (_DIM,          "· seed     "),
}


def _model_row(item: dict) -> None:
    """Print one model line with rarity glyph, model ID, task label, and stat pills."""
    rarity = (item.get("rarity") or "common").lower()
    color, glyph = _RARITY_FMT.get(rarity, (_GRAY, "· "))
    mid = item.get("model_id", "?")
    task = item.get("task") or item.get("source") or ""
    dl = item.get("hf_downloads")
    likes = item.get("hf_likes")
    params = item.get("hf_params_b", 0) or 0
    chips = item.get("mesh_chips", 1)

    # Assemble right-hand stat pills: download count, likes, param count, multi-chip flag.
    stats = []
    if dl is not None:    stats.append(f"↓{_fmt_num(dl)}")
    if likes is not None: stats.append(f"♥{_fmt_num(likes)}")
    if params > 0:        stats.append(f"{params:.1f}B")
    if chips > 1:         stats.append(f"{chips}×chip")
    stats_str = "  " + "  ".join(stats) if stats else ""

    mid_col = 48
    task_col = 28
    line = (f"  {color}{glyph}{_RST}  "
            f"{_WHT}{mid:<{mid_col}}{_RST}"
            f"  {_DIM}{task:<{task_col}}{_RST}"
            f"{_DIM}{stats_str}{_RST}")
    print(line)


def _dir_bytes(path: Path) -> int:
    """Return the on-disk byte size of a directory via `du -sb` (fast, OS-level).

    Returns 0 if the path does not exist or `du` fails.  This is intentionally
    a lighter-weight call than _hf_cache_gb() — it targets a single model's
    cache subdirectory rather than the entire HF hub root.
    """
    if not path.exists():
        return 0
    try:
        r = subprocess.run(["du", "-sb", str(path)],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return int(r.stdout.split()[0])
    except Exception:
        pass
    return 0


def _with_spinner(msg: str, fn, *args, **kwargs):
    """Run *fn* in the foreground while displaying a braille spinner in the terminal.

    The spinner runs in a daemon thread so the user sees animated feedback even
    though the main thread is blocked in *fn*.  The progress line is fully
    erased before returning so subsequent output starts on a clean line.

    Returns *fn*'s return value unchanged.
    """
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    stop = threading.Event()
    state = [0]

    def _spin():
        while not stop.wait(0.1):
            f = frames[state[0] % len(frames)]
            sys.stdout.write(f"\r  {_TEAL}{f}{_RST}  {_DIM}{msg}{_RST}")
            sys.stdout.flush()
            state[0] += 1

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        stop.set()
        t.join()
        sys.stdout.write(f"\r{' ' * _tw()}\r")
        sys.stdout.flush()
    return result


# ── Queue building ────────────────────────────────────────────────────────────

def _scan_forge_models(bestiary_compiled_ids: set[str], include_all: bool = False,
                       framework: str | None = None) -> list[dict]:
    """
    Walk ~/code/tt-forge-models and return QueueItem dicts for loaders.

    Each loader.py under the forge-models tree is expected to define a
    ForgeModel subclass.  We reflect into each module to extract the class
    name and the model's task/source metadata, then build a minimal queue
    item dict that expedition_worker.py can consume.

    Args:
        bestiary_compiled_ids: Set of model IDs already in the bestiary.
            Models in this set are normally skipped.
        include_all: If True, include ALL seed models regardless of bestiary
            status — useful for regression testing after a forge update
            (--staples mode).
    """
    forge_models_root = Path.home() / "code" / "tt-forge-models"
    if not forge_models_root.exists():
        return []

    import importlib
    import types

    # The forge-models loaders use relative imports that reach all the way up to
    # the root of the repo (e.g. `from ...base import ForgeModel`).  The directory
    # name "tt-forge-models" contains a hyphen, so it cannot be imported as a
    # normal Python package.  The workaround is to register a synthetic root
    # package ("_forgems") that points to the forge-models directory.  All child
    # packages then import via "_forgems.musicgen_small.pytorch.loader" etc., and
    # their relative imports resolve correctly against the synthetic root.
    _PKG = "_forgems"
    if _PKG not in sys.modules:
        root_mod = types.ModuleType(_PKG)
        root_mod.__path__ = [str(forge_models_root)]
        root_mod.__package__ = _PKG
        root_mod.__file__ = str(forge_models_root / "__init__.py")
        sys.modules[_PKG] = root_mod

    items = []

    # Suppress C-level fd-2 noise (XLA/CUDA factory warnings, protobuf
    # MessageFactory errors) that forge/TTNN modules emit during import.
    # Python's warnings.filterwarnings() doesn't reach these — only a raw
    # fd-2 redirect works.
    _saved_fd2 = os.dup(2)
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_null_fd, 2)
    os.close(_null_fd)

    try:
        for loader_py in sorted(forge_models_root.rglob("loader.py")):
            # Skip hidden directories and private packages (leading _ or .).
            # Filter on the *relative* path only — absolute path components such
            # as a home directory that starts with '.' would otherwise cause all
            # models to be silently skipped.
            rel = loader_py.relative_to(forge_models_root)
            if any(p.startswith("_") or p.startswith(".") for p in rel.parts):
                continue

            parts_lower = [p.lower() for p in rel.parts]

            # Skip ONNX loaders — they require an onnx_tmp_path argument that
            # the generic worker cannot supply.
            if "onnx" in parts_lower:
                continue

            # Framework filter: skip loaders that don't match the requested framework.
            # "pytorch" skips paths containing /jax/, "jax" skips paths without /jax/.
            if framework is not None:
                is_jax = "jax" in parts_lower
                if framework == "jax" and not is_jax:
                    continue
                if framework == "pytorch" and is_jax:
                    continue

            # model_id is the directory path relative to forge-models root, minus the
            # trailing "loader.py" segment — e.g. "facebook/bart-large-cnn".
            model_id = "/".join(rel.parts[:-1])

            if not include_all and model_id in bestiary_compiled_ids:
                continue

            # Import via the synthetic "_forgems" root package so that relative
            # imports in the loader (e.g. `from ...base import ForgeModel`) resolve
            # correctly against the forge-models root.
            module_path = _PKG + "." + ".".join(rel.parts[:-1]) + ".loader"

            try:
                mod = importlib.import_module(module_path)
                # ForgeModel is defined in the synthetic package's base module.
                ForgeModel = sys.modules[f"{_PKG}.base"].ForgeModel  # type: ignore[attr-defined]
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

            # Derive the library tag from the loader path so the router can send
            # JAX loaders to the XLA backend instead of forge.
            loader_lib = "jax" if "jax" in parts_lower else "pytorch"

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
                # library reflects the actual framework so the dispatch router can
                # send JAX loaders to XLA and pytorch loaders to forge.
                "library": loader_lib,
                "model_type": "",
                # Store the _forgems-prefixed module path so expedition_worker
                # can import the same way (synthetic root stays registered for
                # the lifetime of the process).
                "loader_module": module_path,
                "loader_class": cls_name,
                "is_frontier": False,
            })
    finally:
        # Restore fd 2 before doing anything else so subsequent prints are visible.
        os.dup2(_saved_fd2, 2)
        os.close(_saved_fd2)

    return items


def _scan_frontier(
    bestiary_compiled_ids: set[str],
    forge_model_ids: set[str],
    min_downloads: int = 50,
    min_likes: int = 1,
    max_dl_like_ratio: int = 300,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
    proven_authors: set[str] | None = None,
    library: str | None = "pytorch",
) -> list[dict]:
    """
    Query the HuggingFace frontier for models not yet in the bestiary and not
    already covered by the forge-models library.

    When ``proven_authors`` is provided and the main scan yields fewer than 8
    candidates, the results are supplemented with uncompiled models from those
    authors — authors whose models have already compiled successfully on TT
    hardware are statistically likely to produce more compilable work.

    Returns a list of queue item dicts, one per discovered FrontierModel.
    The loader_module and loader_class fields are None for frontier models —
    expedition_worker.py will build a dynamic loader at runtime using
    hf_discover.build_dynamic_loader().
    """
    from lib.expedition.hf_discover import discover_frontier, discover_from_authors
    models = discover_frontier(
        compiled_ids=bestiary_compiled_ids,
        known_model_ids=forge_model_ids,
        min_downloads=min_downloads,
        min_likes=min_likes,
        max_dl_like_ratio=max_dl_like_ratio,
        max_params_b=max_params_b,
        skip_gated=skip_gated,
        library=library,
    )

    # If the frontier scan is sparse, supplement with models from authors whose
    # submissions have already compiled.  These are treated as a separate pool
    # so they show up distinctly in the queue but apply the same loader path.
    if proven_authors and len(models) < 8:
        supplement = discover_from_authors(
            authors=list(proven_authors),
            compiled_ids=bestiary_compiled_ids,
            known_model_ids=forge_model_ids,
            skip_gated=skip_gated,
            library=library,
        )
        found_ids = {m.model_id for m in models}
        models.extend(m for m in supplement if m.model_id not in found_ids)
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
            "library": m.library,
            "model_type": m.model_type,
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
    limit: int = 0,
    min_downloads: int = 50,
    min_likes: int = 1,
    max_dl_like_ratio: int = 300,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
    staples: bool = False,
) -> list[list[dict]]:
    """
    Build per-chip model queues by merging forge-models seed items with HF
    frontier discoveries.

    The final list is interleaved so each chip gets a mix of familiar seed
    models (60 %) and fresh frontier targets (40 %).  Items are then
    distributed round-robin across chips.  Every model_id appears on at most
    one chip — the limit applies to the total unique model count, not per chip.

    Args:
        num_chips:  Number of Tenstorrent chips to distribute work across.
        seed_only:  If True, skip HuggingFace frontier discovery.
        frontier_only: If True, skip forge-models seed scan.
        limit:      If > 0, cap total unique models across all chips combined.
        staples:    If True, include tt-forge-models seed models even if they
                    have already been compiled (bypass bestiary filter).

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
        # Use _with_spinner so the terminal shows activity during the forge-models
        # scan (which involves importlib reflection and can take a few seconds).
        seed_items = _with_spinner("scanning tt-forge-models library…",
                                   _scan_forge_models, compiled_ids, staples)

        # Always include at least one seed model as a canary — a known-good
        # baseline that confirms the hardware/forge stack is healthy, and a
        # fast first-win to seed the scoreboard.  If all seed models are
        # already in the bestiary (and --staples wasn't set), pull one at
        # random from the full zoo ignoring the compiled filter.
        if not seed_items:
            all_seeds = _scan_forge_models(set(), include_all=True)
            if all_seeds:
                import random as _random
                canary = _random.choice(all_seeds)
                canary = dict(canary)   # copy so we can annotate safely
                seed_items = [canary]

        _section(f"FORGE MODELS  ({len(seed_items)} seed)")
        for item in seed_items:
            _model_row(item)

    forge_ids = {item["model_id"] for item in seed_items}

    if not seed_only:
        # Authors whose models have already compiled — used to supplement the
        # frontier scan when the recent HF window is sparse.
        proven_authors = {mid.split("/")[0] for mid in bestiary.compiled
                          if "/" in mid}

        # HF frontier discovery involves network calls — spinner keeps the user
        # informed while we wait for the HuggingFace API to respond.
        frontier_items = _with_spinner("querying HuggingFace frontier…",
                                       _scan_frontier,
                                       compiled_ids, forge_ids,
                                       min_downloads=min_downloads,
                                       min_likes=min_likes,
                                       max_dl_like_ratio=max_dl_like_ratio,
                                       max_params_b=max_params_b,
                                       skip_gated=skip_gated,
                                       proven_authors=proven_authors)

        # Exclude models whose failure history shows they cannot succeed.
        # Three buckets of permanent failure:
        #   unsupported_arch     — architecture not in installed Transformers; won't load
        #   loader_missing       — build_dynamic_loader() can't trace this pipeline type
        #   missing_dependency   — required optional package (mamba-ssm etc.) not installed
        # Plus a catch-all: 3+ attempts of any error OTHER than tracer_output_type.
        # tracer_output_type is exempted because the new _LogitsWrapper retry in
        # _compile_model is designed to fix exactly that class of failure — models
        # that previously racked up tracer errors deserve a fresh shot.
        _PERM_FAIL_CATS = {"unsupported_arch", "loader_missing", "missing_dependency"}
        perm_fail_ids = {
            mid for mid, info in bestiary.failed.items()
            if info.get("error_category") in _PERM_FAIL_CATS
            or (info.get("attempts", 0) >= 3
                and info.get("error_category") != "tracer_output_type")
        }
        n_perm_skipped = 0
        if perm_fail_ids:
            n_before = len(frontier_items)
            frontier_items = [it for it in frontier_items
                              if it["model_id"] not in perm_fail_ids]
            n_perm_skipped = n_before - len(frontier_items)

        # Author/family dedup: one model per (author, repo-family) per run.
        # Keeps the most recent or best-sized variant; skips the rest.
        # Bypassed for large runs (100+ per chip) where diversity is expected.
        family_note = ""
        if limit == 0 or limit < 100:
            frontier_items, n_dropped = _dedup_by_author_family(
                frontier_items, target_params_b=max_params_b
            )
            notes = []
            if n_dropped:       notes.append(f"{n_dropped} family dupes dropped")
            if n_perm_skipped:  notes.append(f"{n_perm_skipped} perm-failed excluded")
            if notes:
                family_note = f"  {_DIM}({', '.join(notes)}){_RST}"

        _section(f"HF FRONTIER  ({len(frontier_items)} selected){family_note}")
        for item in frontier_items:
            _model_row(item)

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
    all_items = deduped
    # Note: Deduplicate count and total queue size are surfaced in the QUEUE
    # ASSIGNMENT section printed by main() — no low-signal print here.

    # Apply total model cap before distribution so --limit N always means
    # exactly N unique models across the whole run, regardless of chip count.
    if limit > 0:
        all_items = all_items[:limit]

    # Round-robin distribution across chips.
    chip_queues: list[list[dict]] = [[] for _ in range(num_chips)]
    for i, item in enumerate(all_items):
        chip_queues[i % num_chips].append(item)

    return chip_queues


# Matches one trailing qualifier token: numeric versions (-7B, -v0.1, -2.1),
# size labels (-base, -large, -medium, -xl), or variant labels (-instruct, -chat).
# Applied iteratively to collapse model families across naming conventions,
# e.g. gpt2 / gpt2-medium / gpt2-large → "gpt2";
#      Mistral-7B-Instruct-v0.2 → "mistral".
_FAMILY_SUFFIX_RE = re.compile(
    r'[-_](?:v?\d+(?:[._]\d+)*'
    r'|base|small|tiny|mini|nano|micro|medium|large|xl|xxl|huge|big'
    r'|lite|light|instruct|chat|hf|uncased|cased|multilingual)$',
    re.IGNORECASE,
)


def _family_key(repo_name: str) -> str:
    """Strip trailing qualifier tokens from a repo name (lowercase).

    Applies the regex up to 3 times to handle compound suffixes such as
    "-7B-Instruct-v0.2" → (strip -v0.2) → (strip -Instruct) → (strip -7B).

    "gpt2-medium"             → "gpt2"
    "Mistral-7B-Instruct-v0.2"→ "mistral"
    "bert-base-uncased"       → "bert"
    "icl-pruning-wanda-sparsity-0.5" → "icl-pruning-wanda-sparsity"
    """
    name = repo_name.lower()
    for _ in range(3):
        stripped = _FAMILY_SUFFIX_RE.sub('', name)
        if stripped == name:
            break
        name = stripped
    return name or repo_name.lower()


def _dedup_by_author_family(
    items: list[dict],
    target_params_b: float = 0.0,
) -> tuple[list[dict], int]:
    """Keep one model per (author, family) group; return (selected, n_dropped).

    Groups by the author prefix and a family key derived by stripping the
    trailing numeric/version token from the repo name.  Within each group the
    winner is chosen by:
      1. Most recent creation date (expedition favours zero-days).
      2. Param count closest to 70 % of target_params_b sweet-spot (or
         smallest params when no target — safer for single-chip).
      3. Most downloads as a final tiebreak.

    Original discovery order (newest-first from HF API) is preserved in the
    returned list so priority stays intact for the interleave / round-robin.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        author, _, repo = item["model_id"].partition("/")
        key = (author.lower(), _family_key(repo))
        groups.setdefault(key, []).append(item)

    order_map = {item["model_id"]: i for i, item in enumerate(items)}
    sweet = target_params_b * 0.7 if target_params_b > 0 else 0.0

    selected: list[dict] = []
    for group in groups.values():
        if len(group) == 1:
            selected.append(group[0])
            continue

        def _score(item, _sweet=sweet):
            created  = item.get("hf_created_at") or ""
            params   = item.get("hf_params_b") or 0.0
            size_sc  = (-abs(params - _sweet) if _sweet > 0 and params > 0
                        else -params)          # smaller = safer fallback
            return (created, size_sc, item.get("hf_downloads") or 0)

        selected.append(max(group, key=_score))

    selected.sort(key=lambda x: order_map.get(x["model_id"], 0))
    return selected, len(items) - len(selected)


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

# Patterns to ignore when pre-downloading for forge-onnx backend.
# Forge only needs PyTorch safetensors — skip Flax/TF/Keras formats.
_FORGE_IGNORE_PATTERNS = [
    "*.msgpack",   # Flax/JAX checkpoints (not needed by forge)
    "flax_model*", # Flax model shards
    "*.h5",        # Keras/TF HDF5 weights
    "tf_model*",   # TensorFlow SavedModel
    "rust_model*", # Rust/candle weights
    "*.ot",        # OpenNMT tokenizer files
]

# Patterns to ignore when pre-downloading for tt-xla backend.
# XLA needs Flax weights (.msgpack) — skip TF/Keras/Rust/PyTorch-only formats.
_XLA_IGNORE_PATTERNS = [
    "*.h5",        # Keras/TF HDF5 weights
    "tf_model*",   # TensorFlow SavedModel
    "rust_model*", # Rust/candle weights
    "*.ot",        # OpenNMT tokenizer files
]

# Default patterns for CLI paths that don't route per-model (conservative: forge-safe).
_IGNORE_PATTERNS = _FORGE_IGNORE_PATTERNS


def _run_parallel_downloads(
    items_ordered: list[tuple[str, dict]],
    max_workers: int,
    cache_root: Path,
    baseline_gb: float,
    max_cache_gb: float,
    session_download_max_gb: float,
) -> tuple[int, int, int]:
    """Download models in parallel with a live multi-slot progress display.

    Maintains N concurrent download slots (N = min(max_workers, len(items))).
    A single display thread redraws all slots in-place every 0.3 s using ANSI
    cursor-up sequences.  Each worker thread owns a fixed slot for its lifetime
    (thread-local slot assignment) so slot IDs stay stable even as the thread
    pool cycles through the model queue.

    Disk-limit checks run after each completion via as_completed().  When a
    limit fires, pending (not-yet-started) futures are cancelled immediately;
    already-running downloads complete normally.

    Returns (ok_count, fail_count, skipped_count).
    """
    from concurrent.futures import CancelledError as _FutureCancelled
    from huggingface_hub import snapshot_download, HfApi

    # Silence the huggingface_hub Python logger for the duration of the
    # download block — otherwise its INFO messages ("Downloading…", "Fetching…")
    # land on stdout and tear up the ANSI slot display.
    _hf_log  = _logging.getLogger("huggingface_hub")
    _hf_prev = _hf_log.level
    _hf_log.setLevel(_logging.ERROR)
    # Also silence filelock and requests noise that leaks through urllib3.
    for _noisy in ("filelock", "urllib3", "requests"):
        _logging.getLogger(_noisy).setLevel(_logging.ERROR)

    total   = len(items_ordered)
    n_slots = min(max_workers, total)

    # ── Shared slot state ─────────────────────────────────────────────────────
    # Each slot is a dict or None (idle).  Workers update their slot; the
    # display thread reads it.  slot_lock guards all access.
    slots: dict[int, dict | None] = {i: None for i in range(n_slots)}
    slot_lock = threading.Lock()

    # Thread-local slot assignment: each thread in the pool keeps the same slot
    # ID across all the models it processes (pool threads are reused).
    _thread_slot: dict[int, int] = {}
    _next_slot    = [0]
    _ts_lock      = threading.Lock()

    def _claim_slot() -> int:
        tid = threading.current_thread().ident
        with _ts_lock:
            if tid not in _thread_slot:
                _thread_slot[tid] = _next_slot[0]
                _next_slot[0]    += 1
        return _thread_slot[tid]

    # ── Display helpers ───────────────────────────────────────────────────────

    def _fmt_slot(s: dict | None) -> str:
        """Render one slot as a single terminal line (no trailing newline)."""
        if s is None:
            return f"  {_GRAY}——{_RST}"
        mid = s.get("model_id", "?")
        mid_short = mid if len(mid) <= 42 else mid[:39] + "…"
        if s.get("done"):
            if s.get("cached"):
                return f"  {_GRN}✓{_RST}  {_DIM}{mid_short}  (cached){_RST}"
            if s.get("ok"):
                nb  = s.get("new_bytes", 0)
                el  = s.get("elapsed", 0)
                sp  = nb / el if el > 0.1 else 0
                sp_s = f"  {_fmt_speed(sp)}" if sp > 50_000 else ""
                return (f"  {_GRN}✓{_RST}  {mid_short}"
                        f"  {_DIM}{_fmt_bytes(nb)}  {_fmt_eta(el)}{sp_s}{_RST}")
            err = s.get("err", "")[:52]
            return f"  {_RED}✗{_RST}  {_DIM}{mid_short}  {err}{_RST}"
        # Active download — render progress bar.
        # Layout: bar  pct  eta  bytes/total  speed  dim:model-id
        # ETA follows % immediately so "61%  3m24s" reads as one thought.
        ratio = s.get("ratio", 0.0)
        nb    = s.get("new_bytes", 0)
        tb    = s.get("total_bytes", 0)
        speed = s.get("speed", 0.0)
        eta   = s.get("eta", -1.0)
        pct   = f" {_BOLD}{ratio * 100:3.0f}%{_RST}" if tb > 0 else ""
        et_s  = f"  {_LTEAL}{_fmt_eta(eta)}{_RST}"   if eta > 5      else ""
        sz    = (f"  {_fmt_bytes(nb)}/{_fmt_bytes(tb)}" if tb > 0
                 else (f"  {_fmt_bytes(nb)}" if nb > 0 else ""))
        sp_s  = f"  {_DIM}{_fmt_speed(speed)}{_RST}"  if speed > 50_000 else ""
        bar   = _bar(ratio, width=20)
        return f"  {bar}{pct}{et_s}{sz}{sp_s}  {_DIM}{mid_short}{_RST}"

    # Reserve n_slots + 1 lines (blank separator + one per slot).
    print()
    for _ in range(n_slots):
        print()
    display_height = n_slots + 1

    stop_display = threading.Event()

    def _display_loop():
        while not stop_display.is_set():
            with slot_lock:
                snap = {k: (dict(v) if v else None) for k, v in slots.items()}
            buf = [f"\033[{display_height}A", "\033[2K\n"]
            for i in range(n_slots):
                buf.append(f"\033[2K{_fmt_slot(snap.get(i))}\n")
            sys.stdout.write("".join(buf))
            sys.stdout.flush()
            time.sleep(0.3)

    disp = threading.Thread(target=_display_loop, daemon=True)
    disp.start()

    # ── Worker ────────────────────────────────────────────────────────────────

    prev_hf_bar  = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    prev_hf_verb = os.environ.get("HF_HUB_VERBOSITY")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"

    def _worker(model_id: str, item: dict) -> dict:
        sid = _claim_slot()

        # Query expected size from the HF manifest for accurate progress ratio.
        expected_bytes = 0
        try:
            info = HfApi().model_info(model_id, timeout=15)
            for sib in (info.siblings or []):
                sz = getattr(sib, "size", None)
                if sz and not any(_fnmatch.fnmatch(sib.rfilename, p)
                                  for p in _IGNORE_PATTERNS):
                    expected_bytes += sz
        except Exception:
            pass

        cache_key = "models--" + model_id.replace("/", "--")
        model_dir = HF_CACHE_DIR / cache_key
        pre_bytes = _dir_bytes(model_dir)

        with slot_lock:
            slots[sid] = {"model_id": model_id, "ratio": 0.0, "new_bytes": 0,
                          "total_bytes": expected_bytes, "speed": 0.0, "eta": -1.0,
                          "done": False, "ok": True, "err": "", "cached": False,
                          "elapsed": 0}

        stop_mon = threading.Event()
        t0       = time.monotonic()

        def _mon():
            prev_b, prev_t = pre_bytes, t0
            while not stop_mon.wait(0.5):
                cur  = _dir_bytes(model_dir)
                now  = time.monotonic()
                dt   = now - prev_t
                inc  = cur - prev_b
                spd  = inc / dt if dt > 0.2 and inc > 0 else 0
                prev_b, prev_t = cur, now
                nb   = cur - pre_bytes
                rat  = min(nb / expected_bytes, 1.0) if expected_bytes > 0 else 0.0
                eta  = ((expected_bytes - nb) / spd
                        if spd > 50_000 and expected_bytes > 0 else -1.0)
                with slot_lock:
                    st = slots.get(sid)
                    if st and not st.get("done"):
                        st.update({"ratio": rat, "new_bytes": nb,
                                   "speed": spd, "eta": eta})

        mon = threading.Thread(target=_mon, daemon=True)
        mon.start()

        ok, err = True, ""
        try:
            snapshot_download(model_id, ignore_patterns=_IGNORE_PATTERNS,
                              local_files_only=False)
        except Exception as e:
            ok, err = False, str(e)[:80]
        finally:
            stop_mon.set()
            mon.join(timeout=1.0)

        elapsed   = time.monotonic() - t0
        new_bytes = _dir_bytes(model_dir) - pre_bytes
        cached    = ok and elapsed < 2.0 and new_bytes < 1_000

        with slot_lock:
            prev_ratio = (slots[sid].get("ratio", 0.0)
                          if slots.get(sid) else 0.0)
            slots[sid] = {"model_id": model_id, "done": True, "ok": ok,
                          "err": err, "cached": cached, "elapsed": elapsed,
                          "new_bytes": new_bytes, "total_bytes": expected_bytes,
                          "ratio": 1.0 if ok else prev_ratio,
                          "speed": 0.0, "eta": -1.0}

        return {"model_id": model_id, "ok": ok, "err": err,
                "cached": cached, "elapsed": elapsed, "new_bytes": new_bytes}

    # ── Execution with disk-limit checks ─────────────────────────────────────

    ok_c = fail_c = skip_c = 0
    stop_reason: str | None = None
    all_futures: list = []

    # Manage the executor manually (not via `with`) so KeyboardInterrupt can
    # call shutdown(wait=False) and return immediately rather than blocking
    # until every in-flight download thread finishes.
    executor = ThreadPoolExecutor(max_workers=n_slots)
    try:
        future_map = {executor.submit(_worker, mid, item): mid
                      for mid, item in items_ordered}
        all_futures = list(future_map)

        for fut in as_completed(all_futures):
            try:
                res = fut.result()
            except _FutureCancelled:
                continue   # already counted in skip_c via cancel loop below
            except Exception as e:
                res = {"model_id": future_map[fut], "ok": False,
                       "err": str(e)[:80], "cached": False,
                       "elapsed": 0, "new_bytes": 0}

            if res["ok"]:
                ok_c += 1
            else:
                fail_c += 1

            if stop_reason:
                continue  # let running futures finish; skip checks

            current_gb = _hf_cache_gb()
            session_gb = current_gb - baseline_gb
            free_gb    = _shutil.disk_usage(cache_root).free / 1e9

            if max_cache_gb > 0 and current_gb >= max_cache_gb:
                stop_reason = f"cache at {current_gb:.1f}/{max_cache_gb:.0f} GB"
            elif (session_download_max_gb > 0
                  and session_gb >= session_download_max_gb):
                stop_reason = (f"session cap "
                               f"{session_gb:.1f}/{session_download_max_gb:.0f} GB")
            elif free_gb < 5.0:
                stop_reason = f"disk critically low ({free_gb:.1f} GB free)"

            if stop_reason:
                for f in all_futures:
                    if not f.done() and f.cancel():
                        skip_c += 1

    except KeyboardInterrupt:
        # Cancel every future we can; release the executor without waiting for
        # in-flight threads — they are daemon threads and will die on exit.
        for f in all_futures:
            f.cancel()
        executor.shutdown(wait=False)
        stop_display.set()
        disp.join(timeout=0.5)
        print(f"\n{_RED}⚠  download interrupted{_RST}")
        raise  # propagate so main() can exit cleanly

    finally:
        executor.shutdown(wait=False)
        if prev_hf_bar is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_bar
        if prev_hf_verb is None:
            os.environ.pop("HF_HUB_VERBOSITY", None)
        else:
            os.environ["HF_HUB_VERBOSITY"] = prev_hf_verb
        _hf_log.setLevel(_hf_prev)

    # ── Final display repaint ─────────────────────────────────────────────────
    stop_display.set()
    disp.join(timeout=1.0)

    with slot_lock:
        snap = {k: (dict(v) if v else None) for k, v in slots.items()}
    buf = [f"\033[{display_height}A", "\033[2K\n"]
    for i in range(n_slots):
        buf.append(f"\033[2K{_fmt_slot(snap.get(i))}\n")
    sys.stdout.write("".join(buf))
    sys.stdout.flush()

    if stop_reason:
        print(f"\n  {_GOLD}Stopped: {stop_reason} — {skip_c} deferred{_RST}")

    return ok_c, fail_c, skip_c


def _predownload_queues(chip_queues: list[list[dict]], max_cache_gb: float = 0.0,
                        session_download_max_gb: float = 0.0,
                        parallel_downloads: int = 4) -> None:
    """Pre-fetch HuggingFace weights for all frontier models before compile starts.

    Collects unique frontier model IDs across all chip queues and calls
    snapshot_download so that weights land in the local HF cache.  This puts
    all chips on equal footing at compile time — none stalls waiting for a
    download that others finished earlier.

    Downloads run in parallel (up to parallel_downloads concurrent) with a
    live multi-slot progress display.  Forge-model seed entries skip this step;
    their weights are pulled by the loader's own from_pretrained call.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: F401 — availability check
    except ImportError:
        print(f"  {_RED}huggingface_hub not available — skipping pre-download{_RST}")
        return

    # Collect unique frontier IDs across all chip queues.
    # Dict (model_id → item) so we keep metadata for size hints.
    seen: dict[str, dict] = {}
    for queue in chip_queues:
        for item in queue:
            if item.get("is_frontier") and item["model_id"] not in seen:
                seen[item["model_id"]] = item

    if not seen:
        print(f"  {_DIM}No frontier models to pre-download.{_RST}")
        return

    cache_root  = HF_CACHE_DIR.parent if HF_CACHE_DIR.exists() else Path.home()
    baseline_gb = _hf_cache_gb()

    # Report active limits before starting downloads.
    limit_parts = [f"cache {baseline_gb:.1f} GB used"]
    if max_cache_gb > 0:
        limit_parts.append(f"cap {max_cache_gb:.0f} GB")
        if baseline_gb >= max_cache_gb:
            print(f"  {_GOLD}Cache already at limit — skipping pre-download.{_RST}")
            return
    if session_download_max_gb > 0:
        limit_parts.append(f"session ≤ {session_download_max_gb:.0f} GB new")
    n_parallel = min(parallel_downloads, len(seen))
    limit_parts.append(f"{n_parallel}× parallel")
    print(f"  {_DIM}{' · '.join(limit_parts)}{_RST}")

    ok, fail, skipped = _run_parallel_downloads(
        items_ordered=list(seen.items()),
        max_workers=parallel_downloads,
        cache_root=cache_root,
        baseline_gb=baseline_gb,
        max_cache_gb=max_cache_gb,
        session_download_max_gb=session_download_max_gb,
    )

    # Final tally line with color-coded status pills.
    final_session_gb = _hf_cache_gb() - baseline_gb
    parts = [f"{_GRN}✓ {ok} downloaded{_RST}"]
    if fail:    parts.append(f"{_RED}✗ {fail} failed{_RST}")
    if skipped: parts.append(f"{_GOLD}⏭ {skipped} deferred{_RST}")
    if final_session_gb > 0.1: parts.append(f"{_DIM}+{final_session_gb:.1f} GB fetched{_RST}")
    print(f"\n  {'  '.join(parts)}")


# ── Run summary ──────────────────────────────────────────────────────────────

def _print_run_summary(num_chips: int, run_number: int) -> None:
    """
    Aggregate end-of-run summary printed to the launching terminal after tmux exits.

    Reads per-chip CSV result files written by expedition_worker.py from /tmp,
    ranks chips by total points, lists new bestiary entries (first-evers), and
    prints a failure table with full (untruncated) error text via rich.
    Writes a compact run JSON to data/runs/.
    """
    import csv
    from rich.console import Console
    from rich.rule import Rule
    from rich.text import Text
    from lib.expedition.bestiary import Bestiary

    con = Console(highlight=False)

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

    # Guard: no CSV files found — workers may not have completed.
    if not chip_results:
        con.print()
        con.rule(f"[bold red]EXPEDITION #{run_number:03d} — NO RESULTS[/]", style="red")
        con.print("  [dim]No per-chip CSV files found in /tmp.[/]")
        con.print("  [dim]Workers may not have completed. Check /tmp/expedition_results_chip*.csv[/]")
        return

    medals = ["🥇", "🥈", "🥉", "  "]
    con.print()
    con.rule(f"[bold cyan]⚡ EXPEDITION #{run_number:03d} COMPLETE[/]", style="cyan")
    for i, c in enumerate(chip_results):
        medal = medals[min(i, 3)]
        fe = len(c["first_evers"])
        fe_str = f"  [gold1]★{fe} first-evers[/]" if fe else ""
        con.print(
            f"  {medal} CHIP {c['chip_id']}  "
            f"[gold1]{c['pts']:,} pts[/]  "
            f"[green]✓{len(c['successes'])}[/] [red]✗{len(c['failures'])}[/]"
            f"{fe_str}"
        )

    all_first_evers = [r for c in chip_results for r in c["first_evers"]]
    if all_first_evers:
        con.print()
        con.rule("[bold gold1]NEW TO BESTIARY[/]", style="gold1")
        for r in all_first_evers:
            artifact = r.get("artifact") or ""
            con.print(f"  [gold1]★[/] [bold]{r['model']}[/]")
            if artifact:
                con.print(f"    [dim]{artifact}[/]")

    all_failures = [r for c in chip_results for r in c["failures"]]
    if all_failures:
        from lib.expedition.bestiary import _classify_error
        from collections import Counter
        cat_counts: Counter = Counter()
        cat_labels: dict[str, str] = {}
        cat_hints:  dict[str, str] = {}
        for r in all_failures:
            key, label, hint = _classify_error(r.get("error") or "")
            cat_counts[key] += 1
            cat_labels[key] = label
            cat_hints[key]  = hint
        con.print()
        con.rule("[bold red]FAILED THIS RUN[/]", style="red")
        for r in all_failures:
            err = r.get("error") or ""
            key, label, _hint = _classify_error(err)
            con.print(f"  [red]✗[/] [bold]{r['model']}[/]  [dim italic]{label}[/]")
            # Print full error text, each line indented, no truncation.
            for eline in err.splitlines():
                con.print(f"    [dim]{eline}[/]")
        con.print()
        con.print("  [bold]Failure reasons:[/]")
        for key, cnt in cat_counts.most_common():
            con.print(f"    [bold]{cnt:>3}[/]  [cyan]{cat_labels[key]:<28}[/]  [dim]{cat_hints[key]}[/]")

    # Final bestiary headcount — Bestiary only takes path, no runs_dir.
    b = Bestiary(path=str(BESTIARY_PATH))
    compiled_count = len(b.compiled)
    con.print()
    con.rule(style="cyan")
    con.print(f"  [cyan]BESTIARY:[/] {compiled_count} total compiled")
    con.print()

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

def _print_failure_reasons(stats: list[dict], *, W: int = 72, header: str = "FAILURE REASONS") -> None:
    """Print a ranked failure-reason table from Bestiary.failure_stats() output."""
    if not stats:
        return
    total = sum(s["count"] for s in stats)
    print(f"\n{'─'*W}")
    print(f"  {header}  ({total} total failures, {sum(1 for _ in stats)} categories)")
    print(f"  {'#':>4}  {'category':<24}  {'label':<28}  hint")
    print(f"  {'─'*4}  {'─'*24}  {'─'*28}  {'─'*18}")
    for s in stats:
        # Truncate label/hint so they don't wrap on narrow terminals.
        label = s["label"][:27]
        hint  = s["hint"][:40]
        print(f"  {s['count']:>4}  {s['key']:<24}  {label:<28}  {hint}")

    # Top-offender authors: which HF accounts contribute the most failures?
    from collections import Counter
    author_counts: Counter = Counter()
    for s in stats:
        for mid in s["models"]:
            author = mid.split("/")[0] if "/" in mid else mid
            author_counts[author] += 1
    top = author_counts.most_common(5)
    if top and top[0][1] > 1:
        print(f"\n  Top authors by failure count:")
        for author, cnt in top:
            print(f"    {cnt:>4}  {author}")


def cmd_summary():
    """
    Print a human-readable snapshot of the expedition bestiary: total
    compiled, total failed, chip hall-of-fame, and the failure-reason leaderboard.
    """
    from lib.expedition.bestiary import Bestiary
    b = Bestiary(path=str(BESTIARY_PATH))
    compiled = b.compiled
    failed   = b.failed
    totals   = b.chip_totals

    W = 72
    print(f"\n{'═'*W}")
    print(f"  EXPEDITION BESTIARY")
    print(f"{'═'*W}")
    print(f"  Compiled:  {len(compiled)} models")
    print(f"  Failed:    {len(failed)} models")
    if totals:
        print(f"\n  Chip Hall of Fame:")
        for chip_id, data in sorted(totals.items(), key=lambda x: -x[1]["pts"]):
            print(f"    Chip {chip_id}: {data['pts']:,} pts  "
                  f"★{data['first_evers']} first-evers  "
                  f"best streak ×{data['best_streak']}")

    _print_failure_reasons(b.failure_stats(), W=W)
    print(f"\n{'═'*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Expedition Mode — roguelike forge compilation")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Start an expedition run (default)")
    run_p.add_argument("--chips",          type=int, default=0,
                       help="Number of chips (0=auto-detect)")
    run_p.add_argument("--limit",          type=int, default=0,
                       help="Total unique models across all chips (0=unlimited)")
    run_p.add_argument("--seed-only",        action="store_true")
    run_p.add_argument("--frontier-only",    action="store_true")
    run_p.add_argument("--staples",          action="store_true",
                       help="Include tt-forge-models seed models even if already compiled (regression test mode)")
    run_p.add_argument("--backend",          choices=["auto", "forge", "xla", "mixed"], default="auto",
                       help="Compilation backend: auto (default, per-model routing), forge, xla (JAX/PJRT), or mixed (even chips=forge, odd chips=xla)")
    run_p.add_argument("--no-predownload",   action="store_true",
                       help="Skip pre-downloading HF weights (faster start, unequal footing)")
    run_p.add_argument("--monitor",          action="store_true",
                       help="Add a tt-smi hardware monitor pane in the center column")
    # ── Quality / reputation bar ──────────────────────────────────────────────
    run_p.add_argument("--min-downloads",    type=int,   default=50, metavar="N",
                       help="Skip frontier models with fewer than N total downloads "
                            "(default 50; try 1000 for proven models, 10000 for popular ones)")
    run_p.add_argument("--min-likes",        type=int,   default=1, metavar="N",
                       help="Skip frontier models with fewer than N HuggingFace likes "
                            "(default 1; 0 to disable)")
    run_p.add_argument("--max-dl-like-ratio", type=int,  default=300, metavar="R",
                       help="Skip frontier models where downloads/likes > R — bots inflate "
                            "download counts without generating likes (default 300; 0 to disable)")
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
    run_p.add_argument("--parallel-downloads",    type=int,   default=4, metavar="N",
                       help="Number of concurrent model downloads during pre-fetch "
                            "(default: 4; try 2 on slower connections)")
    run_p.add_argument("--tui",                  action="store_true",
                       help="Use the Textual TUI instead of the tmux session "
                            "(equal-sized chip panels, live combat log, roguelike scoring)")

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
        args.staples = False
        args.backend = "auto"
        args.no_predownload = False
        args.monitor = False
        args.tui = False
        args.min_downloads = 50
        args.min_likes = 1
        args.max_dl_like_ratio = 300
        args.max_model_params = 0.0
        args.allow_gated = False
        args.max_cache_gb = 0.0
        args.session_download_max = 0.0
        args.parallel_downloads = 4

    # ── Hardware detection ────────────────────────────────────────────────────
    from lib.hardware import detect_hardware, get_hardware_summary
    hw = detect_hardware()
    num_chips = args.chips if args.chips > 0 else hw.get("num_chips", 1)
    if num_chips == 0:
        print("No chips detected. Check tt-smi.")
        sys.exit(1)

    # ── Run numbering ─────────────────────────────────────────────────────────
    # Derive next run number from the count of existing run JSON files.
    # We do NOT call any Bestiary.next_run_number() — that method does not exist.
    # run_number is computed BEFORE the banner so _banner can display it.
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run_number = len(list(RUNS_DIR.glob("run_*.json"))) + 1

    # ── TUI path: hand off all params and let the TUI manage setup ───────────
    # SetupScreen handles discovery, queue building, and pre-download;
    # RunScreen handles chip workers; SummaryScreen replaces _print_run_summary.
    if getattr(args, "tui", False):
        from expedition_tui import ExpeditionTUI
        app = ExpeditionTUI(
            num_chips=num_chips,
            run_number=run_number,
            arch=hw.get("arch", "blackhole"),
            project_dir=PROJECT_DIR,
            limit=args.limit,
            seed_only=args.seed_only,
            frontier_only=args.frontier_only,
            staples=args.staples,
            backend=args.backend,
            # TUI workers download models on-demand; pre-downloading 40+ models
            # during setup would silently block for 30+ minutes with no progress
            # visible to the user.  Pass-through only if explicitly requested.
            no_predownload=True,
            min_downloads=args.min_downloads,
            min_likes=args.min_likes,
            max_dl_like_ratio=getattr(args, "max_dl_like_ratio", 300),
            max_params_b=args.max_model_params,
            allow_gated=args.allow_gated,
            max_cache_gb=args.max_cache_gb,
            session_download_max=args.session_download_max,
            parallel_downloads=args.parallel_downloads,
        )
        app.run()
        return   # everything handled inside TUI screens

    # Banner goes here — after we know run_number and num_chips.
    _banner(run_number, num_chips, get_hardware_summary(hw))

    # ── Queue building ────────────────────────────────────────────────────────
    chip_queues = build_queues(
        num_chips=num_chips,
        seed_only=args.seed_only,
        frontier_only=args.frontier_only,
        limit=args.limit,
        min_downloads=args.min_downloads,
        min_likes=args.min_likes,
        max_dl_like_ratio=args.max_dl_like_ratio,
        max_params_b=args.max_model_params,
        skip_gated=not args.allow_gated,
        staples=args.staples,
    )

    # ── Queue assignment summary ──────────────────────────────────────────────
    # Print a compact per-chip breakdown so the user can verify distribution
    # before committing ~minutes of download time.
    _section("QUEUE ASSIGNMENT")
    for chip_id, queue in enumerate(chip_queues):
        seeds    = sum(1 for m in queue if not m.get("is_frontier"))
        frontier = sum(1 for m in queue if m.get("is_frontier"))
        chips_4  = sum(1 for m in queue if m.get("mesh_chips", 1) > 1)
        detail   = f"seed ×{seeds}  frontier ×{frontier}"
        if chips_4:
            detail += f"  mesh4 ×{chips_4}"
        print(f"  {_TEAL}C{chip_id}{_RST}  {_BOLD}{len(queue):>3} models{_RST}  {_DIM}{detail}{_RST}")

    # ── Pre-download weights for fairness ────────────────────────────────────
    if not args.no_predownload:
        frontier_total = sum(1 for q in chip_queues for m in q if m.get("is_frontier"))
        _section(f"PRE-DOWNLOAD  ({frontier_total} frontier models)")
        _predownload_queues(chip_queues, max_cache_gb=args.max_cache_gb,
                            session_download_max_gb=args.session_download_max,
                            parallel_downloads=args.parallel_downloads)

    # ── Write per-chip queue JSON to /tmp ─────────────────────────────────────
    # expedition_worker.py reads these files at startup.  Writing happens after
    # pre-download so workers always start with warm weights available.
    for chip_id, queue in enumerate(chip_queues):
        queue_path = f"/tmp/expedition_queue_chip{chip_id}.json"
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)

    # ── Launch UI ─────────────────────────────────────────────────────────────
    # TUI path returns early above; only the tmux path reaches here.
    _section("LAUNCHING")
    print(f"  {_DIM}tmux session:{_RST} {_TEAL}expedition{_RST}  "
          f"{_DIM}· reattach:{_RST}  {_TEAL}tmux attach -t expedition{_RST}")
    print()
    script = PROJECT_DIR / "scripts" / "run_expedition.sh"
    env = {**os.environ, "EXPEDITION_RUN": str(run_number),
           "EXPEDITION_NUM_CHIPS": str(num_chips)}
    cmd = ["bash", str(script), "--chips", str(num_chips),
           "--run", str(run_number)]
    if args.monitor:
        cmd.append("--monitor")
    subprocess.run(cmd, env=env)

    # ── Post-run aggregate summary ────────────────────────────────────────────
    # After tmux exits, gather per-chip CSV results and print the leaderboard.
    _print_run_summary(num_chips, run_number)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{_RED}interrupted{_RST}")
        # os._exit skips Python's atexit/threading shutdown, which would
        # otherwise block trying to join daemon download threads and print
        # a second KeyboardInterrupt traceback.
        os._exit(130)
