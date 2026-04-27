#!/usr/bin/env python3
"""scripts/prune_models.py — HuggingFace model cache pruner

Scans the HF cache directory (default: ~/models) for model entries, queries
the HuggingFace Hub for quality metrics, and identifies candidates for removal.

Two sweep categories:

  Stubs    — cache directories that contain no downloaded weights (the HF
             downloader created the shell but never fetched blobs). These are
             always safe to remove regardless of model quality.

  Culls    — directories with real weights that fall below configurable
             quality thresholds (downloads, likes) or exceed a size cap.

Large models (>13 B params or >30 GB on disk) are shown in a separate
"manual review" bucket — they are never auto-deleted, even if they fail
a threshold, because they were most likely intentionally downloaded.

Usage:
  python3 scripts/prune_models.py [options]

Options:
  --cache-dir PATH        HF cache root   [default: ~/models or $HF_HOME]
  --dry-run               Show what would be removed; delete nothing
  --min-downloads N       Min HF download count to keep  [default: 1000]
  --min-likes N           Min HF likes to keep           [default: 5]
  --max-params-b B        Skip size check (no auto-cull by params alone)
                          Models over this many B params go to manual review
                          [default: 13]
  --large-gb-threshold G  Disk GB above which a model goes to manual review
                          [default: 30]
  --keep MODEL_ID         Always keep this model (repeatable, substring match)
  --no-api                Skip HF API calls; base decisions on disk size only
  --yes                   Skip per-category confirmation prompts (use carefully)

Always kept (hardcoded):
  AvaLovelace/LLaMA-ASCII-Art

Examples:
  # Preview everything that would be removed:
  python3 scripts/prune_models.py --dry-run

  # Lower thresholds and skip confirmation:
  python3 scripts/prune_models.py --min-downloads 500 --min-likes 3 --yes
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

# These are never removed, even if they fail every threshold.
_HARDCODED_KEEPS: set[str] = {
    "AvaLovelace/LLaMA-ASCII-Art",
    "AvaLovelace/LLaMA-ASCII-Art".lower(),
}

_GB = 1024 ** 3

# ── Path helpers ──────────────────────────────────────────────────────────────

def _default_cache_dir() -> Path:
    for env in ("HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return Path.home() / "models"


def _dir_to_model_id(name: str) -> Optional[str]:
    """Convert 'models--org--model-name' → 'org/model-name', or None."""
    if not name.startswith("models--"):
        return None
    parts = name[len("models--"):].split("--", 1)
    if len(parts) != 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
    except Exception:
        pass
    return total


def _fmt_gb(n_bytes: int) -> str:
    gb = n_bytes / _GB
    if gb < 0.01:
        return "< 0.01 GB"
    return f"{gb:.2f} GB"

# ── HF API query ──────────────────────────────────────────────────────────────

def _query_hf(model_id: str) -> dict:
    """Return dict with keys: downloads, likes, params_b, gated. Empty on error."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id, timeout=10)
        downloads = getattr(info, "downloads", 0) or 0
        likes = getattr(info, "likes", 0) or 0
        gated = bool(getattr(info, "gated", False))
        params_b = 0.0
        try:
            st = getattr(info, "safetensors", None)
            if st is not None:
                total = getattr(st, "total", None)
                if isinstance(total, (int, float)) and total > 0:
                    params_b = total / 1e9
        except Exception:
            pass
        return {"downloads": downloads, "likes": likes,
                "params_b": params_b, "gated": gated}
    except Exception as exc:
        return {"error": str(exc)}

# ── Core scan ─────────────────────────────────────────────────────────────────

def scan(
    cache_dir: Path,
    min_downloads: int,
    min_likes: int,
    max_params_b: float,
    large_gb_threshold: float,
    keep_patterns: list[str],
    use_api: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Returns four lists: (stubs, culls, manual_review, keepers)
    Each entry is a dict with keys relevant to its category.
    """
    stubs: list[dict] = []
    culls: list[dict] = []
    manual_review: list[dict] = []
    keepers: list[dict] = []

    entries = sorted(cache_dir.iterdir())
    model_dirs = [e for e in entries if e.is_dir() and e.name.startswith("models--")]

    print(f"\nScanning {len(model_dirs)} model cache directories in {cache_dir} …\n")

    for i, d in enumerate(model_dirs, 1):
        model_id = _dir_to_model_id(d.name)
        if model_id is None:
            continue

        sys.stdout.write(f"\r  [{i:3d}/{len(model_dirs)}] {model_id[:60]:<60}")
        sys.stdout.flush()

        size_bytes = _dir_size_bytes(d)
        size_gb = size_bytes / _GB
        is_stub = size_bytes < 10_000  # < 10 KB → no blobs downloaded

        # Always-keep check
        lower_id = model_id.lower()
        if (model_id in _HARDCODED_KEEPS or lower_id in _HARDCODED_KEEPS or
                any(p.lower() in lower_id for p in keep_patterns)):
            keepers.append({"model_id": model_id, "size_bytes": size_bytes,
                            "reason": "hardcoded keep"})
            continue

        if is_stub:
            stubs.append({"model_id": model_id, "path": d, "size_bytes": size_bytes})
            continue

        # Large disk footprint → manual review regardless of API result
        if size_gb >= large_gb_threshold:
            entry: dict = {"model_id": model_id, "path": d,
                           "size_bytes": size_bytes, "size_gb": size_gb}
            if use_api:
                entry.update(_query_hf(model_id))
            manual_review.append(entry)
            continue

        # Query HF for quality metrics
        hf: dict = {}
        if use_api:
            hf = _query_hf(model_id)

        downloads = hf.get("downloads", None)
        likes = hf.get("likes", None)
        params_b = hf.get("params_b", 0.0)
        gated = hf.get("gated", False)

        # Large by params → manual review
        if params_b > max_params_b:
            manual_review.append({
                "model_id": model_id, "path": d,
                "size_bytes": size_bytes, "size_gb": size_gb,
                "downloads": downloads, "likes": likes,
                "params_b": params_b, "gated": gated,
            })
            continue

        # Quality filters
        cull_reasons: list[str] = []
        if downloads is not None and downloads < min_downloads:
            cull_reasons.append(f"dl={downloads:,} < {min_downloads:,}")
        if likes is not None and likes < min_likes:
            cull_reasons.append(f"likes={likes} < {min_likes}")
        if not use_api:
            # Without API data, don't cull on quality — only stubs and large
            pass

        if cull_reasons:
            culls.append({
                "model_id": model_id, "path": d,
                "size_bytes": size_bytes, "size_gb": size_gb,
                "downloads": downloads, "likes": likes,
                "params_b": params_b, "gated": gated,
                "reasons": cull_reasons,
            })
        else:
            keepers.append({
                "model_id": model_id, "size_bytes": size_bytes,
                "downloads": downloads, "likes": likes,
                "params_b": params_b,
            })

    sys.stdout.write("\r" + " " * 80 + "\r")  # clear progress line
    return stubs, culls, manual_review, keepers

# ── Display ───────────────────────────────────────────────────────────────────

def _print_section(title: str, items: list[dict], show_reasons: bool = False) -> None:
    total_bytes = sum(e.get("size_bytes", 0) for e in items)
    print(f"\n{'─' * 70}")
    print(f"  {title}  ({len(items)} entries, {_fmt_gb(total_bytes)} total)")
    print(f"{'─' * 70}")
    for e in items:
        mid = e["model_id"]
        sz = _fmt_gb(e.get("size_bytes", 0))
        dl = e.get("downloads")
        lk = e.get("likes")
        pb = e.get("params_b", 0.0)
        meta = []
        if dl is not None: meta.append(f"dl={dl:,}")
        if lk is not None: meta.append(f"likes={lk}")
        if pb:             meta.append(f"{pb:.1f}B")
        meta_str = f"  [{', '.join(meta)}]" if meta else ""
        reason_str = ""
        if show_reasons and e.get("reasons"):
            reason_str = f"  ← {'; '.join(e['reasons'])}"
        print(f"    {mid:<55} {sz:>10}{meta_str}{reason_str}")


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"\n{prompt} [y/N] ").strip().lower()
        return ans == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def _delete_entries(entries: list[dict], dry_run: bool, label: str) -> int:
    freed = 0
    for e in entries:
        path: Path = e["path"]
        size = e.get("size_bytes", 0)
        if dry_run:
            print(f"    [dry-run] would remove {path}")
        else:
            try:
                shutil.rmtree(path)
                freed += size
                print(f"    removed {path.name}  ({_fmt_gb(size)})")
            except Exception as exc:
                print(f"    ERROR removing {path}: {exc}")
    return freed

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prune stale or low-quality entries from the HF model cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="HF cache root (default: ~/models or $HF_HOME)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview only — delete nothing")
    ap.add_argument("--min-downloads", type=int, default=1000,
                    metavar="N", help="Minimum HF downloads to keep [1000]")
    ap.add_argument("--min-likes", type=int, default=5,
                    metavar="N", help="Minimum HF likes to keep [5]")
    ap.add_argument("--max-params-b", type=float, default=13.0,
                    metavar="B", help="Params (B) above which → manual review [13]")
    ap.add_argument("--large-gb", type=float, default=30.0,
                    metavar="G", help="Disk GB above which → manual review [30]")
    ap.add_argument("--keep", action="append", default=[], metavar="MODEL_ID",
                    dest="keep_patterns",
                    help="Always keep (substring match, repeatable)")
    ap.add_argument("--no-api", action="store_true",
                    help="Skip HF API calls (stub detection only)")
    ap.add_argument("--yes", action="store_true",
                    help="Skip confirmation prompts")
    args = ap.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()
    if not cache_dir.is_dir():
        sys.exit(f"Cache directory not found: {cache_dir}")

    print(f"\n  HF Model Cache Pruner")
    print(f"  cache : {cache_dir}")
    print(f"  mode  : {'DRY RUN (nothing will be deleted)' if args.dry_run else 'LIVE'}")
    print(f"  thresholds : dl≥{args.min_downloads:,}  likes≥{args.min_likes}"
          f"  params≤{args.max_params_b}B  disk<{args.large_gb}GB (manual review above)")
    if not args.no_api:
        print(f"  HF API : enabled (this may take a few minutes)")
    else:
        print(f"  HF API : disabled (stub detection only)")

    stubs, culls, manual_review, keepers = scan(
        cache_dir=cache_dir,
        min_downloads=args.min_downloads,
        min_likes=args.min_likes,
        max_params_b=args.max_params_b,
        large_gb_threshold=args.large_gb,
        keep_patterns=args.keep_patterns,
        use_api=not args.no_api,
    )

    # ── Report ────────────────────────────────────────────────────────────────

    stub_bytes = sum(e.get("size_bytes", 0) for e in stubs)
    cull_bytes = sum(e.get("size_bytes", 0) for e in culls)
    review_bytes = sum(e.get("size_bytes", 0) for e in manual_review)
    keep_bytes = sum(e.get("size_bytes", 0) for e in keepers)

    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Stubs (empty dirs, safe to remove) : {len(stubs):3d}   {_fmt_gb(stub_bytes)}")
    print(f"  Low-quality (fail thresholds)       : {len(culls):3d}   {_fmt_gb(cull_bytes)}")
    print(f"  Manual review (large models)        : {len(manual_review):3d}   {_fmt_gb(review_bytes)}")
    print(f"  Keepers                             : {len(keepers):3d}   {_fmt_gb(keep_bytes)}")
    print(f"{'─' * 70}")
    print(f"  Reclaimable (stubs + low-quality)   :       {_fmt_gb(stub_bytes + cull_bytes)}")

    if stubs:
        _print_section("STUBS — empty cache dirs (no weights downloaded)", stubs)

    if culls:
        _print_section("LOW-QUALITY — fail thresholds", culls, show_reasons=True)

    if manual_review:
        _print_section("MANUAL REVIEW — large models (not auto-deleted)", manual_review)
        print("\n  ^ These are shown for awareness only. Run with --keep or just leave them.")

    if keepers:
        _print_section("KEEPERS — pass all filters", keepers)

    # ── Deletion ──────────────────────────────────────────────────────────────

    total_freed = 0

    if stubs:
        print(f"\n{'─' * 70}")
        if args.dry_run:
            print(f"  [dry-run] Would remove {len(stubs)} stub directories ({_fmt_gb(stub_bytes)})")
            _delete_entries(stubs, dry_run=True, label="stubs")
        elif args.yes or _confirm(
            f"Remove {len(stubs)} stub directories ({_fmt_gb(stub_bytes)})?"
        ):
            print()
            total_freed += _delete_entries(stubs, dry_run=False, label="stubs")

    if culls:
        print(f"\n{'─' * 70}")
        if args.dry_run:
            print(f"  [dry-run] Would remove {len(culls)} low-quality models ({_fmt_gb(cull_bytes)})")
            _delete_entries(culls, dry_run=True, label="culls")
        elif args.yes or _confirm(
            f"Remove {len(culls)} low-quality model(s) ({_fmt_gb(cull_bytes)})?"
        ):
            print()
            total_freed += _delete_entries(culls, dry_run=False, label="culls")

    if not args.dry_run and total_freed:
        print(f"\n  Total freed: {_fmt_gb(total_freed)}")
    elif args.dry_run:
        print(f"\n  (dry-run — nothing deleted; re-run without --dry-run to act)")

    print()


if __name__ == "__main__":
    main()
