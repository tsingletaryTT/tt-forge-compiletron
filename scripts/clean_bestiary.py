#!/usr/bin/env python3
"""One-shot cleanup of stale harness-caused bestiary entries.

Run once after merging harness-hardening-envfix to evict entries that failed
for infrastructure reasons rather than forge/XLA limitations.

Clears:
  - Entries whose last_error contains "cats_image.jpeg"
  - Entries whose error_category == "wrong_backend"
  - Entries whose env_fingerprint differs from the current env on
    version-signal errors

Usage:
    python3 scripts/clean_bestiary.py           # live run
    python3 scripts/clean_bestiary.py --dry-run # preview only
    python3 scripts/clean_bestiary.py --bestiary path/to/bestiary.json
"""
import argparse, pathlib, sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be cleared without modifying the file")
    parser.add_argument("--bestiary", default="data/bestiary.json",
                        help="Path to bestiary.json (default: data/bestiary.json)")
    args = parser.parse_args()

    bestiary_path = pathlib.Path(args.bestiary)
    if not bestiary_path.exists():
        print(f"Error: {bestiary_path} not found", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from lib.expedition.bestiary import Bestiary, _current_env_fingerprint

    b = Bestiary(path=bestiary_path)
    current_fp = _current_env_fingerprint()

    cats_cleared = [] if args.dry_run else b.clear_entries_matching(error_contains="cats_image.jpeg")
    if args.dry_run:
        # Count without mutating
        cats_cleared = [
            mid for mid, entry in b.failed.items()
            if "cats_image.jpeg" in entry.get("last_error", "")
        ]

    wrong_backend_cleared = [
        mid for mid, entry in list(b.failed.items())
        if entry.get("error_category") == "wrong_backend"
    ]
    if not args.dry_run:
        for mid in wrong_backend_cleared:
            del b._data["failed"][mid]

    env_cleared = [] if args.dry_run else b.clear_stale_env_failures(current_fp)
    if args.dry_run:
        # Simulate what clear_stale_env_failures would do
        import re as _re
        _VERSION_SIGNAL = _re.compile(
            r"(version|require|found|expected|incompatible|>=|<=|!=|==)", _re.I
        )
        _ELIGIBLE = {"other", "api_mismatch", "missing_dependency", "unsupported_arch"}
        for mid, entry in b.failed.items():
            stored_fp = entry.get("env_fingerprint")
            if not stored_fp:
                continue
            if entry.get("error_category") not in _ELIGIBLE:
                continue
            err = entry.get("last_error", "")
            if not _VERSION_SIGNAL.search(err):
                continue
            for pkg, ver in current_fp.items():
                if stored_fp.get(pkg) != ver:
                    env_cleared.append(mid)
                    break

    total = len(cats_cleared) + len(wrong_backend_cleared) + len(env_cleared)

    print(f"cats_image.jpeg entries:  {len(cats_cleared)}")
    for mid in cats_cleared:
        print(f"  - {mid}")
    print(f"wrong_backend entries:    {len(wrong_backend_cleared)}")
    for mid in wrong_backend_cleared:
        print(f"  - {mid}")
    print(f"stale env entries:        {len(env_cleared)}")
    for mid in env_cleared:
        print(f"  - {mid}")
    print(f"\nTotal: {total} entries {'would be' if args.dry_run else ''} cleared")

    if not args.dry_run and total > 0:
        b.save()
        print("bestiary.json updated.")
    elif args.dry_run:
        print("(dry-run — no changes written)")


if __name__ == "__main__":
    main()
