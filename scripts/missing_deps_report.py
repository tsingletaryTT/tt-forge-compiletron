#!/usr/bin/env python3
"""Report which Python packages are blocking the most models.

Usage:
    python3 scripts/missing_deps_report.py
    python3 scripts/missing_deps_report.py --json
    python3 scripts/missing_deps_report.py --bestiary path/to/bestiary.json

Reads data/bestiary.json (or --bestiary path) and prints a ranked table of
packages that appear in failed entries' missing_packages field.
"""
import argparse, json, pathlib, sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bestiary",
        default="data/bestiary.json",
        help="Path to bestiary.json (default: data/bestiary.json)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output machine-readable JSON instead of a table",
    )
    args = parser.parse_args()

    bestiary_path = pathlib.Path(args.bestiary)
    if not bestiary_path.exists():
        print(f"Error: {bestiary_path} not found", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from lib.expedition.bestiary import Bestiary
    b = Bestiary(path=bestiary_path)
    report = b.missing_dep_report()

    if not report:
        print("No missing_packages recorded in bestiary yet.")
        return

    if args.as_json:
        print(json.dumps(report, indent=2))
        return

    col1, col2, col3 = 24, 16, 50
    header = f"{'Package':<{col1}}  {'Models blocked':<{col2}}  {'Example models'}"
    print(header)
    print("─" * (col1 + col2 + col3 + 4))
    for row in report:
        examples = ", ".join(row["models"][:3])
        if len(row["models"]) > 3:
            examples += f", +{len(row['models']) - 3} more"
        print(f"{row['package']:<{col1}}  {row['count']:<{col2}}  {examples}")


if __name__ == "__main__":
    main()
