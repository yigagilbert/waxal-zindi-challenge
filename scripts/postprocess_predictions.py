#!/usr/bin/env python3
"""Strip trailing CTC dot-collapse artifacts from a prediction CSV.

Removes trailing runs of >= --min-run dots (optionally keeping a single final
dot), collapses repeated whitespace, and leaves interior punctuation untouched.
Evaluate the cleaned file against validation references before using it on test.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import id_language, read_prediction_csv, write_csv_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["strip", "single-dot"],
        default="strip",
        help="strip removes the trailing dot run entirely; single-dot replaces it with one '.'",
    )
    parser.add_argument("--min-run", type=int, default=3, help="Minimum trailing dot-run length to clean.")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def clean_target(text: str, *, mode: str, min_run: int) -> str:
    trailing_dots = re.compile(r"(?:\s*\.){%d,}\s*$" % min_run)
    cleaned = trailing_dots.sub("." if mode == "single-dot" else "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "."


def main() -> None:
    args = parse_args()
    rows = read_prediction_csv(args.predictions)
    changed_by_language: Counter[str] = Counter()
    emptied: list[str] = []
    out_rows = []
    for row in rows:
        original = row.get("Target", "")
        cleaned = clean_target(original, mode=args.mode, min_run=args.min_run)
        if cleaned != original:
            changed_by_language[id_language(row["ID"])] += 1
            if cleaned == "." and original.strip(". ") == "":
                emptied.append(row["ID"])
        out_rows.append({"ID": row["ID"], "Target": cleaned})

    write_csv_rows(args.output, out_rows, ["ID", "Target"])
    report = {
        "input": str(args.predictions),
        "output": str(args.output),
        "mode": args.mode,
        "min_run": args.min_run,
        "rows": len(out_rows),
        "changed": sum(changed_by_language.values()),
        "changed_by_language": dict(sorted(changed_by_language.items())),
        "dot_only_rows_left_as_single_dot": len(emptied),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
