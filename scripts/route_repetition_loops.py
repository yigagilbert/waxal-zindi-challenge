#!/usr/bin/env python3
"""Replace only catastrophic repeated n-gram loops with a fallback ASR output."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def load_predictions(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [row["ID"] for row in rows], {row["ID"]: row["Target"] for row in rows}


def max_ngram_count(text: str, order: int) -> int:
    tokens = [EDGE_PUNCTUATION.sub("", token.lower()) for token in text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < order:
        return 0
    counts = Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))
    return max(counts.values(), default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--ngram-order", type=int, default=4)
    parser.add_argument("--min-count", type=int, default=4)
    args = parser.parse_args()

    if args.ngram_order < 1 or args.min_count < 2:
        parser.error("--ngram-order must be >=1 and --min-count must be >=2")

    ordered_ids, primary = load_predictions(args.primary)
    _, fallback = load_predictions(args.fallback)
    missing = [example_id for example_id in ordered_ids if example_id not in fallback]
    if missing:
        raise ValueError(f"Fallback is missing {len(missing)} IDs; first: {missing[:5]}")

    routed: list[dict[str, str]] = []
    switches = []
    for example_id in ordered_ids:
        primary_count = max_ngram_count(primary[example_id], args.ngram_order)
        fallback_count = max_ngram_count(fallback[example_id], args.ngram_order)
        use_fallback = primary_count >= args.min_count and fallback_count < primary_count
        routed.append(
            {
                "ID": example_id,
                "Target": fallback[example_id] if use_fallback else primary[example_id],
            }
        )
        if use_fallback:
            switches.append(
                {
                    "ID": example_id,
                    "primary_max_ngram_count": primary_count,
                    "fallback_max_ngram_count": fallback_count,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(routed)

    report = {
        "primary": str(args.primary),
        "fallback": str(args.fallback),
        "num_rows": len(routed),
        "ngram_order": args.ngram_order,
        "min_count": args.min_count,
        "num_switches": len(switches),
        "switches": switches,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
