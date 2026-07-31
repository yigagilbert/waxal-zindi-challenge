#!/usr/bin/env python3
"""Apply a validation-frozen MBR confidence margin without undoing trusted rows."""

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


def max_ngram_count(text: str, order: int = 4) -> int:
    tokens = [EDGE_PUNCTUATION.sub("", token.lower()) for token in text.split()]
    tokens = [token for token in tokens if token]
    grams = [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]
    return max(Counter(grams).values(), default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True, help="Trusted current candidate.")
    parser.add_argument(
        "--selection-anchor",
        type=Path,
        required=True,
        help="Deterministic anchor used when calculating the decision margins.",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    order, primary = load_predictions(args.primary)
    _, selection_anchor = load_predictions(args.selection_anchor)
    with args.decisions.open(encoding="utf-8-sig") as f:
        decisions = {row["ID"]: row for row in csv.DictReader(f)}

    output_rows = []
    switches = []
    blocked_trusted = []
    blocked_loop = []
    for example_id in order:
        current = primary[example_id]
        decision = decisions.get(example_id, {})
        margin_text = decision.get("anchor_advantage", "")
        alternative = decision.get("best_alternative_text", "")
        qualifies = bool(margin_text and alternative and float(margin_text) > args.threshold)
        trusted_differs = current != selection_anchor.get(example_id, current)
        alternative_loops = max_ngram_count(alternative) >= 4

        use_alternative = qualifies and not trusted_differs and not alternative_loops
        output_rows.append(
            {
                "ID": example_id,
                "Target": alternative if use_alternative else current,
            }
        )
        if use_alternative:
            switches.append(
                {
                    "ID": example_id,
                    "margin": float(margin_text),
                    "source": decision.get("best_alternative_source", ""),
                    "primary": current,
                    "alternative": alternative,
                }
            )
        elif qualifies and trusted_differs:
            blocked_trusted.append(example_id)
        elif qualifies and alternative_loops:
            blocked_loop.append(example_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "threshold": args.threshold,
        "num_rows": len(output_rows),
        "num_switches": len(switches),
        "switches": switches,
        "num_blocked_trusted": len(blocked_trusted),
        "blocked_trusted": blocked_trusted,
        "num_blocked_loop": len(blocked_loop),
        "blocked_loop": blocked_loop,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
