#!/usr/bin/env python3
"""Extract each row's best non-anchor MBR alternative as a prediction CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.anchor.open(encoding="utf-8-sig") as f:
        anchor_rows = list(csv.DictReader(f))
    with args.decisions.open(encoding="utf-8-sig") as f:
        decisions = {row["ID"]: row for row in csv.DictReader(f)}

    output_rows = []
    missing = []
    for row in anchor_rows:
        decision = decisions.get(row["ID"], {})
        alternative = (decision.get("best_alternative_text") or "").strip()
        if not alternative:
            missing.append(row["ID"])
            alternative = row["Target"]
        output_rows.append({"ID": row["ID"], "Target": alternative})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"wrote {len(output_rows)} rows to {args.output}; "
        f"{len(missing)} rows lacked an alternative and retained the anchor"
    )


if __name__ == "__main__":
    main()
