#!/usr/bin/env python3
"""Splice two prediction CSVs by per-clip language routing.

Takes the overlay engine's prediction for clips whose routed language is in
--overlay-languages, and the base engine's prediction for everything else
(including 'unk'). Row order follows the base CSV. Use for per-cluster engine
routing on Phase 2, e.g. SALT-forced output for ach/nyn/xog/myx clips with
af51 output for unclustered clips.

Usage:
  python scripts/splice_predictions.py \
    --base outputs/predictions/phase2_af51_beam5_raw.csv \
    --overlay outputs/predictions/phase2_salt_forced_beam5_raw.csv \
    --routing outputs/analysis/phase2_language_clusters.csv \
    --overlay-languages ach nyn xog myx \
    --output outputs/predictions/phase2_spliced.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_map(path: Path, value_col: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig") as f:
        return {row["ID"]: row[value_col] for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="ID,Target CSV used by default.")
    parser.add_argument("--overlay", type=Path, required=True, help="ID,Target CSV used for routed languages.")
    parser.add_argument("--routing", type=Path, required=True, help="ID,language routing table.")
    parser.add_argument("--overlay-languages", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = read_map(args.base, "Target")
    overlay = read_map(args.overlay, "Target")
    routing = read_map(args.routing, "language")

    if set(base) != set(overlay):
        only_base = sorted(set(base) - set(overlay))[:5]
        only_overlay = sorted(set(overlay) - set(base))[:5]
        raise SystemExit(f"ID mismatch: only-base {only_base} only-overlay {only_overlay}")
    missing_route = sorted(set(base) - set(routing))
    if missing_route:
        print(f"WARNING: {len(missing_route)} IDs missing from routing table -> base engine used")

    overlay_langs = set(args.overlay_languages)
    taken = Counter()
    rows = []
    with args.base.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            example_id = row["ID"]
            lang = routing.get(example_id, "unk")
            if lang in overlay_langs:
                rows.append({"ID": example_id, "Target": overlay[example_id]})
                taken[f"overlay:{lang}"] += 1
            else:
                rows.append({"ID": example_id, "Target": base[example_id]})
                taken[f"base:{lang}"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(rows)

    changed = sum(1 for r in rows if overlay[r["ID"]] != base[r["ID"]] and r["Target"] == overlay[r["ID"]])
    print("routing:", dict(taken.most_common()))
    print(f"rows where overlay differs from base among routed clips: {changed}")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
