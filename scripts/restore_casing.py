#!/usr/bin/env python3
"""Restore sentence casing on CTC output, which cannot emit uppercase.

The champion's 110-token vocabulary contains punctuation (. , ! ? : ;) but no
uppercase letters, so its transcripts are lowercase while WAXAL references are
cased: Shona references start with a capital 99.4% of the time and Lingala 74.4%.
Every affected first word is a full word error, and Shona references are
multi-sentence, so each sentence start costs another one.

Measured on lin+sna validation references (lowercased to mimic the model, then
re-cased), test-weighted combined error caused purely by casing:

    baseline lowercase        0.0449
    capitalize first word     0.0277
    + capitalize after .!?    0.0079   <- applied here

Forcing a trailing period was tested and rejected (0.0319): the model already
emits sentence-final punctuation where the reference has it, and Lingala
references end with .!? only 66% of the time.

Usage:
  python scripts/restore_casing.py --predictions in.csv --output out.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_AFTER_SENTENCE_END = re.compile(r"([.!?]['\"\)\]]?\s+)([a-z])")


def restore(text: str) -> str:
    if not text:
        return text
    out = text[:1].upper() + text[1:]
    return _AFTER_SENTENCE_END.sub(lambda m: m.group(1) + m.group(2).upper(), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.predictions.open(encoding="utf-8-sig")))
    changed = 0
    caps_added = 0
    for row in rows:
        new = restore(row["Target"])
        if new != row["Target"]:
            changed += 1
            caps_added += sum(1 for a, b in zip(row["Target"], new) if a != b)
        row["Target"] = new

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows([{"ID": r["ID"], "Target": r["Target"]} for r in rows])
    print(f"rows: {len(rows)} | changed: {changed} | letters capitalized: {caps_added}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
