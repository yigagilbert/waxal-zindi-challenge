#!/usr/bin/env python3
"""Validate a Zindi submission file before upload."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import default_raw_dir, load_sample_submission, read_prediction_csv  # noqa: E402


def physical_line_count(path: Path) -> int:
    """Count physical lines in a file."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    args = parser.parse_args()

    rows = read_prediction_csv(args.submission)
    sample_rows = load_sample_submission(args.raw_dir)
    sample_ids = [row["ID"] for row in sample_rows]
    row_ids = [row["ID"] for row in rows]
    row_id_set = set(row_ids)
    sample_id_set = set(sample_ids)

    missing_ids = [example_id for example_id in sample_ids if example_id not in row_id_set]
    extra_ids = sorted(row_id_set - sample_id_set)
    duplicate_ids = sorted({example_id for example_id in row_ids if row_ids.count(example_id) > 1})
    empty_targets = [row["ID"] for row in rows if not row.get("Target", "").strip()]
    newline_targets = [
        row["ID"]
        for row in rows
        if "\n" in row.get("Target", "") or "\r" in row.get("Target", "")
    ]
    replacement_targets = [row["ID"] for row in rows if "\ufffd" in row.get("Target", "")]
    aligned = row_ids == sample_ids
    physical_lines = physical_line_count(args.submission)

    print(f"rows: {len(rows)}")
    print(f"sample rows: {len(sample_rows)}")
    print(f"physical lines: {physical_lines}")
    print(f"expected physical lines if no embedded newlines: {len(rows) + 1}")
    print(f"columns: {list(rows[0].keys()) if rows else []}")
    print(f"aligned to sample: {aligned}")
    print(f"missing ids: {len(missing_ids)} {missing_ids[:20]}")
    print(f"extra ids: {len(extra_ids)} {extra_ids[:20]}")
    print(f"duplicate ids: {len(duplicate_ids)} {duplicate_ids[:20]}")
    print(f"empty targets: {len(empty_targets)} {empty_targets[:20]}")
    print(f"targets with newlines: {len(newline_targets)} {newline_targets[:20]}")
    print(f"targets with replacement char: {len(replacement_targets)} {replacement_targets[:20]}")

    if len(rows) != len(sample_rows) or missing_ids or extra_ids or duplicate_ids or empty_targets:
        raise SystemExit(1)
    if physical_lines != len(rows) + 1:
        raise SystemExit(1)
    if list(rows[0].keys()) != ["ID", "Target"]:
        raise SystemExit(1)
    print("Submission validation passed.")


if __name__ == "__main__":
    main()
