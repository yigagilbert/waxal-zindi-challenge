#!/usr/bin/env python3
"""Merge disjoint per-language prediction CSVs into one full-coverage file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import (  # noqa: E402
    default_raw_dir,
    load_sample_submission,
    read_csv_dicts,
    read_prediction_csv,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        required=True,
        help="Prediction CSV covering a disjoint subset of IDs (e.g. one language). Repeat per file.",
    )
    parser.add_argument("--order", type=Path, default=None, help="CSV with an ID/id column giving the required output row order.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Use SampleSubmission.csv from this raw dir for order/coverage.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_order_ids(order: Path | None, raw_dir: Path | None) -> list[str] | None:
    if order is not None:
        columns, rows, bad = read_csv_dicts(order)
        if bad:
            raise ValueError(f"{order} has rows with extra fields: {bad[:3]}")
        id_column = "ID" if "ID" in columns else "id" if "id" in columns else None
        if id_column is None:
            raise ValueError(f"{order} must contain an ID or id column; got {columns}")
        return [row[id_column] for row in rows]
    base = raw_dir if raw_dir is not None else default_raw_dir()
    sample = base / "SampleSubmission.csv"
    if sample.exists():
        return [row["ID"] for row in load_sample_submission(base)]
    return None


def main() -> None:
    args = parse_args()
    merged: dict[str, str] = {}
    duplicates: list[str] = []
    for path in args.predictions:
        for row in read_prediction_csv(path):
            example_id = row["ID"]
            if example_id in merged:
                duplicates.append(example_id)
                continue
            merged[example_id] = row.get("Target", "")

    if duplicates:
        raise ValueError(
            f"{len(duplicates)} IDs appear in more than one --predictions file "
            f"(inputs are expected to be disjoint, e.g. one file per language). "
            f"First examples: {sorted(set(duplicates))[:10]}"
        )

    order_ids = load_order_ids(args.order, args.raw_dir)
    if order_ids is not None:
        missing = sorted(set(order_ids) - set(merged))
        if missing:
            raise ValueError(
                f"{len(missing)} IDs from the order file are missing from --predictions inputs. "
                f"First examples: {missing[:10]}"
            )
        extra = sorted(set(merged) - set(order_ids))
        if extra:
            print(f"WARNING: {len(extra)} predicted IDs are not present in the order file. First examples: {extra[:10]}")
        ids = order_ids
    else:
        ids = list(merged)

    rows = [{"ID": example_id, "Target": merged[example_id]} for example_id in ids]
    write_csv_rows(args.output, rows, ["ID", "Target"])
    print(f"Wrote {len(rows)} merged predictions to {args.output}")


if __name__ == "__main__":
    main()
