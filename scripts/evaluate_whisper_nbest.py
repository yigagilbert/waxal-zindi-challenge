#!/usr/bin/env python3
"""Measure top-1 and exact per-utterance oracle headroom in a Whisper n-best CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import references_from_validation_csv  # noqa: E402
from waxal.scoring import edit_distance, score_records  # noqa: E402
from waxal.text_normalization import POLICIES, normalize_text  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein as FastLevenshtein
except ImportError:
    FastLevenshtein = None


def distance(left, right) -> int:
    if FastLevenshtein is not None:
        return FastLevenshtein.distance(left, right)
    return edit_distance(left, right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--normalization", choices=POLICIES, default="raw")
    parser.add_argument("--output-oracle", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    args = parser.parse_args()

    references = references_from_validation_csv(args.references)
    ref_by_id = {row["ID"]: row for row in references}
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.nbest.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["ID"] in ref_by_id:
                candidates[row["ID"]].append(row)

    missing = [row["ID"] for row in references if row["ID"] not in candidates]
    if missing:
        raise ValueError(f"n-best file is missing {len(missing)} reference IDs; first: {missing[:5]}")

    normalized_refs = {
        row["ID"]: normalize_text(row["Target"], args.normalization)
        for row in references
    }
    total_words = sum(len(text.split()) for text in normalized_refs.values())
    total_chars = sum(len(text) for text in normalized_refs.values())
    if total_words == 0 or total_chars == 0:
        raise ValueError("Reference corpus is empty after normalization")

    top1_rows: list[dict[str, str]] = []
    oracle_rows: list[dict[str, str]] = []
    oracle_ranks: Counter[int] = Counter()
    unique_counts: list[int] = []
    for ref_row in references:
        example_id = ref_row["ID"]
        ordered = sorted(candidates[example_id], key=lambda row: int(row["rank"]))
        top1_rows.append({"ID": example_id, "Target": ordered[0]["Target"]})
        unique_counts.append(len({normalize_text(row["Target"], args.normalization) for row in ordered}))

        ref = normalized_refs[example_id]
        ref_words = ref.split()
        scored = []
        for row in ordered:
            hyp = normalize_text(row["Target"], args.normalization)
            word_errors = distance(ref_words, hyp.split())
            char_errors = distance(ref, hyp)
            # These are the exact additive contributions to the corpus-level
            # 0.5*WER + 0.5*CER objective; both denominators are global constants.
            cost = 0.5 * word_errors / total_words + 0.5 * char_errors / total_chars
            scored.append((cost, int(row["rank"]), row))
        _, rank, best = min(scored, key=lambda item: (item[0], item[1]))
        oracle_ranks[rank] += 1
        oracle_rows.append({"ID": example_id, "Target": best["Target"]})

    top1 = score_records(references, top1_rows, normalization=args.normalization)
    oracle = score_records(references, oracle_rows, normalization=args.normalization)
    top1_error = top1["overall_weighted"]["combined"]
    oracle_error = oracle["overall_weighted"]["combined"]
    report = {
        "normalization": args.normalization,
        "num_examples": len(references),
        "mean_unique_hypotheses": sum(unique_counts) / len(unique_counts),
        "top1": top1,
        "oracle": oracle,
        "oracle_absolute_error_reduction": top1_error - oracle_error,
        "oracle_score_gain": top1_error - oracle_error,
        "oracle_rank_counts": dict(sorted(oracle_ranks.items())),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output_oracle:
        args.output_oracle.parent.mkdir(parents=True, exist_ok=True)
        with args.output_oracle.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
            writer.writeheader()
            writer.writerows(oracle_rows)
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
