#!/usr/bin/env python3
"""Extract the highest-log-probability n-best hypothesis and routing features."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rapidfuzz.distance import Levenshtein as FastLevenshtein
except ImportError:
    FastLevenshtein = None

TOKEN_EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def token_keys(text: str) -> list[str]:
    return [
        value
        for token in text.split()
        if (value := TOKEN_EDGE.sub("", token.lower()))
    ]


def distance(left, right) -> int:
    if FastLevenshtein is not None:
        return FastLevenshtein.distance(left, right)
    # The local fallback is intentionally simple; H100 runs have rapidfuzz.
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, 1):
        current = [i]
        for j, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def pair_cost(left: str, right: str) -> float:
    left_words, right_words = left.split(), right.split()
    return (
        0.5
        * distance(left_words, right_words)
        / max(len(left_words), len(right_words), 1)
        + 0.5 * distance(left, right) / max(len(left), len(right), 1)
    )


def score_value(row: dict[str, str]) -> float:
    try:
        score = float(row.get("sequence_score", ""))
    except (TypeError, ValueError):
        return float("-inf")
    return score if math.isfinite(score) else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-candidate", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.nbest.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[row["ID"]].append(row)
    with args.anchor.open(encoding="utf-8-sig") as f:
        anchor_rows = list(csv.DictReader(f))

    candidate_rows = []
    feature_rows = []
    for anchor_row in anchor_rows:
        row_id = anchor_row["ID"]
        rows = groups.get(row_id, [])
        if not rows:
            candidate_rows.append(anchor_row)
            feature_rows.append(
                {
                    "ID": row_id,
                    "score_best": 0.0,
                    "score_second": 0.0,
                    "score_gap": 0.0,
                    "score_mean": 0.0,
                    "score_std": 0.0,
                    "score_range": 0.0,
                    "anchor_seen": 0.0,
                    "anchor_score_gap": 0.0,
                    "best_count": 0.0,
                    "anchor_count": 0.0,
                    "num_unique": 0.0,
                    "best_risk": 0.0,
                    "anchor_risk": 0.0,
                    "risk_advantage": 0.0,
                    "best_rank": 0.0,
                    "best_equals_anchor": 1.0,
                }
            )
            continue

        ranked = sorted(rows, key=score_value, reverse=True)
        best = ranked[0]
        finite_scores = [score_value(row) for row in rows if math.isfinite(score_value(row))]
        if not finite_scores:
            finite_scores = [0.0]
        sorted_scores = sorted(finite_scores, reverse=True)
        score_best = sorted_scores[0]
        score_second = sorted_scores[1] if len(sorted_scores) > 1 else score_best
        score_mean = sum(finite_scores) / len(finite_scores)
        score_std = (
            sum((score - score_mean) ** 2 for score in finite_scores) / len(finite_scores)
        ) ** 0.5

        texts = [normalized(row["Target"]) for row in rows]
        counts = Counter(texts)
        anchor_text = normalized(anchor_row["Target"])
        best_text = normalized(best["Target"])
        anchor_scores = [
            score_value(row)
            for row in rows
            if normalized(row["Target"]) == anchor_text and math.isfinite(score_value(row))
        ]
        unique_texts = list(dict.fromkeys([anchor_text, *texts]))
        risks = {
            text: sum(pair_cost(text, other) for other in texts) / max(len(texts), 1)
            for text in unique_texts
        }

        candidate_rows.append({"ID": row_id, "Target": best["Target"]})
        feature_rows.append(
            {
                "ID": row_id,
                "score_best": score_best,
                "score_second": score_second,
                "score_gap": score_best - score_second,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_range": max(finite_scores) - min(finite_scores),
                "anchor_seen": float(bool(anchor_scores)),
                "anchor_score_gap": (
                    score_best - max(anchor_scores) if anchor_scores else 0.0
                ),
                "best_count": float(counts[best_text]),
                "anchor_count": float(counts[anchor_text]),
                "num_unique": float(len(set(texts))),
                "best_risk": risks[best_text],
                "anchor_risk": risks[anchor_text],
                "risk_advantage": risks[anchor_text] - risks[best_text],
                "best_rank": float(best.get("rank", 0) or 0),
                "best_equals_anchor": float(best_text == anchor_text),
            }
        )

    args.output_candidate.parent.mkdir(parents=True, exist_ok=True)
    with args.output_candidate.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(candidate_rows)
    feature_fields = list(feature_rows[0])
    with args.output_features.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fields)
        writer.writeheader()
        writer.writerows(feature_rows)
    print(
        f"wrote {len(candidate_rows)} candidates and {len(feature_rows)} feature rows; "
        f"{sum(row['Target'] != anchor['Target'] for row, anchor in zip(candidate_rows, anchor_rows, strict=True))} "
        "surface changes"
    )


if __name__ == "__main__":
    main()
