#!/usr/bin/env python3
"""Select an n-best hypothesis by edit-distance minimum Bayes risk (medoid)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_prediction_csv, references_from_validation_csv  # noqa: E402
from waxal.scoring import edit_distance, score_records  # noqa: E402
from waxal.text_normalization import POLICIES, normalize_text  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein as FastLevenshtein
except ImportError:
    FastLevenshtein = None


def pair_cost(left: str, right: str) -> float:
    left_words, right_words = left.split(), right.split()
    word_denom = max(len(left_words), len(right_words), 1)
    char_denom = max(len(left), len(right), 1)
    if FastLevenshtein is not None:
        word_distance = FastLevenshtein.distance(left_words, right_words)
        char_distance = FastLevenshtein.distance(left, right)
    else:
        word_distance = edit_distance(left_words, right_words)
        char_distance = edit_distance(list(left), list(right))
    return (
        0.5 * word_distance / word_denom
        + 0.5 * char_distance / char_denom
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, default=None, help="Optional deterministic candidate to include in every set.")
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--normalization", choices=POLICIES, default="raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--decision-log", type=Path, default=None)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.nbest.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[row["ID"]].append(row)
    anchor = {}
    if args.anchor:
        anchor = {row["ID"]: row["Target"] for row in read_prediction_csv(args.anchor)}

    selected = []
    decisions = []
    source_counts: Counter[str] = Counter()
    unique_counts = []
    for example_id, rows in groups.items():
        candidates: list[tuple[str, str]] = []
        if example_id in anchor:
            candidates.append(("anchor", anchor[example_id]))
        candidates.extend((f"sample_{row['rank']}", row["Target"]) for row in rows)

        # Collapse exact normalized duplicates while retaining anchor priority.
        unique: list[tuple[str, str, str]] = []
        seen = set()
        for source, text in candidates:
            normalized = normalize_text(text, args.normalization)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append((source, text, normalized))
        unique_counts.append(len(unique))

        risks = []
        for index, (source, text, normalized) in enumerate(unique):
            risk = sum(
                pair_cost(normalized, other_normalized)
                for _, _, other_normalized in unique
            ) / max(len(unique), 1)
            risks.append((risk, index, source, text))
        _, _, source, text = min(risks, key=lambda item: (item[0], item[1]))
        anchor_risk = next((risk for risk, _, candidate_source, _ in risks if candidate_source == "anchor"), None)
        alternative_risks = [item for item in risks if item[2] != "anchor"]
        best_alternative = min(alternative_risks, key=lambda item: (item[0], item[1])) if alternative_risks else None
        decisions.append(
            {
                "ID": example_id,
                "selected_source": source,
                "anchor_risk": "" if anchor_risk is None else anchor_risk,
                "best_alternative_source": "" if best_alternative is None else best_alternative[2],
                "best_alternative_risk": "" if best_alternative is None else best_alternative[0],
                "anchor_advantage": (
                    ""
                    if anchor_risk is None or best_alternative is None
                    else anchor_risk - best_alternative[0]
                ),
                "best_alternative_text": "" if best_alternative is None else best_alternative[3],
            }
        )
        source_counts[source] += 1
        selected.append({"ID": example_id, "Target": text})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(selected)

    report = {
        "num_examples": len(selected),
        "mean_unique_candidates": sum(unique_counts) / max(len(unique_counts), 1),
        "source_counts": dict(source_counts),
    }
    if args.references:
        references = references_from_validation_csv(args.references)
        wanted = {row["ID"] for row in selected}
        references = [row for row in references if row["ID"] in wanted]
        report["mbr"] = score_records(references, selected, normalization=args.normalization)
        if anchor:
            anchor_rows = [{"ID": row["ID"], "Target": anchor[row["ID"]]} for row in references]
            report["anchor"] = score_records(references, anchor_rows, normalization=args.normalization)
            report["score_gain"] = (
                report["anchor"]["overall_weighted"]["combined"]
                - report["mbr"]["overall_weighted"]["combined"]
            )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.decision_log:
        args.decision_log.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "ID",
            "selected_source",
            "anchor_risk",
            "best_alternative_source",
            "best_alternative_risk",
            "anchor_advantage",
            "best_alternative_text",
        ]
        with args.decision_log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(decisions)


if __name__ == "__main__":
    main()
