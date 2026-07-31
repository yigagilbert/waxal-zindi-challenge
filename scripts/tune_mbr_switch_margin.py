#!/usr/bin/env python3
"""Nested-CV gate for switching from an anchor to the n-best MBR alternative."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_prediction_csv, references_from_validation_csv  # noqa: E402
from waxal.scoring import edit_distance, score_records  # noqa: E402


def fold_for(example_id: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha1(example_id.encode()).digest()[:4], "big") % folds


def choose_threshold(rows: list[dict], thresholds: list[float]) -> float:
    choices = []
    for threshold in thresholds:
        gain = sum(row["delta"] for row in rows if row["margin"] > threshold)
        switches = sum(row["margin"] > threshold for row in rows)
        choices.append((gain, -switches, threshold))
    return max(choices)[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--per-language", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 999.0],
    )
    args = parser.parse_args()

    anchor = {row["ID"]: row["Target"] for row in read_prediction_csv(args.anchor)}
    references = references_from_validation_csv(args.references)
    ref_by_id = {row["ID"]: row for row in references}
    language_by_id = {
        row["ID"]: (
            row.get("language")
            or row.get("Language")
            or row["ID"].split("_", 1)[0]
        )
        for row in references
    }
    with args.decision_log.open(encoding="utf-8-sig") as f:
        decisions = {row["ID"]: row for row in csv.DictReader(f)}

    total_words = sum(len(row["Target"].split()) for row in references)
    total_chars = sum(len(" ".join(row["Target"].split())) for row in references)
    rows = []
    for example_id, decision in decisions.items():
        if example_id not in ref_by_id:
            continue
        if not decision["anchor_advantage"] or not decision["best_alternative_text"]:
            continue
        ref = " ".join(ref_by_id[example_id]["Target"].split())
        primary = " ".join(anchor[example_id].split())
        alternative = " ".join(decision["best_alternative_text"].split())
        primary_word = edit_distance(ref.split(), primary.split())
        alternative_word = edit_distance(ref.split(), alternative.split())
        primary_char = edit_distance(list(ref), list(primary))
        alternative_char = edit_distance(list(ref), list(alternative))
        rows.append(
            {
                "ID": example_id,
                "language": language_by_id[example_id],
                "margin": float(decision["anchor_advantage"]),
                "alternative": decision["best_alternative_text"],
                "delta": (
                    0.5 * (primary_word - alternative_word) / total_words
                    + 0.5 * (primary_char - alternative_char) / total_chars
                ),
            }
        )

    predictions = {row["ID"]: anchor[row["ID"]] for row in references}
    if args.per_language:
        thresholds_by_fold: list[dict[str, float]] = []
        languages = sorted({row["language"] for row in rows})
        for fold in range(args.folds):
            fold_thresholds = {}
            for language in languages:
                training = [
                    row
                    for row in rows
                    if row["language"] == language and fold_for(row["ID"], args.folds) != fold
                ]
                threshold = choose_threshold(training, args.thresholds)
                fold_thresholds[language] = threshold
                for row in rows:
                    if row["language"] == language and fold_for(row["ID"], args.folds) == fold:
                        predictions[row["ID"]] = (
                            row["alternative"] if row["margin"] > threshold else anchor[row["ID"]]
                        )
            thresholds_by_fold.append(fold_thresholds)
        final_threshold = {
            language: choose_threshold(
                [row for row in rows if row["language"] == language],
                args.thresholds,
            )
            for language in languages
        }
    else:
        thresholds_by_fold = []
        for fold in range(args.folds):
            training = [row for row in rows if fold_for(row["ID"], args.folds) != fold]
            threshold = choose_threshold(training, args.thresholds)
            thresholds_by_fold.append(threshold)
            for row in rows:
                if fold_for(row["ID"], args.folds) == fold:
                    predictions[row["ID"]] = (
                        row["alternative"] if row["margin"] > threshold else anchor[row["ID"]]
                    )
        final_threshold = choose_threshold(rows, args.thresholds)

    baseline_rows = [{"ID": row["ID"], "Target": anchor[row["ID"]]} for row in references]
    oof_rows = [{"ID": row["ID"], "Target": predictions[row["ID"]]} for row in references]
    baseline = score_records(references, baseline_rows, normalization="raw")
    oof = score_records(references, oof_rows, normalization="raw")
    report = {
        "folds": args.folds,
        "per_language": args.per_language,
        "thresholds_by_fold": thresholds_by_fold,
        "final_threshold": final_threshold,
        "oof_num_switches": sum(
            predictions[row["ID"]] != anchor[row["ID"]] for row in references
        ),
        "baseline": baseline,
        "oof": oof,
        "oof_score_gain": (
            baseline["overall_weighted"]["combined"]
            - oof["overall_weighted"]["combined"]
        ),
    }
    print(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(oof_rows)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
