#!/usr/bin/env python3
"""Tune and apply log-linear fusion of text and acoustic LID probabilities."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

LANGUAGES = ("ach", "myx", "nyn", "xog")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-validation", type=Path, required=True)
    parser.add_argument("--audio-validation", type=Path, required=True)
    parser.add_argument("--text-test", type=Path, required=True)
    parser.add_argument("--audio-test", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--weight-steps", type=int, default=101)
    parser.add_argument(
        "--objective",
        choices=["accuracy", "macro_accuracy"],
        default="macro_accuracy",
    )
    parser.add_argument("--unknown-threshold", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/phase2_lid_fused.csv"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("outputs/analysis/validation_lid_fused.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/analysis/phase2_lid_fusion_report.json"),
    )
    return parser.parse_args()


def read_probabilities(
    path: Path,
    languages: list[str],
) -> tuple[list[str], np.ndarray, list[str] | None]:
    ids: list[str] = []
    probabilities: list[list[float]] = []
    truths: list[str] = []
    saw_truth = False
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ids.append(str(row["ID"]))
            probabilities.append([float(row[f"p_{language}"]) for language in languages])
            truth = str(row.get("true_language") or "")
            truths.append(truth)
            saw_truth = saw_truth or bool(truth)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate IDs")
    return ids, np.asarray(probabilities, dtype=np.float64), truths if saw_truth else None


def align_second_view(
    primary_ids: list[str],
    secondary_ids: list[str],
    secondary_probabilities: np.ndarray,
    path: Path,
) -> np.ndarray:
    secondary_index = {example_id: index for index, example_id in enumerate(secondary_ids)}
    primary_set = set(primary_ids)
    missing = [example_id for example_id in primary_ids if example_id not in secondary_index]
    extra = [example_id for example_id in secondary_ids if example_id not in primary_set]
    if missing or extra:
        raise ValueError(f"{path}: {len(missing)} missing and {len(extra)} extra IDs")
    return np.stack(
        [secondary_probabilities[secondary_index[example_id]] for example_id in primary_ids]
    )


def fuse(text_probabilities: np.ndarray, audio_probabilities: np.ndarray, weight: float):
    epsilon = 1e-9
    log_scores = (
        weight * np.log(np.clip(text_probabilities, epsilon, 1.0))
        + (1.0 - weight) * np.log(np.clip(audio_probabilities, epsilon, 1.0))
    )
    log_scores -= log_scores.max(axis=1, keepdims=True)
    probabilities = np.exp(log_scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def metrics(
    probabilities: np.ndarray,
    truths: list[str],
    languages: list[str],
) -> dict[str, Any]:
    language_array = np.asarray(languages)
    truth_array = np.asarray(truths)
    predicted = language_array[np.argmax(probabilities, axis=1)]
    correct = predicted == truth_array
    per_language = {}
    recalls = []
    for language in languages:
        mask = truth_array == language
        accuracy = float(correct[mask].mean()) if mask.any() else None
        if accuracy is not None:
            recalls.append(accuracy)
        per_language[language] = {
            "examples": int(mask.sum()),
            "correct": int(correct[mask].sum()),
            "accuracy": accuracy,
        }
    return {
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(np.mean(recalls)),
        "correct": int(correct.sum()),
        "examples": len(truths),
        "per_language": per_language,
    }


def write_probabilities(
    *,
    path: Path,
    ids: list[str],
    probabilities: np.ndarray,
    languages: list[str],
    unknown_threshold: float,
    truths: list[str] | None = None,
) -> dict[str, Any]:
    language_array = np.asarray(languages)
    order = np.argsort(probabilities, axis=1)
    top_index = order[:, -1]
    second_index = order[:, -2]
    predicted = language_array[top_index]
    confidence = probabilities[np.arange(len(ids)), top_index]
    margin = confidence - probabilities[np.arange(len(ids)), second_index]
    routed = np.where(confidence >= unknown_threshold, predicted, "unk")

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ID",
        "language",
        "predicted_language",
        "confidence",
        "margin",
        *[f"p_{language}" for language in languages],
    ]
    if truths is not None:
        fields.insert(1, "true_language")
        fields.append("correct")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, example_id in enumerate(ids):
            row: dict[str, Any] = {
                "ID": example_id,
                "language": routed[row_index],
                "predicted_language": predicted[row_index],
                "confidence": f"{confidence[row_index]:.8f}",
                "margin": f"{margin[row_index]:.8f}",
            }
            for class_index, language in enumerate(languages):
                row[f"p_{language}"] = f"{probabilities[row_index, class_index]:.8f}"
            if truths is not None:
                row["true_language"] = truths[row_index]
                row["correct"] = int(predicted[row_index] == truths[row_index])
            writer.writerow(row)

    return {
        "examples": len(ids),
        "unknown_threshold": unknown_threshold,
        "histogram": dict(Counter(routed.tolist())),
        "top_prediction_histogram": dict(Counter(predicted.tolist())),
        "mean_confidence": float(confidence.mean()),
        "mean_margin": float(margin.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.weight_steps < 2:
        raise SystemExit("--weight-steps must be at least 2")
    if not 0.0 <= args.unknown_threshold <= 1.0:
        raise SystemExit("--unknown-threshold must be between 0 and 1")
    languages = sorted(set(args.languages))

    val_ids, text_val, text_truths = read_probabilities(
        args.text_validation,
        languages,
    )
    audio_val_ids, audio_val_unaligned, audio_truths = read_probabilities(
        args.audio_validation,
        languages,
    )
    if text_truths is None or audio_truths is None:
        raise ValueError("Both validation files need true_language columns")
    audio_val = align_second_view(
        val_ids,
        audio_val_ids,
        audio_val_unaligned,
        args.audio_validation,
    )
    audio_truth_by_id = dict(zip(audio_val_ids, audio_truths, strict=True))
    aligned_audio_truths = [audio_truth_by_id[example_id] for example_id in val_ids]
    if text_truths != aligned_audio_truths:
        raise ValueError("Text and audio validation truth labels disagree")

    trials = []
    best_weight = None
    best_metrics = None
    for weight in np.linspace(0.0, 1.0, args.weight_steps):
        trial_metrics = metrics(fuse(text_val, audio_val, float(weight)), text_truths, languages)
        trials.append(
            {
                "text_weight": float(weight),
                "accuracy": trial_metrics["accuracy"],
                "macro_accuracy": trial_metrics["macro_accuracy"],
            }
        )
        if (
            best_metrics is None
            or trial_metrics[args.objective] > best_metrics[args.objective]
            or (
                trial_metrics[args.objective] == best_metrics[args.objective]
                and abs(float(weight) - 0.5) < abs(float(best_weight) - 0.5)
            )
        ):
            best_weight = float(weight)
            best_metrics = trial_metrics

    if best_weight is None or best_metrics is None:
        raise RuntimeError("No fusion weights evaluated")
    fused_validation = fuse(text_val, audio_val, best_weight)
    validation_summary = write_probabilities(
        path=args.validation_output,
        ids=val_ids,
        probabilities=fused_validation,
        languages=languages,
        unknown_threshold=args.unknown_threshold,
        truths=text_truths,
    )

    test_ids, text_test, _ = read_probabilities(args.text_test, languages)
    audio_test_ids, audio_test_unaligned, _ = read_probabilities(args.audio_test, languages)
    audio_test = align_second_view(
        test_ids,
        audio_test_ids,
        audio_test_unaligned,
        args.audio_test,
    )
    fused_test = fuse(text_test, audio_test, best_weight)
    test_summary = write_probabilities(
        path=args.output,
        ids=test_ids,
        probabilities=fused_test,
        languages=languages,
        unknown_threshold=args.unknown_threshold,
    )

    report = {
        "data_policy": "fusion of WAXAL-trained text and acoustic LID only",
        "objective": args.objective,
        "text_validation": str(args.text_validation),
        "audio_validation": str(args.audio_validation),
        "text_test": str(args.text_test),
        "audio_test": str(args.audio_test),
        "best_text_weight": best_weight,
        "validation": best_metrics,
        "validation_output": str(args.validation_output),
        "validation_routing": validation_summary,
        "trials": trials,
        "test": test_summary,
        "output": str(args.output),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
