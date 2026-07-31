#!/usr/bin/env python3
"""Train and evaluate a WAXAL-only character n-gram language router.

The classifier is trained only on labeled transcripts from a prepared
``google/WaxalNLP`` DatasetDict. Its honest gate is accuracy on ASR-generated
validation transcripts, not accuracy on clean validation references.

Example:

  python scripts/train_text_lid.py \
    --dataset-dir data/phase2_train \
    --eval-predictions outputs/predictions/salt_val_auto_full_lp08.csv \
    --predict-csv outputs/predictions/phase2_af51_beam5_raw.csv \
    --fit-validation-for-predict \
    --unknown-threshold 0.80 \
    --routing-output outputs/analysis/phase2_text_lid.csv \
    --eval-details-output outputs/analysis/text_lid_validation.csv \
    --report-output outputs/analysis/text_lid_report.json \
    --model-output outputs/models/text_lid.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

LANGUAGES = ("ach", "myx", "nyn", "xog")
CONFIDENCE_THRESHOLDS = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument(
        "--eval-predictions",
        type=Path,
        default=None,
        help="ID,Target ASR predictions for the labeled eval split.",
    )
    parser.add_argument(
        "--predict-csv",
        type=Path,
        default=None,
        help="ID,Target ASR predictions to route after evaluation (normally Phase-2 test).",
    )
    parser.add_argument(
        "--fit-validation-for-predict",
        action="store_true",
        help="After evaluation, refit on train+validation clean references before routing predict-csv.",
    )
    parser.add_argument("--unknown-threshold", type=float, default=0.0)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=250_000)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--routing-output", type=Path, default=None)
    parser.add_argument("--eval-details-output", type=Path, default=None)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/analysis/text_lid_report.json"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("outputs/models/text_lid.joblib"),
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace without destroying orthographic cues."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_dataset_dict(dataset_dir: Path):
    from datasets import load_from_disk

    dataset_path = dataset_dir / "hf_dataset"
    dataset_dict = load_from_disk(dataset_path)
    return dataset_dict


def select_labeled_rows(dataset, languages: set[str]) -> tuple[list[str], list[str], list[str]]:
    required = {"ID", "language", "transcription"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns {sorted(missing)}; has {dataset.column_names}")

    ids: list[str] = []
    texts: list[str] = []
    labels: list[str] = []
    for example_id, language, transcript in zip(
        dataset["ID"],
        dataset["language"],
        dataset["transcription"],
        strict=True,
    ):
        if language not in languages:
            continue
        ids.append(str(example_id))
        texts.append(normalize_text(str(transcript or "")))
        labels.append(str(language))
    return ids, texts, labels


def read_prediction_csv(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            example_id = str(row["ID"])
            if example_id in seen:
                raise ValueError(f"{path}: duplicate ID {example_id!r}")
            seen.add(example_id)
            ids.append(example_id)
            texts.append(normalize_text(str(row.get("Target") or "")))
    return ids, texts


def build_model(args: argparse.Namespace):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    lowercase=False,
                    min_df=args.min_df,
                    max_features=args.max_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=args.seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def predict_with_confidence(
    model,
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(texts)
    order = np.argsort(probabilities, axis=1)
    top_index = order[:, -1]
    second_index = order[:, -2]
    classes = np.asarray(model.classes_)
    labels = classes[top_index]
    confidence = probabilities[np.arange(len(texts)), top_index]
    margin = confidence - probabilities[np.arange(len(texts)), second_index]
    return labels, confidence, margin, probabilities


def classification_report(
    true_labels: list[str],
    predicted_labels: np.ndarray,
    confidence: np.ndarray,
    languages: list[str],
) -> dict[str, Any]:
    true_array = np.asarray(true_labels)
    correct = predicted_labels == true_array
    confusion = {
        true: {
            predicted: int(np.sum((true_array == true) & (predicted_labels == predicted)))
            for predicted in languages
        }
        for true in languages
    }
    per_language = {}
    for language in languages:
        mask = true_array == language
        count = int(mask.sum())
        per_language[language] = {
            "examples": count,
            "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()) if count else None,
            "mean_confidence": float(confidence[mask].mean()) if count else None,
        }

    confidence_gates = {}
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        confidence_gates[f"{threshold:.2f}"] = {
            "accepted": accepted_count,
            "coverage": float(accepted.mean()),
            "accuracy": float(correct[accepted].mean()) if accepted_count else None,
            "errors": int((~correct[accepted]).sum()) if accepted_count else 0,
        }

    return {
        "examples": len(true_labels),
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "per_language": per_language,
        "confusion": confusion,
        "confidence_gates": confidence_gates,
    }


def evaluate_predictions(
    *,
    model,
    eval_ids: list[str],
    eval_labels: list[str],
    prediction_path: Path,
    languages: list[str],
    details_output: Path | None,
) -> dict[str, Any]:
    pred_ids, pred_texts = read_prediction_csv(prediction_path)
    text_by_id = dict(zip(pred_ids, pred_texts, strict=True))
    label_by_id = dict(zip(eval_ids, eval_labels, strict=True))

    missing = [example_id for example_id in eval_ids if example_id not in text_by_id]
    extra = [example_id for example_id in pred_ids if example_id not in label_by_id]
    if missing or extra:
        raise ValueError(
            f"{prediction_path}: alignment failure: {len(missing)} missing and {len(extra)} extra IDs"
        )

    ordered_texts = [text_by_id[example_id] for example_id in eval_ids]
    predicted, confidence, margin, probabilities = predict_with_confidence(
        model,
        ordered_texts,
    )
    classes = [str(value) for value in model.classes_]
    report = classification_report(eval_labels, predicted, confidence, languages)
    report["source"] = str(prediction_path)
    report["empty_transcripts"] = int(sum(not text for text in ordered_texts))

    if details_output is not None:
        details_output.parent.mkdir(parents=True, exist_ok=True)
        with details_output.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "ID",
                "true_language",
                "predicted_language",
                "confidence",
                "margin",
                *[f"p_{language}" for language in classes],
                "correct",
                "Target",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row_index, (
                example_id,
                truth,
                prediction,
                conf,
                gap,
                text,
            ) in enumerate(
                zip(
                    eval_ids,
                    eval_labels,
                    predicted,
                    confidence,
                    margin,
                    ordered_texts,
                    strict=True,
                )
            ):
                row = {
                    "ID": example_id,
                    "true_language": truth,
                    "predicted_language": prediction,
                    "confidence": f"{conf:.8f}",
                    "margin": f"{gap:.8f}",
                    "correct": int(prediction == truth),
                    "Target": text,
                }
                for class_index, language in enumerate(classes):
                    row[f"p_{language}"] = (
                        f"{probabilities[row_index, class_index]:.8f}"
                    )
                writer.writerow(row)
    return report


def write_routing(
    *,
    model,
    prediction_path: Path,
    output_path: Path,
    unknown_threshold: float,
) -> dict[str, Any]:
    ids, texts = read_prediction_csv(prediction_path)
    predicted, confidence, margin, probabilities = predict_with_confidence(
        model,
        texts,
    )
    classes = [str(value) for value in model.classes_]
    routed = np.where(confidence >= unknown_threshold, predicted, "unk")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "ID",
            "language",
            "predicted_language",
            "confidence",
            "margin",
            *[f"p_{language}" for language in classes],
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        for row_index, (example_id, route, prediction, conf, gap) in enumerate(
            zip(
                ids,
                routed,
                predicted,
                confidence,
                margin,
                strict=True,
            )
        ):
            row = {
                "ID": example_id,
                "language": route,
                "predicted_language": prediction,
                "confidence": f"{conf:.8f}",
                "margin": f"{gap:.8f}",
            }
            for class_index, language in enumerate(classes):
                row[f"p_{language}"] = (
                    f"{probabilities[row_index, class_index]:.8f}"
                )
            writer.writerow(row)

    return {
        "source": str(prediction_path),
        "output": str(output_path),
        "examples": len(ids),
        "empty_transcripts": int(sum(not text for text in texts)),
        "unknown_threshold": unknown_threshold,
        "histogram": dict(Counter(routed.tolist())),
        "top_prediction_histogram": dict(Counter(predicted.tolist())),
        "mean_confidence": float(confidence.mean()),
        "mean_margin": float(margin.mean()),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.unknown_threshold <= 1.0:
        raise SystemExit("--unknown-threshold must be between 0 and 1")
    languages = sorted(set(args.languages))
    language_set = set(languages)

    dataset_dict = load_dataset_dict(args.dataset_dir)
    if args.train_split not in dataset_dict or args.eval_split not in dataset_dict:
        raise SystemExit(
            f"Need splits {args.train_split!r}/{args.eval_split!r}; "
            f"available: {list(dataset_dict)}"
        )

    train_ids, train_texts, train_labels = select_labeled_rows(
        dataset_dict[args.train_split],
        language_set,
    )
    eval_ids, eval_texts, eval_labels = select_labeled_rows(
        dataset_dict[args.eval_split],
        language_set,
    )
    if not train_texts or not eval_texts:
        raise SystemExit("No labeled train/eval examples after language filtering")

    model = build_model(args)
    print(f"Fitting text LID on {len(train_texts)} WAXAL train transcripts")
    print(f"train languages: {dict(Counter(train_labels))}")
    model.fit(train_texts, train_labels)

    clean_predicted, clean_confidence, _, _ = predict_with_confidence(
        model,
        eval_texts,
    )
    report: dict[str, Any] = {
        "data_policy": "google/WaxalNLP train/validation only",
        "dataset_dir": str(args.dataset_dir),
        "languages": languages,
        "train_examples": len(train_texts),
        "train_histogram": dict(Counter(train_labels)),
        "eval_examples": len(eval_texts),
        "eval_histogram": dict(Counter(eval_labels)),
        "model": {
            "features": "TF-IDF char_wb 2-6 grams",
            "min_df": args.min_df,
            "max_features": args.max_features,
            "classifier": "balanced logistic regression",
            "c": args.c,
            "seed": args.seed,
        },
        "clean_reference_eval": classification_report(
            eval_labels,
            clean_predicted,
            clean_confidence,
            languages,
        ),
    }

    if args.eval_predictions is not None:
        report["noisy_asr_eval"] = evaluate_predictions(
            model=model,
            eval_ids=eval_ids,
            eval_labels=eval_labels,
            prediction_path=args.eval_predictions,
            languages=languages,
            details_output=args.eval_details_output,
        )

    if args.fit_validation_for_predict:
        print(f"Refitting final router on {len(train_texts) + len(eval_texts)} train+validation references")
        model = build_model(args)
        model.fit(train_texts + eval_texts, train_labels + eval_labels)
        report["final_fit_examples"] = len(train_texts) + len(eval_texts)
        report["final_fit_includes_validation"] = True
    else:
        report["final_fit_examples"] = len(train_texts)
        report["final_fit_includes_validation"] = False

    if args.predict_csv is not None:
        if args.routing_output is None:
            raise SystemExit("--routing-output is required with --predict-csv")
        report["routing"] = write_routing(
            model=model,
            prediction_path=args.predict_csv,
            output_path=args.routing_output,
            unknown_threshold=args.unknown_threshold,
        )

    import joblib

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    report["model_output"] = str(args.model_output)

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {args.report_output}")
    print(f"Wrote {args.model_output}")


if __name__ == "__main__":
    main()
