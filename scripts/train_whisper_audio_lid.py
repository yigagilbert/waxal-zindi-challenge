#!/usr/bin/env python3
"""Train a four-language acoustic router on frozen SALT encoder features.

Only labeled WAXAL train/validation audio is used to fit or evaluate the
classifier. Phase-2 audio is passed through the frozen encoder for prediction;
it is never assigned pseudo-labels for ASR training.

Example:

  python scripts/train_whisper_audio_lid.py \
    --dataset-dir data/phase2_train \
    --predict-dataset-dir data/processed_phase2 \
    --predict-split test \
    --max-train-per-language 1200 \
    --batch-size 16 \
    --output-dir outputs/audio_lid_phase2 \
    --fit-validation-for-predict
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MODEL = "Sunbird/asr-whisper-large-v3-salt"
DEFAULT_REVISION = "7448016c50bdec469b8454c9631c76fc1d1dd40e"
LANGUAGES = ("ach", "myx", "nyn", "xog")
CONFIDENCE_THRESHOLDS = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--predict-dataset-dir", type=Path, default=None)
    parser.add_argument("--predict-split", default="test")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--max-train-per-language", type=int, default=1200)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-predict-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--unknown-threshold", type=float, default=0.0)
    parser.add_argument(
        "--fit-validation-for-predict",
        action="store_true",
        help="After honest validation, refit the linear router on train+validation features.",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse matching .npz feature files in output-dir instead of extracting again.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audio_lid_phase2"),
    )
    return parser.parse_args()


def load_split(dataset_dir: Path, split: str):
    from datasets import load_from_disk

    dataset_dict = load_from_disk(dataset_dir / "hf_dataset")
    if split not in dataset_dict:
        raise ValueError(
            f"{split!r} not found in {dataset_dir / 'hf_dataset'}; "
            f"available: {list(dataset_dict)}"
        )
    return dataset_dict[split]


def balanced_train_subset(dataset, languages: list[str], limit: int, seed: int):
    if "language" not in dataset.column_names:
        raise ValueError("Training split needs a language column")
    by_language: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(dataset["language"]):
        if language in languages:
            by_language[str(language)].append(index)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for language in languages:
        indices = np.asarray(by_language[language], dtype=np.int64)
        rng.shuffle(indices)
        selected.extend(indices[:limit].tolist())
    selected.sort()
    return dataset.select(selected)


def labeled_subset(dataset, languages: set[str]):
    if "language" not in dataset.column_names:
        raise ValueError("Evaluation split needs a language column")
    indices = [
        index
        for index, language in enumerate(dataset["language"])
        if language in languages
    ]
    return dataset.select(indices)


def feature_cache_matches(
    path: Path,
    *,
    model_name: str,
    model_revision: str,
    ids: list[str],
) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as cached:
        return (
            str(cached["model_name"].item()) == model_name
            and str(cached["model_revision"].item()) == model_revision
            and cached["ids"].astype(str).tolist() == ids
        )


def extract_features(
    *,
    model,
    processor,
    dataset,
    batch_size: int,
    device: str,
    model_name: str,
    model_revision: str,
    cache_path: Path,
    reuse_features: bool,
    label: str,
) -> tuple[np.ndarray, list[str], list[str] | None]:
    import torch

    ids = [str(value) for value in dataset["ID"]]
    labels = (
        [str(value) for value in dataset["language"]]
        if "language" in dataset.column_names
        else None
    )
    if reuse_features and feature_cache_matches(
        cache_path,
        model_name=model_name,
        model_revision=model_revision,
        ids=ids,
    ):
        print(f"Reusing {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_labels = cached["labels"].astype(str).tolist()
            return (
                cached["features"].astype(np.float32),
                ids,
                cached_labels if cached_labels else None,
            )

    encoder = model.get_encoder()
    model_dtype = next(model.parameters()).dtype
    feature_batches: list[np.ndarray] = []
    completed = 0

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        audio_arrays = [audio["array"] for audio in batch["audio"]]
        inputs = processor(
            audio_arrays,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        input_features = inputs["input_features"].to(device=device, dtype=model_dtype)
        with torch.inference_mode():
            hidden = encoder(input_features=input_features).last_hidden_state

        pooled: list[torch.Tensor] = []
        for row_index, audio in enumerate(audio_arrays):
            # Whisper uses a 160-sample feature hop and the encoder downsamples
            # the 3000 log-Mel frames by two.
            mel_frames = min(3000, max(1, math.ceil(len(audio) / 160)))
            encoder_frames = min(hidden.shape[1], max(1, math.ceil(mel_frames / 2)))
            pooled.append(hidden[row_index, :encoder_frames].float().mean(dim=0))
        feature_batches.append(torch.stack(pooled).cpu().numpy())
        completed += len(audio_arrays)
        print(f"[{label}] extracted {completed}/{len(dataset)}", flush=True)

    features = np.concatenate(feature_batches, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        ids=np.asarray(ids),
        labels=np.asarray(labels or []),
        model_name=np.asarray(model_name),
        model_revision=np.asarray(model_revision),
    )
    return features, ids, labels


def build_classifier(c: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def predict_with_confidence(classifier, features: np.ndarray):
    probabilities = classifier.predict_proba(features)
    order = np.argsort(probabilities, axis=1)
    top_index = order[:, -1]
    second_index = order[:, -2]
    classes = np.asarray(classifier.classes_)
    predicted = classes[top_index]
    confidence = probabilities[np.arange(len(features)), top_index]
    margin = confidence - probabilities[np.arange(len(features)), second_index]
    return predicted, confidence, margin, probabilities


def evaluation_report(
    *,
    true_labels: list[str],
    predicted: np.ndarray,
    confidence: np.ndarray,
    languages: list[str],
) -> dict[str, Any]:
    truth = np.asarray(true_labels)
    correct = predicted == truth
    per_language = {}
    for language in languages:
        mask = truth == language
        per_language[language] = {
            "examples": int(mask.sum()),
            "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
        }
    confusion = {
        true: {
            guess: int(np.sum((truth == true) & (predicted == guess)))
            for guess in languages
        }
        for true in languages
    }
    gates = {}
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        gates[f"{threshold:.2f}"] = {
            "accepted": count,
            "coverage": float(accepted.mean()),
            "accuracy": float(correct[accepted].mean()) if count else None,
            "errors": int((~correct[accepted]).sum()) if count else 0,
        }
    return {
        "examples": len(true_labels),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "per_language": per_language,
        "confusion": confusion,
        "confidence_gates": gates,
    }


def write_predictions(
    *,
    path: Path,
    ids: list[str],
    true_labels: list[str] | None,
    predicted: np.ndarray,
    confidence: np.ndarray,
    margin: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    unknown_threshold: float,
) -> None:
    routed = np.where(confidence >= unknown_threshold, predicted, "unk")
    fields = [
        "ID",
        "language",
        "predicted_language",
        "confidence",
        "margin",
        *[f"p_{language}" for language in classes],
    ]
    if true_labels is not None:
        fields.insert(1, "true_language")
        fields.append("correct")

    path.parent.mkdir(parents=True, exist_ok=True)
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
            for class_index, language in enumerate(classes):
                row[f"p_{language}"] = f"{probabilities[row_index, class_index]:.8f}"
            if true_labels is not None:
                row["true_language"] = true_labels[row_index]
                row["correct"] = int(predicted[row_index] == true_labels[row_index])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.unknown_threshold <= 1.0:
        raise SystemExit("--unknown-threshold must be between 0 and 1")

    import joblib
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    languages = sorted(set(args.languages))
    language_set = set(languages)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = balanced_train_subset(
        load_split(args.dataset_dir, args.train_split),
        languages,
        args.max_train_per_language,
        args.seed,
    )
    eval_dataset = labeled_subset(
        load_split(args.dataset_dir, args.eval_split),
        language_set,
    )
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), args.max_eval_samples))
        )
    predict_dataset = (
        load_split(args.predict_dataset_dir, args.predict_split)
        if args.predict_dataset_dir is not None
        else None
    )
    if predict_dataset is not None and args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(
            range(min(len(predict_dataset), args.max_predict_samples))
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        revision=args.model_revision,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    train_features, train_ids, train_labels = extract_features(
        model=model,
        processor=processor,
        dataset=train_dataset,
        batch_size=args.batch_size,
        device=device,
        model_name=args.model_name,
        model_revision=args.model_revision,
        cache_path=output_dir / "train_features.npz",
        reuse_features=args.reuse_features,
        label="train",
    )
    eval_features, eval_ids, eval_labels = extract_features(
        model=model,
        processor=processor,
        dataset=eval_dataset,
        batch_size=args.batch_size,
        device=device,
        model_name=args.model_name,
        model_revision=args.model_revision,
        cache_path=output_dir / "validation_features.npz",
        reuse_features=args.reuse_features,
        label="validation",
    )
    if train_labels is None or eval_labels is None:
        raise RuntimeError("Train and validation features require language labels")

    classifier = build_classifier(args.c, args.seed)
    classifier.fit(train_features, train_labels)
    predicted, confidence, margin, probabilities = predict_with_confidence(
        classifier,
        eval_features,
    )
    write_predictions(
        path=output_dir / "validation_predictions.csv",
        ids=eval_ids,
        true_labels=eval_labels,
        predicted=predicted,
        confidence=confidence,
        margin=margin,
        probabilities=probabilities,
        classes=list(classifier.classes_),
        unknown_threshold=args.unknown_threshold,
    )

    report: dict[str, Any] = {
        "data_policy": "google/WaxalNLP train/validation only; Phase-2 audio inference only",
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "languages": languages,
        "train_examples": len(train_ids),
        "train_histogram": dict(Counter(train_labels)),
        "validation_examples": len(eval_ids),
        "validation_histogram": dict(Counter(eval_labels)),
        "max_train_per_language": args.max_train_per_language,
        "classifier": {
            "type": "standardized balanced logistic regression",
            "c": args.c,
            "seed": args.seed,
        },
        "validation": evaluation_report(
            true_labels=eval_labels,
            predicted=predicted,
            confidence=confidence,
            languages=languages,
        ),
    }

    predict_features = None
    predict_ids = None
    if predict_dataset is not None:
        predict_features, predict_ids, _ = extract_features(
            model=model,
            processor=processor,
            dataset=predict_dataset,
            batch_size=args.batch_size,
            device=device,
            model_name=args.model_name,
            model_revision=args.model_revision,
            cache_path=output_dir / "predict_features.npz",
            reuse_features=args.reuse_features,
            label="predict",
        )

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if args.fit_validation_for_predict:
        classifier = build_classifier(args.c, args.seed)
        classifier.fit(
            np.concatenate([train_features, eval_features], axis=0),
            train_labels + eval_labels,
        )
        report["final_fit_examples"] = len(train_labels) + len(eval_labels)
        report["final_fit_includes_validation"] = True
    else:
        report["final_fit_examples"] = len(train_labels)
        report["final_fit_includes_validation"] = False

    if predict_features is not None and predict_ids is not None:
        predicted, confidence, margin, probabilities = predict_with_confidence(
            classifier,
            predict_features,
        )
        write_predictions(
            path=output_dir / "phase2_predictions.csv",
            ids=predict_ids,
            true_labels=None,
            predicted=predicted,
            confidence=confidence,
            margin=margin,
            probabilities=probabilities,
            classes=list(classifier.classes_),
            unknown_threshold=args.unknown_threshold,
        )
        routed = np.where(confidence >= args.unknown_threshold, predicted, "unk")
        report["phase2_prediction"] = {
            "examples": len(predict_ids),
            "unknown_threshold": args.unknown_threshold,
            "histogram": dict(Counter(routed.tolist())),
            "top_prediction_histogram": dict(Counter(predicted.tolist())),
            "mean_confidence": float(confidence.mean()),
            "mean_margin": float(margin.mean()),
        }

    joblib.dump(classifier, output_dir / "classifier.joblib")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
