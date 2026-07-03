#!/usr/bin/env python3
"""Fine-tune XLS-R / wav2vec2 CTC models on prepared WAXAL data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.config import load_config  # noqa: E402
from waxal.data import read_csv_dicts  # noqa: E402
from waxal.scoring import compute_group_metrics  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import ensure_dir, save_experiment_log  # noqa: E402


@dataclass
class DataCollatorCTCWithPadding:
    """Pad CTC audio inputs and labels separately."""

    processor: Any
    padding: bool | str = True

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def build_ctc_vocab(texts: list[str], output_dir: Path, normalization: str) -> Path:
    """Build and save a character-level CTC vocabulary."""
    chars = sorted({char for text in texts for char in normalize_text(text, normalization)})
    vocab = {char: idx for idx, char in enumerate(chars) if char != " "}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    out = output_dir / "vocab.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return out


def str_to_bool(value: str | bool | None) -> bool | None:
    """Parse optional bool CLI/config values."""
    if value is None or isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    """Load a quality-bucket manifest with ID, Target, and language columns."""
    columns, rows, bad = read_csv_dicts(path)
    if bad:
        raise ValueError(f"{path} has malformed rows with extra fields: {bad[:3]}")
    required = {"ID", "Target", "language"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}; got {columns}")
    return rows


def manifest_with_optional_medium(path: Path, include_medium: bool) -> list[dict[str, str]]:
    """Load a train manifest, optionally appending sibling medium_train.csv rows."""
    rows = load_manifest_rows(path)
    if include_medium:
        medium_path = path.with_name(path.name.replace("clean_", "medium_", 1))
        if medium_path != path and medium_path.exists():
            rows.extend(load_manifest_rows(medium_path))
            print(f"Included medium bucket manifest: {medium_path}")
        else:
            print(f"Medium bucket requested but not found beside manifest: {medium_path}")
    seen: set[str] = set()
    deduped = []
    for row in rows:
        if row["ID"] in seen:
            continue
        seen.add(row["ID"])
        deduped.append(row)
    return deduped


def apply_train_manifest(train_ds, manifest_rows: list[dict[str, str]]):
    """Filter a prepared train split to manifest IDs and attach manifest labels."""
    manifest_by_id = {row["ID"]: row for row in manifest_rows}
    wanted = set(manifest_by_id)
    filtered = train_ds.filter(lambda row, ids=wanted: row["ID"] in ids)
    found = set(filtered["ID"]) if len(filtered) else set()
    missing = sorted(wanted - found)
    if missing:
        print(f"WARNING: {len(missing)} manifest IDs were not found in the prepared train split. First examples: {missing[:10]}")

    def attach_manifest_label(example):
        manifest_row = manifest_by_id[example["ID"]]
        return {
            "transcription": manifest_row["Target"],
            "quality_bucket": manifest_row.get("quality_bucket", ""),
        }

    return filtered.map(attach_manifest_label)


def filter_duration(ds, *, min_duration: float | None, max_duration: float | None):
    """Filter by duration when the prepared dataset has duration metadata."""
    if min_duration is None and max_duration is None:
        return ds
    if "duration" not in ds.column_names:
        print("Duration filtering requested but dataset has no duration column; keeping all examples.")
        return ds

    def keep(row):
        duration = row.get("duration")
        if duration is None:
            return True
        if min_duration is not None and duration < min_duration:
            return False
        if max_duration is not None and duration > max_duration:
            return False
        return True

    return ds.filter(keep)


def balance_by_language(ds, *, seed: int):
    """Oversample minority languages in a deterministic Dataset.select pass."""
    languages = list(ds["language"])
    indices_by_language: dict[str, list[int]] = {}
    for idx, language in enumerate(languages):
        indices_by_language.setdefault(language, []).append(idx)
    if len(indices_by_language) <= 1:
        return ds
    target = max(len(indices) for indices in indices_by_language.values())
    balanced_indices: list[int] = []
    for language, indices in sorted(indices_by_language.items()):
        repeats, remainder = divmod(target, len(indices))
        balanced_indices.extend(indices * repeats)
        balanced_indices.extend(indices[:remainder])
        print(f"Language-balanced sampling: {language} {len(indices)} -> {target}")
    return ds.select(balanced_indices).shuffle(seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--include-medium", type=str_to_bool, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--language-balanced-sampling", type=str_to_bool, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
        from transformers import (
            AutoModelForCTC,
            Trainer,
            TrainingArguments,
            Wav2Vec2CTCTokenizer,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Processor,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `uv sync` before running XLS-R training.") from exc

    config = load_config(args.config)
    if args.max_steps is not None:
        config.setdefault("training_args", {})["max_steps"] = args.max_steps
    if args.seed is not None:
        config.setdefault("training_args", {})["seed"] = args.seed
    if args.output_dir is not None:
        config.setdefault("training_args", {})["output_dir"] = str(args.output_dir)
    set_seed(int(config.get("training_args", {}).get("seed", 42)))
    model_name = config["model"]["name"]
    data_config = config.get("data", {})
    dataset_dir = args.dataset_dir or Path(data_config.get("dataset_dir", "data/processed"))
    dataset_dict = load_from_disk(Path(dataset_dir) / "hf_dataset")
    train_ds = dataset_dict["train"]
    eval_ds = dataset_dict["validation"]
    languages = data_config.get("languages")
    if languages:
        language_set = set(languages)
        train_ds = train_ds.filter(lambda row: row["language"] in language_set)
        eval_ds = eval_ds.filter(lambda row: row["language"] in language_set)

    train_manifest = args.train_manifest or data_config.get("train_manifest")
    include_medium = args.include_medium
    if include_medium is None:
        include_medium = bool(data_config.get("include_medium", False))
    if train_manifest:
        manifest_rows = manifest_with_optional_medium(Path(train_manifest), include_medium)
        train_ds = apply_train_manifest(train_ds, manifest_rows)

    min_duration = args.min_duration
    if min_duration is None:
        min_duration = data_config.get("min_duration")
    max_duration = args.max_duration
    if max_duration is None:
        max_duration = data_config.get("max_duration")
    train_ds = filter_duration(train_ds, min_duration=min_duration, max_duration=max_duration)
    eval_ds = filter_duration(eval_ds, min_duration=min_duration, max_duration=max_duration)

    language_balanced_sampling = args.language_balanced_sampling
    if language_balanced_sampling is None:
        language_balanced_sampling = bool(data_config.get("language_balanced_sampling", False))
    if language_balanced_sampling:
        train_ds = balance_by_language(train_ds, seed=int(config.get("training_args", {}).get("seed", 42)))

    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(len(train_ds), args.max_train_samples)))
    if args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(len(eval_ds), args.max_eval_samples)))

    output_dir = ensure_dir(config["training_args"]["output_dir"])
    normalization = config.get("text", {}).get("normalization", "language_safe")
    vocab_path = build_ctc_vocab(list(train_ds["transcription"]), output_dir, normalization)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_example(example):
        audio = example["audio"]
        inputs = processor(audio["array"], sampling_rate=16_000)
        labels = processor.tokenizer(normalize_text(example["transcription"], normalization)).input_ids
        return {
            "input_values": inputs.input_values[0],
            "labels": labels,
            "language": example["language"],
            "ID": example["ID"],
        }

    remove_columns = [c for c in train_ds.column_names if c not in {"language", "ID"}]
    train_data = train_ds.map(prepare_example, remove_columns=remove_columns)
    eval_data = eval_ds.map(prepare_example, remove_columns=remove_columns)

    model = AutoModelForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    if config.get("model", {}).get("freeze_feature_encoder", True):
        model.freeze_feature_encoder()

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    def compute_metrics(eval_pred):
        import numpy as np

        pred_logits = eval_pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        return compute_group_metrics(
            label_str,
            pred_str,
            normalization=config.get("evaluation", {}).get("normalization", normalization),
        )

    training_kwargs = dict(config["training_args"])
    training_kwargs["push_to_hub"] = False
    training_kwargs["report_to"] = config.get("tracking", {}).get("report_to", "none")
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    print(metrics)

    save_experiment_log(
        config.get("tracking", {}).get("local_dir", "outputs/experiments"),
        run_name=config.get("run_name", Path(args.config).stem),
        config=config,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
