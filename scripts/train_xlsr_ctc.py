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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
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
    dataset_dir = args.dataset_dir or Path(config.get("data", {}).get("dataset_dir", "data/processed"))
    dataset_dict = load_from_disk(Path(dataset_dir) / "hf_dataset")
    train_ds = dataset_dict["train"]
    eval_ds = dataset_dict["validation"]
    languages = config.get("data", {}).get("languages")
    if languages:
        language_set = set(languages)
        train_ds = train_ds.filter(lambda row: row["language"] in language_set)
        eval_ds = eval_ds.filter(lambda row: row["language"] in language_set)
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
