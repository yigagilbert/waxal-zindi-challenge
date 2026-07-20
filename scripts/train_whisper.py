#!/usr/bin/env python3
"""Fine-tune Whisper models on prepared WAXAL data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.config import load_config  # noqa: E402
from waxal.scoring import compute_group_metrics  # noqa: E402
from waxal.utils import save_experiment_log  # noqa: E402


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad Whisper input features and text labels separately."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        import torch

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


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
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperFeatureExtractor,
            WhisperForConditionalGeneration,
            WhisperProcessor,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `uv sync` before running Whisper training.") from exc

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

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name, language=None, task="transcribe")
    # Force fp32 load. Recent transformers loads whisper-large-v3 in its native fp16, which
    # crashes eval generate() ("Input type (float) and bias type (c10::Half)") because the
    # fp32 audio hits fp16 conv weights outside autocast. fp32 params + bf16 autocast (from
    # training_args) keeps training fast and makes generation dtype-consistent.
    import torch as _torch

    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=_torch.float32)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if config["training_args"].get("gradient_checkpointing", False):
        model.config.use_cache = False

    def prepare_example(example):
        audio = example["audio"]
        features = feature_extractor(
            audio["array"],
            sampling_rate=16_000,
            do_normalize=True,
        ).input_features[0]
        labels = processor.tokenizer(str(example["transcription"])).input_ids
        return {
            "input_features": features,
            "labels": labels,
            "language": example["language"],
            "ID": example["ID"],
        }

    remove_columns = [c for c in train_ds.column_names if c not in {"language", "ID"}]
    train_data = train_ds.map(prepare_example, remove_columns=remove_columns)
    eval_data = eval_ds.map(prepare_example, remove_columns=remove_columns)

    if config.get("lora", {}).get("enabled", False):
        try:
            import peft
        except ImportError as exc:
            raise RuntimeError("peft is required for LoRA fine-tuning.") from exc
        lora_cfg = dict(config["lora"])
        lora_cfg.pop("enabled", None)
        model.enable_input_require_grads()
        model = peft.get_peft_model(model, peft.LoraConfig(**lora_cfg))
        model.print_trainable_parameters()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return compute_group_metrics(
            label_str,
            pred_str,
            normalization=config.get("evaluation", {}).get("normalization", "starter_lower"),
        )

    training_kwargs = dict(config["training_args"])
    training_kwargs.setdefault("predict_with_generate", True)
    training_kwargs["push_to_hub"] = False
    training_kwargs["report_to"] = config.get("tracking", {}).get("report_to", "none")
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if config.get("evaluation", {}).get("skip_final_eval", False):
        metrics = {"final_eval_skipped": True}
        print("Final trainer.evaluate() skipped by config.")
    else:
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
