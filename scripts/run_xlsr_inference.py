#!/usr/bin/env python3
"""Run XLS-R / wav2vec2 CTC inference on a prepared WAXAL split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import TARGET_LANGUAGES, write_csv_rows  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import clean_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--languages", nargs="*", default=None, choices=list(TARGET_LANGUAGES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_split(dataset_dir: Path, split: str):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install dependencies with `uv sync`.") from exc

    dataset_path = dataset_dir / "hf_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {dataset_path}. "
            "Run scripts/prepare_dataset.py first."
        )
    dataset_dict = load_from_disk(dataset_path)
    if split not in dataset_dict:
        raise ValueError(f"Split {split!r} not found in {dataset_path}. Available: {list(dataset_dict.keys())}")
    return dataset_dict[split]


def resolve_vocab_path(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit vocab path does not exist: {explicit}")
        return explicit

    candidates = [
        checkpoint / "vocab.json",
        checkpoint.parent / "vocab.json",
        checkpoint.parent.parent / "vocab.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find vocab.json. Pass --vocab-path explicitly. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def build_processor(vocab_path: Path):
    try:
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    except ImportError as exc:
        raise RuntimeError("transformers is required. Install dependencies with `uv sync`.") from exc

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
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def clean_ctc_prediction(text: str, word_delimiter: str | None) -> str:
    delimiter = word_delimiter or "|"
    text = str(text).replace(delimiter, " ")
    return normalize_text(text, "raw")


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCTC
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required. Install dependencies with `uv sync`.") from exc

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    vocab_path = resolve_vocab_path(args.checkpoint, args.vocab_path)
    processor = build_processor(vocab_path)

    ds = load_split(args.dataset_dir, args.split)
    if args.languages:
        language_set = set(args.languages)
        ds = ds.filter(lambda row, wanted=language_set: row["language"] in wanted)
    if args.max_samples is not None:
        ds = ds.select(range(min(len(ds), args.max_samples)))
    if len(ds) == 0:
        raise ValueError("No examples selected for inference.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCTC.from_pretrained(args.checkpoint).to(device)
    model.eval()

    output = args.output
    if output is None:
        output = Path("outputs/predictions") / f"{clean_name(str(args.checkpoint))}_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        audios = [audio["array"] for audio in batch["audio"]]
        inputs = processor(
            audios,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        decoded = processor.batch_decode(pred_ids)
        decoded = [clean_ctc_prediction(text, processor.tokenizer.word_delimiter_token) for text in decoded]
        for example_id, pred in zip(batch["ID"], decoded, strict=True):
            rows.append({"ID": example_id, "Target": pred})
        print(f"Predicted {len(rows)}/{len(ds)}", flush=True)

    write_csv_rows(output, rows, ["ID", "Target"])
    print(f"Wrote predictions to {output}")


if __name__ == "__main__":
    main()
