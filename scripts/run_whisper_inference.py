#!/usr/bin/env python3
"""Run deterministic Whisper-style inference on prepared WAXAL splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import write_csv_rows  # noqa: E402
from waxal.utils import clean_name  # noqa: E402


def load_split(dataset_dir: Path, split: str):
    """Load a prepared Hugging Face split from disk."""
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install requirements.txt first.") from exc
    dataset_path = dataset_dir / "hf_dataset"
    dataset_dict = load_from_disk(dataset_path)
    if split not in dataset_dict:
        raise ValueError(f"Split {split!r} not found in {dataset_path}. Available: {list(dataset_dict)}")
    return dataset_dict[split]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required. Install requirements.txt first.") from exc

    ds = load_split(args.dataset_dir, args.split)
    if args.languages:
        language_set = set(args.languages)
        ds = ds.filter(lambda row: row["language"] in language_set)
    if args.max_samples is not None:
        ds = ds.select(range(min(len(ds), args.max_samples)))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None
    if hasattr(model.config, "suppress_tokens"):
        model.config.suppress_tokens = []

    output = args.output
    if output is None:
        output = Path("outputs/predictions") / f"{clean_name(args.model_name)}_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        audios = [audio["array"] for audio in batch["audio"]]
        inputs = processor(
            audios,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        if "input_features" not in inputs:
            raise ValueError(f"Expected processor to return input_features, got keys: {list(inputs)}")
        
        input_features = inputs["input_features"]
        
        if input_features.shape[-1] != 3000:
            raise ValueError(
                f"Whisper expects input_features length 3000, "
                f"got shape {tuple(input_features.shape)}"
            )
        
        model_dtype = next(model.parameters()).dtype
        
        input_features = input_features.to(device=device, dtype=model_dtype)
        
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
        
        with torch.no_grad():
            generated = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        for example_id, pred in zip(batch["ID"], decoded, strict=True):
            rows.append({"ID": example_id, "Target": pred.strip()})
        print(f"Predicted {len(rows)}/{len(ds)}", flush=True)

    write_csv_rows(output, rows, ["ID", "Target"])
    print(f"Wrote predictions to {output}")


if __name__ == "__main__":
    main()

