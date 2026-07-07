#!/usr/bin/env python3
"""Run Lingala ASR model inference on WAXAL splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import write_csv_rows  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import clean_name, json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--language", default="lin", choices=["lin"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--token-stats-output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-type", choices=["auto", "ctc", "whisper"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_model_type(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if "whisper" in model_name.lower():
        return "whisper"
    return "ctc"


def likely_gated_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("401", "403", "gated", "private", "unauthorized", "forbidden"))


def raise_with_token_hint(step: str, exc: Exception) -> None:
    if likely_gated_error(exc) and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError(
            f"{step} failed with a likely gated/private-model error, and no HF_TOKEN or "
            "HUGGINGFACE_HUB_TOKEN is visible. Run `hf auth login` or export HF_TOKEN, then retry."
        ) from exc
    raise RuntimeError(f"{step} failed: {type(exc).__name__}: {exc}") from exc


def load_lingala_split(dataset_dir: Path, split: str, max_samples: int | None):
    from datasets import load_from_disk

    dataset_path = dataset_dir / "hf_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared dataset missing: {dataset_path}")
    dataset_dict = load_from_disk(dataset_path)
    ds = dataset_dict[split]
    ds = ds.filter(lambda row: row["language"] == "lin")
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    if len(ds) == 0:
        raise ValueError(f"No Lingala examples selected from split={split}")
    return ds


def clean_ctc_text(text: str, word_delimiter: str | None) -> str:
    delimiter = word_delimiter or "|"
    return normalize_text(str(text).replace(delimiter, " "), "raw")


def decode_ctc_batch(processor, logits, pred_ids, word_delimiter: str | None) -> list[str]:
    """Decode CTC outputs for both plain and LM-backed Wav2Vec2 processors."""
    if hasattr(processor, "decoder"):
        decoded = processor.batch_decode(logits.detach().float().cpu().numpy())
        texts = decoded.text if hasattr(decoded, "text") else decoded["text"]
    else:
        texts = processor.batch_decode(pred_ids)
    return [clean_ctc_text(text, word_delimiter) for text in texts]


def token_stat_row(
    example_id: str,
    logits,
    pred_ids,
    *,
    pad_token_id: int | None,
    blank_token_id: int | None,
    prediction: str,
) -> dict[str, str]:
    import torch

    token_ids = pred_ids.detach().cpu().tolist()
    probs = torch.softmax(logits.detach().float().cpu(), dim=-1)
    max_probs = probs.max(dim=-1).values.tolist()
    blank_id = blank_token_id if blank_token_id is not None else pad_token_id
    blank_frames = sum(1 for token_id in token_ids if blank_id is not None and token_id == blank_id)
    repeated_frames = 0
    previous = None
    for token_id in token_ids:
        if token_id == previous:
            repeated_frames += 1
        previous = token_id
    num_frames = len(token_ids)
    return {
        "ID": example_id,
        "num_frames": str(num_frames),
        "blank_token_id": "" if blank_id is None else str(blank_id),
        "blank_frames": str(blank_frames),
        "blank_ratio": f"{blank_frames / num_frames:.6f}" if num_frames else "",
        "repeated_frame_ratio": f"{repeated_frames / num_frames:.6f}" if num_frames else "",
        "mean_max_probability": f"{sum(max_probs) / len(max_probs):.6f}" if max_probs else "",
        "min_max_probability": f"{min(max_probs):.6f}" if max_probs else "",
        "max_max_probability": f"{max(max_probs):.6f}" if max_probs else "",
        "prediction_chars": str(len(prediction)),
        "prediction_words": str(len(prediction.split())),
    }


def write_token_stats(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_ctc(args: argparse.Namespace, ds) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        processor = AutoProcessor.from_pretrained(args.model_name)
        model = AutoModelForCTC.from_pretrained(args.model_name).to(device)
    except Exception as exc:
        raise_with_token_hint("CTC model/processor load", exc)
    model.eval()

    tokenizer = getattr(processor, "tokenizer", None)
    word_delimiter = getattr(tokenizer, "word_delimiter_token", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    blank_token_id = getattr(model.config, "pad_token_id", pad_token_id)

    rows: list[dict[str, str]] = []
    token_stats: list[dict[str, str]] = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        audios = [audio["array"] for audio in batch["audio"]]
        sample_rates = {audio.get("sampling_rate", 16_000) for audio in batch["audio"]}
        if sample_rates != {16_000}:
            print(f"WARNING: expected 16 kHz audio, got sample rates {sorted(sample_rates)}", file=sys.stderr)
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
        predictions = decode_ctc_batch(processor, logits, pred_ids, word_delimiter)
        for idx, (example_id, pred) in enumerate(zip(batch["ID"], predictions, strict=True)):
            rows.append({"ID": example_id, "Target": pred, "language": "lin", "model": args.model_name})
            if args.token_stats_output:
                token_stats.append(
                    token_stat_row(
                        example_id,
                        logits[idx],
                        pred_ids[idx],
                        pad_token_id=pad_token_id,
                        blank_token_id=blank_token_id,
                        prediction=pred,
                    )
                )
        print(f"Predicted {len(rows)}/{len(ds)}", flush=True)

    meta = {
        "model_type": "ctc",
        "device": device,
        "processor_class": processor.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__ if tokenizer is not None else "",
        "word_delimiter_token": word_delimiter,
        "pad_token_id": pad_token_id,
        "blank_token_id": blank_token_id,
        "has_lm_decoder": hasattr(processor, "decoder"),
    }
    return rows, token_stats, meta


def run_whisper(args: argparse.Namespace, ds) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    try:
        processor = AutoProcessor.from_pretrained(args.model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    except Exception as exc:
        raise_with_token_hint("Whisper model/processor load", exc)
    model.eval()
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None
    if hasattr(model.config, "suppress_tokens"):
        model.config.suppress_tokens = []

    rows: list[dict[str, str]] = []
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
        input_features = inputs["input_features"].to(device=device, dtype=next(model.parameters()).dtype)
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
            rows.append({"ID": example_id, "Target": pred.strip(), "language": "lin", "model": args.model_name})
        print(f"Predicted {len(rows)}/{len(ds)}", flush=True)
    return rows, [], {"model_type": "whisper", "device": device}


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = (
            Path("outputs/lingala_models")
            / f"{clean_name(args.model_name)}_{args.split}_lingala.csv"
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
    if args.token_stats_output and args.token_stats_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Token stats output exists: {args.token_stats_output}. Pass --overwrite to replace it."
        )

    ds = load_lingala_split(args.dataset_dir, args.split, args.max_samples)
    model_type = infer_model_type(args.model_name, args.model_type)
    if model_type == "whisper":
        rows, token_stats, meta = run_whisper(args, ds)
    else:
        rows, token_stats, meta = run_ctc(args, ds)

    write_csv_rows(output, rows, ["ID", "Target", "language", "model"])
    if args.token_stats_output:
        write_token_stats(args.token_stats_output, token_stats)
    summary_path = output.with_suffix(".summary.json")
    text_lengths = [len(normalize_text(row["Target"], "raw")) for row in rows]
    summary = {
        "model_name": args.model_name,
        "split": args.split,
        "language": args.language,
        "dataset_dir": str(args.dataset_dir),
        "num_examples": len(rows),
        "output": str(output),
        "token_stats_output": str(args.token_stats_output) if args.token_stats_output else "",
        "meta": meta,
        "prediction_length": {
            "mean_chars": sum(text_lengths) / len(text_lengths) if text_lengths else math.nan,
            "min_chars": min(text_lengths) if text_lengths else None,
            "max_chars": max(text_lengths) if text_lengths else None,
        },
    }
    json_dump(summary, summary_path)
    print(f"Wrote predictions to {output}")
    if args.token_stats_output:
        print(f"Wrote token stats to {args.token_stats_output}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
