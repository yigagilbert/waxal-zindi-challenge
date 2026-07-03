#!/usr/bin/env python3
"""Run teacher ASR models on prepared WAXAL splits for label diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import TARGET_LANGUAGES, write_csv_rows  # noqa: E402
from waxal.utils import clean_name  # noqa: E402


MMS_LICENSE_WARNING = (
    "WARNING: MMS/Seamless-style Meta models are diagnostic only until license "
    "and challenge-rule suitability are explicitly approved. Do not use their "
    "outputs as final training labels in Phase 1."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--languages", nargs="*", default=None, choices=list(TARGET_LANGUAGES))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--model-type", choices=["auto", "whisper", "ctc"], default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file. By default existing teacher outputs are preserved.",
    )
    return parser.parse_args()


def load_split(dataset_dir: Path, split: str):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install with `uv sync`.") from exc

    dataset_path = dataset_dir / "hf_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared audio dataset not found: {dataset_path}. "
            "Run scripts/prepare_dataset.py without --metadata-only first."
        )
    dataset_dict = load_from_disk(dataset_path)
    if split not in dataset_dict:
        raise ValueError(f"Split {split!r} not found in {dataset_path}. Available: {list(dataset_dict.keys())}")
    return dataset_dict[split]


def likely_gated_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("gated", "private", "401", "403", "unauthorized", "forbidden"))


def ensure_token_hint(exc: Exception) -> None:
    if likely_gated_error(exc) and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError(
            "Model access failed and no HF_TOKEN/HUGGINGFACE_HUB_TOKEN is available. "
            "If this model is gated, run `hf auth login` or export HF_TOKEN before retrying."
        ) from exc
    raise exc


def infer_model_type(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = model_name.lower()
    if "whisper" in lowered:
        return "whisper"
    return "ctc"


def run_whisper_teacher(args: argparse.Namespace, ds) -> list[dict[str, str]]:
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required. Install with `uv sync`.") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(args.model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        ensure_token_hint(exc)

    model = model.to(device)
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
        if "input_features" not in inputs:
            raise ValueError(f"Expected input_features from Whisper processor, got {list(inputs)}")
        input_features = inputs["input_features"]
        if input_features.shape[-1] != 3000:
            raise ValueError(f"Whisper expects 3000 mel frames, got {tuple(input_features.shape)}")

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
        for example_id, language, pred in zip(batch["ID"], batch["language"], decoded, strict=True):
            rows.append(
                {
                    "ID": example_id,
                    "Target": pred.strip(),
                    "language": language,
                    "teacher_model": args.model_name,
                }
            )
        print(f"Teacher predicted {len(rows)}/{len(ds)}", flush=True)
    return rows


def ctc_device_arg(device: str | None) -> int:
    if device is None:
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    if device == "cpu":
        return -1
    if device.startswith("cuda"):
        if ":" in device:
            return int(device.split(":", 1)[1])
        return 0
    try:
        return int(device)
    except ValueError:
        return -1


def run_ctc_teacher(args: argparse.Namespace, ds) -> list[dict[str, str]]:
    print(MMS_LICENSE_WARNING, file=sys.stderr)
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("transformers is required. Install with `uv sync`.") from exc

    try:
        asr = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            device=ctc_device_arg(args.device),
        )
    except Exception as exc:
        ensure_token_hint(exc)

    rows: list[dict[str, str]] = []
    for idx, example in enumerate(ds, start=1):
        audio = example["audio"]
        try:
            result = asr({"array": audio["array"], "sampling_rate": audio["sampling_rate"]})
            text = result["text"] if isinstance(result, dict) else str(result)
        except Exception as exc:
            text = ""
            print(f"Teacher failed for {example['ID']}: {type(exc).__name__}: {exc}", file=sys.stderr)
        rows.append(
            {
                "ID": example["ID"],
                "Target": text.strip(),
                "language": example["language"],
                "teacher_model": args.model_name,
            }
        )
        if idx % max(args.batch_size, 1) == 0:
            print(f"Teacher predicted {idx}/{len(ds)}", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    output = args.output or Path("outputs/teachers") / f"{clean_name(args.model_name)}_{args.split}.csv"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Teacher output already exists: {output}. Pass --overwrite to replace it.")

    ds = load_split(args.dataset_dir, args.split)
    if args.languages:
        language_set = set(args.languages)
        ds = ds.filter(lambda row, wanted=language_set: row["language"] in wanted)
    if args.max_samples is not None:
        ds = ds.select(range(min(len(ds), args.max_samples)))
    if len(ds) == 0:
        raise ValueError("No examples selected for teacher inference.")

    if "sunbird/asr-whisper-large-v3-salt" in args.model_name.lower():
        selected_languages = sorted(set(ds["language"]))
        if selected_languages != ["lug"]:
            print(
                "WARNING: Sunbird SALT Whisper is primarily useful for Luganda diagnostics. "
                f"Selected languages: {selected_languages}",
                file=sys.stderr,
            )

    model_type = infer_model_type(args.model_name, args.model_type)
    if model_type == "whisper":
        rows = run_whisper_teacher(args, ds)
    else:
        rows = run_ctc_teacher(args, ds)

    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output, rows, ["ID", "Target", "language", "teacher_model"])
    print(f"Wrote teacher predictions to {output}")


if __name__ == "__main__":
    main()
