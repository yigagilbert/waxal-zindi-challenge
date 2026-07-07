#!/usr/bin/env python3
"""Check that a Hugging Face ASR model can be loaded and run on WAXAL audio."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import TARGET_LANGUAGES  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--languages", nargs="*", default=["lin"], choices=list(TARGET_LANGUAGES))
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-type", choices=["auto", "ctc", "whisper"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/lingala_models/alvin_access_check.json"))
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
            "HUGGINGFACE_HUB_TOKEN is visible. Run `hf auth login` or export HF_TOKEN, "
            "then retry. The token value is never printed by this script."
        ) from exc
    raise RuntimeError(f"{step} failed: {type(exc).__name__}: {exc}") from exc


def load_dataset_slice(dataset_dir: Path, split: str, languages: list[str], max_samples: int):
    from datasets import load_from_disk

    dataset_path = dataset_dir / "hf_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared dataset missing: {dataset_path}")
    dataset_dict = load_from_disk(dataset_path)
    ds = dataset_dict[split]
    if languages:
        wanted = set(languages)
        ds = ds.filter(lambda row: row["language"] in wanted)
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    if len(ds) == 0:
        raise ValueError(f"No examples selected for split={split}, languages={languages}")
    return ds


def run_ctc_check(args: argparse.Namespace, ds) -> tuple[dict[str, Any], list[dict[str, str]]]:
    import torch
    from transformers import AutoConfig, AutoModelForCTC, AutoProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    status: dict[str, Any] = {"model_type": "ctc", "device": device, "steps": {}}

    try:
        config = AutoConfig.from_pretrained(args.model_name)
    except Exception as exc:
        raise_with_token_hint("config download", exc)
    status["steps"]["config_download"] = "ok"
    status["architecture"] = getattr(config, "architectures", None)
    status["model_type_name"] = getattr(config, "model_type", None)
    status["vocab_size"] = getattr(config, "vocab_size", None)

    try:
        processor = AutoProcessor.from_pretrained(args.model_name)
    except Exception as exc:
        raise_with_token_hint("processor/tokenizer download", exc)
    status["steps"]["processor_download"] = "ok"
    tokenizer = getattr(processor, "tokenizer", None)
    status["processor_class"] = processor.__class__.__name__
    status["tokenizer_class"] = tokenizer.__class__.__name__ if tokenizer is not None else ""
    status["word_delimiter_token"] = getattr(tokenizer, "word_delimiter_token", None)
    status["has_lm_decoder"] = hasattr(processor, "decoder")

    try:
        model = AutoModelForCTC.from_pretrained(args.model_name).to(device)
    except Exception as exc:
        raise_with_token_hint("model weights download", exc)
    status["steps"]["model_weights_download"] = "ok"
    status["num_parameters"] = sum(param.numel() for param in model.parameters())

    model.eval()
    rows: list[dict[str, str]] = []
    for example in ds:
        audio = example["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio.get("sampling_rate", 16_000),
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = decode_ctc_batch(
            processor,
            logits,
            pred_ids,
            getattr(tokenizer, "word_delimiter_token", None),
        )[0]
        rows.append({"ID": example["ID"], "Target": text, "language": example["language"]})
    status["steps"]["inference"] = "ok"
    return status, rows


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


def run_whisper_check(args: argparse.Namespace, ds) -> tuple[dict[str, Any], list[dict[str, str]]]:
    import torch
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    status: dict[str, Any] = {"model_type": "whisper", "device": device, "steps": {}}

    try:
        config = AutoConfig.from_pretrained(args.model_name)
    except Exception as exc:
        raise_with_token_hint("config download", exc)
    status["steps"]["config_download"] = "ok"
    status["architecture"] = getattr(config, "architectures", None)
    status["model_type_name"] = getattr(config, "model_type", None)

    try:
        processor = AutoProcessor.from_pretrained(args.model_name)
    except Exception as exc:
        raise_with_token_hint("processor/tokenizer download", exc)
    status["steps"]["processor_download"] = "ok"

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    except Exception as exc:
        raise_with_token_hint("model weights download", exc)
    status["steps"]["model_weights_download"] = "ok"
    status["num_parameters"] = sum(param.numel() for param in model.parameters())

    model.eval()
    rows: list[dict[str, str]] = []
    for example in ds:
        audio = example["audio"]
        inputs = processor(
            audio["array"],
            sampling_rate=audio.get("sampling_rate", 16_000),
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
                num_beams=1,
                max_new_tokens=256,
            )
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        rows.append({"ID": example["ID"], "Target": text, "language": example["language"]})
    status["steps"]["inference"] = "ok"
    return status, rows


def main() -> None:
    args = parse_args()
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers with `uv sync` before running this check.") from exc

    ds = load_dataset_slice(args.dataset_dir, args.split, args.languages, args.max_samples)
    model_type = infer_model_type(args.model_name, args.model_type)
    if model_type == "whisper":
        status, predictions = run_whisper_check(args, ds)
    else:
        status, predictions = run_ctc_check(args, ds)

    payload = {
        "model_name": args.model_name,
        "access_status": "ok",
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "languages": args.languages,
        "max_samples": args.max_samples,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "hf_token_available": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "status": status,
        "sample_predictions": predictions,
    }
    json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
