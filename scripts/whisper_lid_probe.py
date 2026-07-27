#!/usr/bin/env python3
"""Probe what languages a dataset's audio actually contains, via Whisper language detection.

Motivation (2026-07-27): Phase-2 clips confidently LID'd as Luganda by two independent
in-house methods turned out NOT to be Luganda by ear — suggesting Phase 2 contains languages
outside lin/lug/sna. Whisper's decoder emits a language token before transcribing; this
script captures it (plus a short transcript for eyeballing) for a sample of clips.

Caveats: stock whisper-large-v3 covers 99 languages but NOT Luganda (actual Luganda will land
on a neighbor like sw); fine-tunes (e.g. whisper-v3-ft-af51, already cached on the box) may
bias detection toward their training languages. Read the histogram as evidence, not truth.

Usage:
  python scripts/whisper_lid_probe.py \
    --model-name huwenjie333/whisper-v3-ft-af51 \
    --dataset-dir data/processed_phase2 --split test --max-samples 40 \
    --output outputs/analysis/phase2_whisper_lid_probe.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import write_csv_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="huwenjie333/whisper-v3-ft-af51")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--stride", type=int, default=None,
                        help="Sample every Nth clip instead of the first max-samples (spreads coverage).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/whisper_lid_probe.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dd = load_from_disk(args.dataset_dir / "hf_dataset")
    ds = dd[args.split] if hasattr(dd, "keys") else dd
    stride = args.stride or max(len(ds) // max(args.max_samples, 1), 1)
    idxs = list(range(0, len(ds), stride))[: args.max_samples]
    ds = ds.select(idxs)
    print(f"probing {len(ds)} clips (stride {stride})")

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name, low_cpu_mem_usage=True).to(device)
    model.eval()
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None  # let it auto-detect language
    model_dtype = next(model.parameters()).dtype

    rows = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        audios = [audio["array"] for audio in batch["audio"]]
        inputs = processor(audios, sampling_rate=16_000, return_tensors="pt",
                           padding="max_length", truncation=True, return_attention_mask=True)
        feats = inputs["input_features"].to(device=device, dtype=model_dtype)
        with torch.no_grad():
            gen = model.generate(input_features=feats, do_sample=False, max_new_tokens=args.max_new_tokens)
        for i, example_id in enumerate(batch["ID"]):
            raw = processor.tokenizer.decode(gen[i], skip_special_tokens=False)
            match = re.search(r"<\|([a-z]{2,3})\|>", raw)
            lang = match.group(1) if match else "?"
            text = processor.tokenizer.decode(gen[i], skip_special_tokens=True).strip()
            rows.append({"ID": example_id, "detected_lang": lang, "transcript": text[:160]})
        print(f"  probed {len(rows)}/{len(ds)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output, rows, ["ID", "detected_lang", "transcript"])
    counts = Counter(r["detected_lang"] for r in rows)
    print("detected language histogram:", dict(counts.most_common()))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
