#!/usr/bin/env python3
"""Decode clips longer than Whisper's 30 s window by overlapped chunking.

Whisper's feature extractor pads/truncates every input to exactly 30 s, so a
35 s clip silently loses its tail — pure deletion error, the most expensive kind
for WER. The corrected Phase-2 test set contains 29 such clips (up to 35.2 s),
which the previous set did not.

Each long clip is split into overlapping windows (default 28 s window, 4 s
overlap), every window is decoded with the same forced language token as the
main pass, and the pieces are stitched by removing the duplicated words in the
overlap region (longest suffix/prefix word match, punctuation-insensitive).

Emits ID,Target only for the clips it processed, to be spliced into the main
decode with scripts/splice_predictions.py or a direct merge.

Usage:
  python scripts/decode_long_clips.py --model-name Sunbird/asr-whisper-large-v3-salt \
    --dataset-dir data/processed_phase2_v2 --split test --min-duration 30 \
    --language-csv outputs/analysis/phase2_v2_lid_fused.csv \
    --language-map ach=50357 nyn=50354 xog=50352 myx=50349 \
    --num-beams 5 --length-penalty 0.8 \
    --output outputs/predictions/phase2_v2_longclips.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def _key(word: str) -> str:
    return EDGE.sub("", word.lower())


def stitch(left: str, right: str, max_overlap_words: int = 25) -> str:
    """Join two chunk transcripts, dropping words duplicated across the overlap."""
    lw, rw = left.split(), right.split()
    if not lw:
        return right.strip()
    if not rw:
        return left.strip()
    lk = [_key(w) for w in lw]
    rk = [_key(w) for w in rw]
    best = 0
    limit = min(len(lw), len(rw), max_overlap_words)
    for n in range(limit, 1, -1):  # require >=2 words to call it an overlap
        if lk[-n:] == rk[:n]:
            best = n
            break
    return " ".join(lw + rw[best:]).strip()


def windows(duration: float, window: float, overlap: float) -> list[tuple[float, float]]:
    if duration <= window:
        return [(0.0, duration)]
    step = window - overlap
    out, start = [], 0.0
    while start < duration:
        end = min(start + window, duration)
        out.append((start, end))
        if end >= duration:
            break
        start += step
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--window", type=float, default=28.0)
    parser.add_argument("--overlap", type=float, default=4.0)
    parser.add_argument("--language-csv", type=Path, default=None)
    parser.add_argument("--language-map", nargs="*", default=None)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    remap: dict[str, str] = {}
    for spec in args.language_map or []:
        src, _, dst = spec.partition("=")
        remap[src] = dst
    routing: dict[str, str] = {}
    if args.language_csv:
        with args.language_csv.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                routing[row["ID"]] = (row.get("language") or "").strip()

    ds = load_from_disk(args.dataset_dir / "hf_dataset")[args.split]
    ids, durations = ds["ID"], ds["duration"]
    targets = [i for i, (_, d) in enumerate(zip(ids, durations)) if d > args.min_duration]
    print(f"{len(targets)} clips longer than {args.min_duration}s (of {len(ids)})")
    if not targets:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name, low_cpu_mem_usage=True)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model = model.to(device=device, dtype=dtype).eval()
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None
    tok = processor.tokenizer
    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    nots = tok.convert_tokens_to_ids("<|notimestamps|>")
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

    rows = []
    for n, idx in enumerate(targets, 1):
        example = ds[idx]
        example_id = example["ID"]
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        code = remap.get(routing.get(example_id, ""), routing.get(example_id, ""))
        prefix = None
        if code:
            token_id = int(code) if code.isdigit() else tok.convert_tokens_to_ids(f"<|{code}|>")
            if token_id and token_id != tok.unk_token_id:
                prefix = [token_id, transcribe, nots]

        pieces = []
        for start, end in windows(len(audio) / sr, args.window, args.overlap):
            seg = audio[int(start * sr) : int(end * sr)]
            feats = processor(seg, sampling_rate=sr, return_tensors="pt", padding="max_length", truncation=True)
            inp = feats["input_features"].to(device=device, dtype=dtype)
            gen = dict(input_features=inp, do_sample=False, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
            if args.length_penalty is not None:
                gen["length_penalty"] = args.length_penalty
            with torch.no_grad():
                if prefix is None:
                    out = model.generate(**gen)
                else:
                    try:
                        out = model.generate(**gen, forced_decoder_ids=[(i + 1, t) for i, t in enumerate(prefix)])
                    except (TypeError, ValueError):
                        dec = torch.tensor([[sot, *prefix]], device=device, dtype=torch.long)
                        out = model.generate(**gen, decoder_input_ids=dec)
            pieces.append(processor.batch_decode(out, skip_special_tokens=True)[0].strip())

        text = pieces[0]
        for piece in pieces[1:]:
            text = stitch(text, piece)
        rows.append({"ID": example_id, "Target": text})
        print(f"[{n}/{len(targets)}] {example_id} {durations[idx]:.1f}s -> {len(pieces)} chunks, {len(text)} chars", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} long-clip transcripts to {args.output}")


if __name__ == "__main__":
    main()
