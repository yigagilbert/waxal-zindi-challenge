#!/usr/bin/env python3
"""Pseudo-label unlabeled WAXAL clips with the champion (greedy + beam+LM).

Language is known per clip (unlabeled shards are per-language), so no LID.
Emits BOTH the greedy and beam+LM transcripts: their agreement is the
reference-free confidence signal used downstream to filter pseudo-labels
(noisy-student style — train only on clips the teacher is sure about).

Usage:
  python scripts/pseudo_label_decode.py \
    --manifest data/unlabeled_linsna/manifest.csv \
    --checkpoint champion_repo/checkpoint-24000 --vocab-path champion_repo/vocab.json \
    --kenlm-dir champion_repo/lm_phase2 \
    --params-json outputs/analysis/best_decode_params.json \
    --output outputs/day4_h100/pseudo_labels_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_xlsr_inference import (  # noqa: E402
    build_ctc_labels,
    build_processor,
    clean_ctc_prediction,
    resolve_vocab_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--kenlm-dir", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from pyctcdecode import build_ctcdecoder
    from transformers import AutoModelForCTC

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    done: set[str] = set()
    if args.output.exists():  # resumable
        with args.output.open(encoding="utf-8") as f:
            done = {r["ID"] for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} already decoded")
    todo = [r for r in rows if r["ID"] not in done]
    print(f"clips to decode: {len(todo)} of {len(rows)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = build_processor(resolve_vocab_path(args.checkpoint, args.vocab_path))
    labels = build_ctc_labels(processor.tokenizer)
    model = AutoModelForCTC.from_pretrained(args.checkpoint).to(device).eval()

    params = json.loads(args.params_json.read_text())
    decoders = {}
    for lang in sorted({r["language"] for r in todo}):
        binary = args.kenlm_dir / f"{lang}_5gram.binary"
        unigrams_file = args.kenlm_dir / f"{lang}.txt"
        unigrams = sorted({w for line in unigrams_file.open(encoding="utf-8", errors="ignore")
                           for w in line.split()})
        p = params.get(lang, {"alpha": 0.5, "beta": 1.5})
        decoders[lang] = build_ctcdecoder(
            labels, kenlm_model_path=str(binary), unigrams=unigrams,
            alpha=float(p["alpha"]), beta=float(p["beta"]),
        )
        print(f"decoder[{lang}]: alpha={p['alpha']} beta={p['beta']} unigrams={len(unigrams)}")

    mode = "a" if done else "w"
    with args.output.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "language", "duration", "greedy", "beam"])
        if mode == "w":
            writer.writeheader()
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            audios = [sf.read(r["path"], dtype="float32")[0] for r in batch]
            inputs = processor(audios, sampling_rate=16_000, return_tensors="pt",
                               padding=True, return_attention_mask=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            lengths = model._get_feat_extract_output_lengths(inputs["attention_mask"].sum(-1))
            for i, r in enumerate(batch):
                lg = logits[i, : int(lengths[i])].float().cpu().numpy()
                greedy_ids = lg.argmax(-1)
                greedy = clean_ctc_prediction(
                    processor.tokenizer.decode(greedy_ids),
                    processor.tokenizer.word_delimiter_token,
                )
                log_probs = lg - np.logaddexp.reduce(lg, axis=-1, keepdims=True)
                beam = decoders[r["language"]].decode(log_probs, beam_width=args.beam_width).strip()
                writer.writerow({"ID": r["ID"], "language": r["language"],
                                 "duration": r["duration"], "greedy": greedy, "beam": beam})
            if (start // args.batch_size) % 25 == 0:
                f.flush()
                print(f"decoded {start + len(batch)}/{len(todo)}", flush=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
