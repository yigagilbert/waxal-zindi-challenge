#!/usr/bin/env python3
"""LID by forced-decode likelihood: decode clips under each candidate language
token and keep the highest-scoring transcript (beam sequences_scores).

Validate the selector on labeled clips FIRST (reports accuracy per language),
then apply to unlabeled clips (e.g. the Phase-2 'unk' residue).

Validate on the bench (known languages):
  python scripts/pick_language_by_likelihood.py \
    --model-name Sunbird/asr-whisper-large-v3-salt \
    --dataset-dir data/phase2_train --split validation --max-samples 200 \
    --candidates ach=50357 nyn=50354 xog=50352 myx=50349 \
    --validate-references data/phase2_train/validation.csv \
    --output outputs/analysis/likelihood_lid_bench.csv

Apply to the Phase-2 unk clips:
  python scripts/pick_language_by_likelihood.py \
    --model-name Sunbird/asr-whisper-large-v3-salt \
    --dataset-dir data/processed_phase2 --split test \
    --routing outputs/analysis/phase2_language_clusters.csv --only-language unk \
    --candidates ach=50357 nyn=50354 xog=50352 myx=50349 \
    --output outputs/predictions/phase2_unk_likelihood.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidates", nargs="+", required=True, help="lang=token_id pairs.")
    parser.add_argument("--routing", type=Path, default=None, help="ID,language table for --only-language filtering.")
    parser.add_argument("--only-language", default=None, help="Keep only clips with this routing label (e.g. unk).")
    parser.add_argument("--validate-references", type=Path, default=None, help="ID,language[,Target] CSV with true labels.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    candidates: dict[str, int] = {}
    for spec in args.candidates:
        lang, _, tid = spec.partition("=")
        candidates[lang] = int(tid)

    ds = load_from_disk(args.dataset_dir / "hf_dataset")[args.split]
    if args.routing is not None and args.only_language:
        routing = {r["ID"]: r["language"] for r in csv.DictReader(args.routing.open(encoding="utf-8-sig"))}
        wanted = {k for k, v in routing.items() if v == args.only_language}
        keep = [i for i, example_id in enumerate(ds["ID"]) if example_id in wanted]
        ds = ds.select(keep)
    if args.max_samples is not None:
        ds = ds.select(range(min(len(ds), args.max_samples)))
    print(f"{len(ds)} clips x {len(candidates)} candidate languages")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name, low_cpu_mem_usage=True)
    model = model.to(device=device, dtype=dtype).eval()
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None
    tok = processor.tokenizer
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    nots = tok.convert_tokens_to_ids("<|notimestamps|>")

    all_ids = ds["ID"]
    best: dict[str, tuple[float, str, str]] = {}  # ID -> (score, lang, text)
    scores_log: dict[str, dict[str, float]] = {k: {} for k in all_ids}

    for lang, lang_id in candidates.items():
        done = 0
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start : start + args.batch_size]
            audios = [a["array"] for a in batch["audio"]]
            feats = processor(audios, sampling_rate=16_000, return_tensors="pt",
                              padding="max_length", truncation=True, return_attention_mask=True)
            input_features = feats["input_features"].to(device=device, dtype=dtype)
            attention_mask = feats.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            n = input_features.shape[0]
            gen_kwargs = dict(
                input_features=input_features, attention_mask=attention_mask,
                do_sample=False, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True, output_scores=True,
            )
            with torch.no_grad():
                try:
                    out = model.generate(
                        **gen_kwargs,
                        forced_decoder_ids=[(1, lang_id), (2, transcribe), (3, nots)],
                    )
                except (TypeError, ValueError):
                    dec = torch.tensor([[sot, lang_id, transcribe, nots]] * n, device=device, dtype=torch.long)
                    out = model.generate(**gen_kwargs, decoder_input_ids=dec)
            texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
            seq_scores = out.sequences_scores.tolist()
            for example_id, text, score in zip(batch["ID"], texts, seq_scores, strict=True):
                scores_log[example_id][lang] = round(score, 4)
                if example_id not in best or score > best[example_id][0]:
                    best[example_id] = (score, lang, text.strip())
            done += n
            print(f"[{lang}] {done}/{len(ds)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "language", "score", "margin", "Target"])
        w.writeheader()
        for example_id in all_ids:
            score, lang, text = best[example_id]
            others = [v for k, v in scores_log[example_id].items() if k != lang]
            margin = round(score - max(others), 4) if others else 0.0
            w.writerow({"ID": example_id, "language": lang, "score": round(score, 4),
                        "margin": margin, "Target": text})
    print(f"Wrote {len(all_ids)} rows to {args.output}")

    if args.validate_references is not None:
        truth = {r["ID"]: r["language"] for r in csv.DictReader(args.validate_references.open(encoding="utf-8-sig"))}
        from collections import Counter, defaultdict
        per_lang = defaultdict(Counter)
        for example_id in all_ids:
            true_lang = truth.get(example_id)
            if true_lang:
                per_lang[true_lang][best[example_id][1]] += 1
        correct = total = 0
        for true_lang, picks in sorted(per_lang.items()):
            n = sum(picks.values())
            c = picks.get(true_lang, 0)
            correct += c; total += n
            print(f"  {true_lang}: {c}/{n} = {c/n:.1%}   picks: {dict(picks.most_common())}")
        print(f"selector accuracy: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    main()
