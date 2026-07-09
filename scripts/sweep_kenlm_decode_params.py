#!/usr/bin/env python3
"""Sweep pyctcdecode alpha/beta per language using cached validation logits.

Runs the acoustic model exactly once per language, caches trimmed logits in
RAM, then decodes the alpha x beta grid on CPU and scores each combination
against the references.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from waxal.scoring import compute_group_metrics  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402

from run_xlsr_inference import (  # noqa: E402
    build_ctc_labels,
    build_processor,
    clean_ctc_prediction,
    load_split,
    resolve_vocab_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--languages", nargs="*", default=["lin", "lug", "sna"])
    parser.add_argument("--kenlm-dir", type=Path, default=Path("data/lm"))
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--betas", type=float, nargs="*", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--max-samples-per-language", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/kenlm_alpha_beta_sweep.json"))
    return parser.parse_args()


def compute_language_logits(model, processor, ds, *, batch_size: int, device):
    """Return per-example trimmed logits (float16 numpy) and normalized refs."""
    import numpy as np
    import torch

    logits_list, refs, ids = [], [], []
    for start in range(0, len(ds), batch_size):
        batch = ds[start : start + batch_size]
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
        input_lengths = inputs["attention_mask"].sum(-1)
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)
        for row_idx in range(logits.shape[0]):
            length = int(output_lengths[row_idx])
            logits_list.append(logits[row_idx, :length].detach().to(torch.float16).cpu().numpy())
        refs.extend(batch["transcription"])
        ids.extend(batch["ID"])
        print(f"  logits {len(logits_list)}/{len(ds)}", flush=True)
    return logits_list, refs, ids


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    from pyctcdecode import build_ctcdecoder
    from transformers import AutoModelForCTC

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = resolve_vocab_path(args.checkpoint, args.vocab_path)
    processor = build_processor(vocab_path)
    labels = build_ctc_labels(processor.tokenizer)
    word_delimiter = processor.tokenizer.word_delimiter_token

    model = AutoModelForCTC.from_pretrained(args.checkpoint).to(device)
    model.eval()

    full_ds = load_split(args.dataset_dir, args.split)
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "normalization": args.normalization,
        "beam_width": args.beam_width,
        "alphas": args.alphas,
        "betas": args.betas,
        "max_samples_per_language": args.max_samples_per_language,
        "languages": {},
    }

    for language in args.languages:
        kenlm_path = args.kenlm_dir / f"{language}_{args.order}gram.binary"
        if not kenlm_path.exists():
            print(f"WARNING: {kenlm_path} missing; skipping {language}")
            continue
        ds = full_ds.filter(lambda lang, want=language: lang == want, input_columns=["language"])
        if args.max_samples_per_language and len(ds) > args.max_samples_per_language:
            ds = ds.shuffle(seed=42).select(range(args.max_samples_per_language))
        print(f"== {language}: {len(ds)} examples ==")
        logits_list, raw_refs, _ = compute_language_logits(
            model, processor, ds, batch_size=args.batch_size, device=device
        )
        refs = [normalize_text(text, args.normalization) for text in raw_refs]

        grid = []
        greedy_preds = []
        for item in logits_list:
            ids = item.astype(np.float32).argmax(-1)
            greedy_preds.append(
                normalize_text(
                    clean_ctc_prediction(processor.tokenizer.decode(ids), word_delimiter),
                    args.normalization,
                )
            )
        greedy_metrics = compute_group_metrics(refs, greedy_preds, normalization=args.normalization)
        print(f"  greedy: combined={greedy_metrics['combined']:.4f}")

        best = None
        for alpha in args.alphas:
            for beta in args.betas:
                decoder = build_ctcdecoder(labels, kenlm_model_path=str(kenlm_path), alpha=alpha, beta=beta)
                preds = [
                    normalize_text(
                        clean_ctc_prediction(
                            decoder.decode(item.astype(np.float32), beam_width=args.beam_width),
                            word_delimiter,
                        ),
                        args.normalization,
                    )
                    for item in logits_list
                ]
                metrics = compute_group_metrics(refs, preds, normalization=args.normalization)
                entry = {
                    "alpha": alpha,
                    "beta": beta,
                    "wer": metrics["wer"],
                    "cer": metrics["cer"],
                    "combined": metrics["combined"],
                }
                grid.append(entry)
                if best is None or entry["combined"] < best["combined"]:
                    best = entry
                print(
                    f"  alpha={alpha} beta={beta}: wer={metrics['wer']:.4f} "
                    f"cer={metrics['cer']:.4f} combined={metrics['combined']:.4f}",
                    flush=True,
                )

        report["languages"][language] = {
            "num_examples": len(ds),
            "greedy": {
                "wer": greedy_metrics["wer"],
                "cer": greedy_metrics["cer"],
                "combined": greedy_metrics["combined"],
            },
            "grid": grid,
            "best": best,
            "best_beats_greedy": bool(best and best["combined"] < greedy_metrics["combined"]),
        }

    json_dump(report, args.output)
    print(f"\nWrote sweep report to {args.output}")
    for language, info in report["languages"].items():
        best = info["best"]
        print(
            f"{language}: best alpha={best['alpha']} beta={best['beta']} "
            f"combined={best['combined']:.4f} (greedy {info['greedy']['combined']:.4f})"
        )


if __name__ == "__main__":
    main()
