#!/usr/bin/env python3
"""Predict language (lin/lug/sna) for audio without metadata.

Strategy: greedy-decode each clip with the multilingual CTC model (or take an
existing transcript CSV), then score the transcript under each per-language
KenLM and pick the best-scoring language. Phase 2 test IDs carry no language
metadata, so this replaces ID-prefix-based routing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from waxal.data import read_prediction_csv, write_csv_rows  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402

DEFAULT_LANGUAGES = ("lin", "lug", "sna")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=None, help="Existing transcript CSV (ID,Target). If omitted, decode audio.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="CTC checkpoint for greedy decoding when --predictions is absent.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--kenlm-dir", type=Path, default=Path("data/lm"))
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--languages", nargs="*", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--default-language", default="lin", help="Assigned when the transcript is empty/undecidable.")
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_language_models(kenlm_dir: Path, order: int, languages: list[str]) -> dict:
    import kenlm

    models = {}
    for language in languages:
        path = kenlm_dir / f"{language}_{order}gram.binary"
        if not path.exists():
            raise FileNotFoundError(f"KenLM binary missing for {language}: {path}")
        models[language] = kenlm.Model(str(path))
    return models


def score_language(text: str, models: dict, *, normalization: str = "language_safe") -> tuple[str, dict[str, float]]:
    """Return (best_language, per-language length-normalized log10 scores)."""
    normalized = normalize_text(text or "", normalization)
    words = normalized.split()
    if not words:
        return "", {}
    scores = {
        language: model.score(" ".join(words), bos=True, eos=True) / len(words)
        for language, model in models.items()
    }
    return max(scores, key=scores.get), scores


def greedy_decode_split(args: argparse.Namespace) -> list[dict[str, str]]:
    import torch
    from transformers import AutoModelForCTC

    from run_xlsr_inference import build_processor, clean_ctc_prediction, load_split, resolve_vocab_path

    if args.checkpoint is None or args.dataset_dir is None:
        raise ValueError("Provide either --predictions or both --checkpoint and --dataset-dir.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = build_processor(resolve_vocab_path(args.checkpoint, None))
    model = AutoModelForCTC.from_pretrained(args.checkpoint).to(device)
    model.eval()
    ds = load_split(args.dataset_dir, args.split)
    rows = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        audios = [audio["array"] for audio in batch["audio"]]
        inputs = processor(audios, sampling_rate=16_000, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        decoded = processor.batch_decode(pred_ids)
        for example_id, text in zip(batch["ID"], decoded, strict=True):
            rows.append({"ID": example_id, "Target": clean_ctc_prediction(text, processor.tokenizer.word_delimiter_token)})
        print(f"Greedy decoded {len(rows)}/{len(ds)}", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    rows = read_prediction_csv(args.predictions) if args.predictions else greedy_decode_split(args)
    models = load_language_models(args.kenlm_dir, args.order, args.languages)

    out_rows = []
    defaulted = 0
    for row in rows:
        best, scores = score_language(row.get("Target", ""), models, normalization=args.normalization)
        if not best:
            best = args.default_language
            defaulted += 1
        out_rows.append(
            {
                "ID": row["ID"],
                "predicted_language": best,
                **{f"score_{language}": f"{scores.get(language, float('nan')):.4f}" for language in args.languages},
            }
        )
    write_csv_rows(args.output, out_rows, list(out_rows[0].keys()))
    print(f"Wrote {len(out_rows)} language predictions to {args.output} ({defaulted} defaulted to {args.default_language!r})")


if __name__ == "__main__":
    main()
