#!/usr/bin/env python3
"""Phase 2 dress rehearsal: full ASR pipeline with zero metadata.

Never reads the language column or ID prefixes for routing. Flow:
  1. one acoustic pass -> cached trimmed logits per clip
  2. greedy transcript from the cached logits
  3. language ID by scoring the greedy transcript under per-language KenLMs
  4. per-language beam+LM re-decode of the cached logits (no second GPU pass)
  5. predictions CSV + report (LID accuracy/confusion + WER/CER when the split
     has references — validation — purely for evaluation, never for routing)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from waxal.scoring import compute_group_metrics  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402

from predict_language_from_audio import load_language_models, score_language  # noqa: E402
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
    parser.add_argument(
        "--ids-csv",
        type=Path,
        default=None,
        help="Optional ID-only manifest used to restrict the split for an evaluation gate. "
        "No manifest columns are exposed to routing.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the selected dataset into contiguous decode shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index used with --num-shards.",
    )
    parser.add_argument("--kenlm-dir", type=Path, default=Path("data/lm"))
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="*",
        default=None,
        help="Extra same-vocabulary CTC checkpoints whose log-probabilities are "
        "averaged with --checkpoint before decoding.",
    )
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--languages", nargs="*", default=["lin", "lug", "sna"])
    parser.add_argument("--default-language", default="lin")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help='Optional per-language decode params, e.g. {"lin": {"alpha": 0.6, "beta": 1.0}}',
    )
    parser.add_argument(
        "--greedy-languages",
        nargs="*",
        default=[],
        help="Languages to decode greedily instead of beam+LM; default is none.",
    )
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/analysis/no_metadata_validation_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import logging

    import numpy as np
    import torch
    from pyctcdecode import build_ctcdecoder
    from transformers import AutoModelForCTC

    logging.getLogger("pyctcdecode").setLevel(logging.ERROR)

    from waxal.data import write_csv_rows

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = build_processor(resolve_vocab_path(args.checkpoint, args.vocab_path))
    labels = build_ctc_labels(processor.tokenizer)
    word_delimiter = processor.tokenizer.word_delimiter_token
    model = AutoModelForCTC.from_pretrained(args.checkpoint).to(device)
    model.eval()
    ensemble_models = []
    for extra_path in args.ensemble_checkpoints or []:
        extra_model = AutoModelForCTC.from_pretrained(extra_path).to(device)
        extra_model.eval()
        ensemble_models.append(extra_model)
        print(f"Ensembling with {extra_path}")

    ds = load_split(args.dataset_dir, args.split)
    selected_ids = None
    if args.ids_csv is not None:
        with args.ids_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "ID" not in rows[0]:
            raise ValueError(f"{args.ids_csv} must contain a non-empty ID column")
        manifest_ids = [row["ID"] for row in rows]
        if len(manifest_ids) != len(set(manifest_ids)):
            raise ValueError(f"{args.ids_csv} contains duplicate IDs")
        selected_ids = set(manifest_ids)
        ds = ds.filter(lambda row: row["ID"] in selected_ids)
        found_ids = set(ds["ID"])
        missing = selected_ids - found_ids
        if missing:
            preview = ", ".join(sorted(missing)[:10])
            raise ValueError(f"{len(missing)} IDs from {args.ids_csv} are absent from the split: {preview}")
        print(f"Restricted {args.split} to {len(ds)} IDs from {args.ids_csv}")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if args.num_shards > 1:
        before = len(ds)
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index, contiguous=True)
        print(f"Decode shard {args.shard_index + 1}/{args.num_shards}: {len(ds)}/{before} examples")
    if args.max_samples is not None:
        ds = ds.select(range(min(len(ds), args.max_samples)))
    has_references = "transcription" in ds.column_names
    has_true_language = "language" in ds.column_names

    # ---- pass 1: acoustic model once, cache trimmed logits ----
    logits_list: list = []
    ids: list[str] = []
    true_languages: list[str] = []
    references: list[str] = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        audios = [audio["array"] for audio in batch["audio"]]
        inputs = processor(audios, sampling_rate=16_000, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = torch.log_softmax(model(**inputs).logits, dim=-1)
            for extra_model in ensemble_models:
                logits += torch.log_softmax(extra_model(**inputs).logits, dim=-1)
            if ensemble_models:
                logits /= 1 + len(ensemble_models)
        output_lengths = model._get_feat_extract_output_lengths(inputs["attention_mask"].sum(-1))
        for row_idx in range(logits.shape[0]):
            length = int(output_lengths[row_idx])
            logits_list.append(logits[row_idx, :length].detach().to(torch.float16).cpu().numpy())
        ids.extend(batch["ID"])
        if has_true_language:
            true_languages.extend(batch["language"])
        if has_references:
            references.extend(batch["transcription"])
        print(f"Acoustic pass {len(ids)}/{len(ds)}", flush=True)

    # ---- greedy transcripts from cached logits ----
    greedy = []
    for item in logits_list:
        token_ids = item.astype(np.float32).argmax(-1)
        greedy.append(clean_ctc_prediction(processor.tokenizer.decode(token_ids), word_delimiter))

    # ---- language ID from greedy transcripts ----
    lms = load_language_models(args.kenlm_dir, args.order, args.languages)
    predicted_language = []
    lid_defaulted = 0
    for text in greedy:
        best, _ = score_language(text, lms, normalization=args.normalization)
        if not best:
            best = args.default_language
            lid_defaulted += 1
        predicted_language.append(best)

    # The remaining work is CPU-only KenLM decoding. Release the acoustic
    # models before a potentially long, wide-beam search so independent shards
    # can run in parallel without pinning one model copy per shard on the GPU.
    del model
    ensemble_models.clear()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        print("Released acoustic model GPU memory before CPU beam decoding", flush=True)

    # ---- per-language beam+LM decode of the cached logits ----
    params = {}
    if args.params_json and args.params_json.exists():
        params = json.loads(args.params_json.read_text())
    decoders = {}
    for language in args.languages:
        lang_params = params.get(language, {})
        corpus_path = args.kenlm_dir / f"{language}.txt"
        unigrams = (
            sorted(set(corpus_path.read_text(encoding="utf-8").split())) if corpus_path.exists() else None
        )
        decoders[language] = build_ctcdecoder(
            labels,
            kenlm_model_path=str(args.kenlm_dir / f"{language}_{args.order}gram.binary"),
            unigrams=unigrams,
            alpha=float(lang_params.get("alpha", args.alpha)),
            beta=float(lang_params.get("beta", args.beta)),
        )
    greedy_languages = set(args.greedy_languages or [])
    routed = []
    for index, item in enumerate(logits_list):
        language = predicted_language[index]
        if language in greedy_languages:
            token_ids = item.astype(np.float32).argmax(-1)
            text = processor.tokenizer.decode(token_ids)
        else:
            text = decoders[language].decode(item.astype(np.float32), beam_width=args.beam_width)
            for special in (processor.tokenizer.unk_token, processor.tokenizer.bos_token, processor.tokenizer.eos_token, "⁇"):
                if special:
                    text = text.replace(special, " ")
        routed.append(clean_ctc_prediction(text, word_delimiter))
        if (index + 1) % 250 == 0:
            print(f"Routed decode {index + 1}/{len(logits_list)}", flush=True)

    rows = [{"ID": example_id, "Target": text.strip() or "."} for example_id, text in zip(ids, routed, strict=True)]
    write_csv_rows(args.output_predictions, rows, ["ID", "Target"])

    # ---- report ----
    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "ids_csv": str(args.ids_csv) if args.ids_csv else None,
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "num_examples": len(ids),
        "normalization": args.normalization,
        "decoder": {
            "kenlm_dir": str(args.kenlm_dir),
            "order": args.order,
            "languages": list(args.languages),
            "default_language": args.default_language,
            "greedy_languages": sorted(greedy_languages),
            "beam_width": args.beam_width,
        },
        "decode_params": {language: params.get(language, {"alpha": args.alpha, "beta": args.beta}) for language in args.languages},
        "lid": {
            "defaulted_empty_transcript": lid_defaulted,
            "assigned_counts": dict(sorted(Counter(predicted_language).items())),
        },
        "output_predictions": str(args.output_predictions),
    }

    if has_true_language:
        correct = sum(1 for t, p in zip(true_languages, predicted_language, strict=True) if t == p)
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        mistakes = []
        for example_id, t, p, text in zip(ids, true_languages, predicted_language, greedy, strict=True):
            confusion[t][p] += 1
            if t != p and len(mistakes) < 40:
                mistakes.append({"ID": example_id, "true": t, "predicted": p, "greedy_transcript": text[:160]})
        report["lid"].update(
            {
                "accuracy": correct / len(ids) if ids else 0.0,
                "confusion_true_vs_predicted": {t: dict(sorted(row.items())) for t, row in sorted(confusion.items())},
                "misclassified_examples": mistakes,
            }
        )

    if has_references:
        refs_norm = [normalize_text(text, args.normalization) for text in references]

        def grouped_metrics(preds: list[str]) -> dict:
            preds_norm = [normalize_text(text, args.normalization) for text in preds]
            overall = compute_group_metrics(refs_norm, preds_norm, normalization=args.normalization)
            by_language = {}
            if has_true_language:
                index_by_lang: dict[str, list[int]] = defaultdict(list)
                for index, language in enumerate(true_languages):
                    index_by_lang[language].append(index)
                for language, indices in sorted(index_by_lang.items()):
                    by_language[language] = compute_group_metrics(
                        [refs_norm[i] for i in indices],
                        [preds_norm[i] for i in indices],
                        normalization=args.normalization,
                    )
            return {"overall": overall, "by_language": by_language}

        report["metrics"] = {
            "greedy_no_metadata": grouped_metrics(greedy),
            "routed_no_metadata": grouped_metrics(routed),
        }

    json_dump(report, args.report)
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2, ensure_ascii=False))
    if "metrics" in report:
        for name, metrics in report["metrics"].items():
            overall = metrics["overall"]
            print(f"{name}: wer={overall['wer']:.4f} cer={overall['cer']:.4f} combined={overall['combined']:.4f}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
