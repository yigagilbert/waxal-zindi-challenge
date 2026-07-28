#!/usr/bin/env python3
"""Prepare WaxalNLP train/validation splits for the Phase-2 languages.

Unlike prepare_dataset.py (which is Zindi-ID-driven for lin/lug/sna), this loads
the WaxalNLP parquet shards for the given languages directly — no Zindi CSV needed.
Output schema matches the trio pipeline (ID, audio, transcription, language) so
train_whisper.py can concatenate both via data.extra_dataset_dirs.

Usage:
  python scripts/prepare_phase2_train.py --languages ach nyn xog myx \
    --max-per-language 8000 --output-dir data/phase2_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEXT_COLUMN_CANDIDATES = ("transcription", "text", "sentence", "transcript", "target")
ID_COLUMN_CANDIDATES = ("id", "ID", "audio_id", "utterance_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["ach", "nyn", "xog", "myx"])
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--max-per-language", type=int, default=None, help="Cap per language per split (shuffled, seeded).")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase2_train"))
    args = parser.parse_args()

    from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

    report: dict = {"languages": args.languages, "splits": {}, "max_per_language": args.max_per_language}
    dataset_dict = {}
    for split in args.splits:
        parts = []
        for lang in args.languages:
            # Restrict data_files to this split's shards — selecting by HF config would
            # resolve+download every split including the huge `unlabeled` one.
            data_files = f"data/ASR/{lang}/{lang}-{split}-*.parquet"
            print(f"Loading google/WaxalNLP {data_files}")
            ds = load_dataset("google/WaxalNLP", data_files=data_files, split="train")

            text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in ds.column_names), None)
            id_col = next((c for c in ID_COLUMN_CANDIDATES if c in ds.column_names), None)
            if text_col is None or id_col is None:
                raise SystemExit(f"{lang}/{split}: can't find text/id columns in {ds.column_names}")

            if args.max_per_language is not None and len(ds) > args.max_per_language:
                ds = ds.shuffle(seed=args.seed).select(range(args.max_per_language))

            rename = {}
            if id_col != "ID":
                rename[id_col] = "ID"
            if text_col != "transcription":
                rename[text_col] = "transcription"
            if rename:
                ds = ds.rename_columns(rename)
            if "language" not in ds.column_names:
                ds = ds.add_column("language", [lang] * len(ds))
            else:
                ds = ds.remove_columns(["language"]).add_column("language", [lang] * len(ds))
            ds = ds.select_columns(["ID", "audio", "transcription", "language"])
            ds = ds.cast_column("audio", Audio(sampling_rate=args.sample_rate))

            parts.append(ds)
            report["splits"].setdefault(split, {})[lang] = len(ds)
            print(f"  {lang}/{split}: {len(ds)} rows (text column was '{text_col}')")
        dataset_dict[split] = concatenate_datasets(parts).shuffle(seed=args.seed)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    DatasetDict(dataset_dict).save_to_disk(out / "hf_dataset")

    if "validation" in dataset_dict:
        import csv

        val = dataset_dict["validation"]
        # ID,Target header: what references_from_validation_csv/evaluate_predictions expect.
        with (out / "validation.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "language", "Target"])
            writer.writeheader()
            for example_id, lang, text in zip(val["ID"], val["language"], val["transcription"], strict=True):
                writer.writerow({"ID": example_id, "language": lang, "Target": text})
        print(f"Wrote {out / 'validation.csv'} ({len(val)} rows) for external eval gates")

    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved DatasetDict to {out / 'hf_dataset'}")


if __name__ == "__main__":
    main()
