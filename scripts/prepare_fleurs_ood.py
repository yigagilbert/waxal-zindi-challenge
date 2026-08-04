#!/usr/bin/env python3
"""Prepare a small, labeled FLEURS Lingala/Shona out-of-domain ASR gate.

FLEURS is public CC-BY-4.0 data. This script intentionally downloads only the
validation split and stores it separately from WAXAL training data. It is an
evaluation gate, not a source of Phase 2 metadata or pseudo-labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGS = {"lin": "ln_cd", "sna": "sn_zw"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fleurs_ood"))
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--languages", nargs="+", default=["lin", "sna"], choices=sorted(CONFIGS))
    args = parser.parse_args()

    from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

    parts = []
    counts = {}
    for language in args.languages:
        config = CONFIGS[language]
        # Restrict the parquet builder to the requested split. Loading a named
        # FLEURS config causes datasets to download/cache train+validation+test
        # even when split=validation, wasting roughly 8 GB for these two langs.
        data_files = f"parquet-data/{config}/{args.split}-*.parquet"
        ds = load_dataset("google/fleurs", data_files=data_files, split="train")
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

        def attach(batch, *, lang=language, split=args.split):
            return {
                "ID": [f"fleurs_{lang}_{split}_{example_id}" for example_id in batch["id"]],
                "language": [lang] * len(batch["id"]),
                "original_split": [f"fleurs_{split}"] * len(batch["id"]),
                "transcription": batch["transcription"],
            }

        remove = [column for column in ds.column_names if column != "audio"]
        ds = ds.map(attach, batched=True, remove_columns=remove)
        counts[language] = len(ds)
        parts.append(ds)

    prepared = DatasetDict({args.split: concatenate_datasets(parts)})
    dataset_path = args.output_dir / "hf_dataset"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save_to_disk(dataset_path)
    report = {
        "source": "google/fleurs",
        "license": "CC-BY-4.0",
        "usage": "evaluation-only OOD gate",
        "split": args.split,
        "configs": {language: CONFIGS[language] for language in args.languages},
        "counts": counts,
        "dataset_path": str(dataset_path),
    }
    (args.output_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
