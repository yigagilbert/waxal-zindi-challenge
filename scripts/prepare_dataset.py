#!/usr/bin/env python3
"""Prepare local WAXAL datasets from Zindi CSV IDs and Hugging Face audio."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.audio import audio_array_from_example, duration_seconds  # noqa: E402
from waxal.data import (  # noqa: E402
    LANGUAGE_CONFIGS,
    TARGET_LANGUAGES,
    default_raw_dir,
    id_language,
    load_zindi_test,
    load_zindi_train,
    split_rows,
    write_csv_rows,
)
from waxal.utils import json_dump  # noqa: E402


def metadata_rows(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]], output_dir: Path) -> None:
    """Write lightweight train/validation/test CSV metadata."""
    train_meta = [
        {
            "ID": row["id"],
            "Target": row["transcription"],
            "language": row["language"],
            "original_split": row["original_split"],
        }
        for row in train_rows
        if row["original_split"] == "train"
    ]
    val_meta = [
        {
            "ID": row["id"],
            "Target": row["transcription"],
            "language": row["language"],
            "original_split": row["original_split"],
        }
        for row in train_rows
        if row["original_split"] == "validation"
    ]
    test_meta = [
        {
            "ID": row["ID"],
            "language": id_language(row["ID"]),
            "original_split": "test",
        }
        for row in test_rows
    ]
    write_csv_rows(output_dir / "train.csv", train_meta, ["ID", "Target", "language", "original_split"])
    write_csv_rows(output_dir / "validation.csv", val_meta, ["ID", "Target", "language", "original_split"])
    write_csv_rows(output_dir / "test.csv", test_meta, ["ID", "language", "original_split"])


def attach_batch(batch, *, csv_by_id: dict[str, dict[str, str]], split_name: str, include_labels: bool):
    """Attach Zindi metadata to a Hugging Face batch without trusting HF test labels."""
    out = {
        "ID": [],
        "language": [],
        "original_split": [],
        "duration": [],
    }
    if include_labels:
        out["transcription"] = []

    for example_id, audio in zip(batch["id"], batch["audio"], strict=True):
        csv_row = csv_by_id[example_id]
        lang = csv_row.get("language") or id_language(example_id)
        out["ID"].append(example_id)
        out["language"].append(lang)
        out["original_split"].append(csv_row.get("original_split", split_name))
        try:
            array, sample_rate = audio_array_from_example(audio)
            out["duration"].append(duration_seconds(array, sample_rate))
        except Exception:
            out["duration"].append(None)
        if include_labels:
            out["transcription"].append(csv_row["transcription"])
    return out


def collect_streaming_dataset(ds, *, ids: set[str], csv_by_id: dict[str, dict[str, str]], split_name: str, include_labels: bool, limit: int | None):
    """Collect a small streaming subset into a regular Dataset."""
    from datasets import Dataset

    rows = []
    for example in ds:
        example_id = example["id"]
        if example_id not in ids:
            continue
        csv_row = csv_by_id[example_id]
        record = {
            "id": example_id,
            "ID": example_id,
            "audio": example["audio"],
            "language": csv_row.get("language") or id_language(example_id),
            "original_split": csv_row.get("original_split", split_name),
        }
        if include_labels:
            record["transcription"] = csv_row["transcription"]
        try:
            array, sample_rate = audio_array_from_example(example["audio"])
            record["duration"] = duration_seconds(array, sample_rate)
        except Exception:
            record["duration"] = None
        rows.append(record)
        if limit is not None and len(rows) >= limit:
            break
    return Dataset.from_list(rows)


def prepare_hf_dataset(
    *,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    languages: list[str],
    sample_rate: int,
    output_dir: Path,
    streaming: bool,
    max_per_language_split: int | None,
) -> dict:
    """Prepare and save a local Hugging Face DatasetDict."""
    from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

    csv_by_id = {row["id"]: row for row in train_rows}
    csv_by_id.update(
        {
            row["ID"]: {
                "id": row["ID"],
                "language": id_language(row["ID"]),
                "original_split": "test",
            }
            for row in test_rows
        }
    )

    zindi_ids_by_split_lang: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in train_rows:
        if row["language"] in languages:
            zindi_ids_by_split_lang[(row["original_split"], row["language"])].add(row["id"])
    for row in test_rows:
        lang = id_language(row["ID"])
        if lang in languages:
            zindi_ids_by_split_lang[("test", lang)].add(row["ID"])

    prepared = {"train": [], "validation": [], "test": []}
    report = {"splits": {}, "missing_ids": {}}

    for split_name in ["train", "validation", "test"]:
        for lang in languages:
            ids = zindi_ids_by_split_lang[(split_name, lang)]
            if not ids:
                continue
            hf_split = split_name
            print(f"Loading google/WaxalNLP {LANGUAGE_CONFIGS[lang]} split={hf_split} for {len(ids)} Zindi IDs")
            ds = load_dataset(
                "google/WaxalNLP",
                LANGUAGE_CONFIGS[lang],
                split=hf_split,
                streaming=streaming,
            )
            ds = ds.cast_column("audio", Audio(sampling_rate=sample_rate))
            include_labels = split_name != "test"

            if streaming:
                if max_per_language_split is None:
                    raise ValueError("--streaming requires --max-per-language-split for bounded local materialization")
                filtered = collect_streaming_dataset(
                    ds,
                    ids=ids,
                    csv_by_id=csv_by_id,
                    split_name=split_name,
                    include_labels=include_labels,
                    limit=max_per_language_split,
                )
                seen = set(filtered["ID"]) if len(filtered) else set()
            else:
                filtered = ds.filter(lambda row, wanted=ids: row["id"] in wanted)
                if max_per_language_split is not None:
                    filtered = filtered.select(range(min(len(filtered), max_per_language_split)))
                seen = set(filtered["id"]) if len(filtered) else set()
                remove_columns = [c for c in filtered.column_names if c not in {"id", "audio", "language"}]
                filtered = filtered.map(
                    attach_batch,
                    batched=True,
                    fn_kwargs={
                        "csv_by_id": csv_by_id,
                        "split_name": split_name,
                        "include_labels": include_labels,
                    },
                    remove_columns=remove_columns,
                )

            missing = sorted(ids - seen)
            key = f"{split_name}_{lang}"
            report["splits"][key] = {"expected": len(ids), "saved": len(filtered), "missing": len(missing)}
            report["missing_ids"][key] = missing[:50]
            prepared[split_name].append(filtered)

    dataset_dict = DatasetDict()
    for split_name, parts in prepared.items():
        if parts:
            dataset_dict[split_name] = concatenate_datasets(parts)
    out_path = output_dir / "hf_dataset"
    dataset_dict.save_to_disk(out_path)
    report["dataset_path"] = str(out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--languages", nargs="+", default=list(TARGET_LANGUAGES), choices=list(TARGET_LANGUAGES))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--metadata-only", action="store_true", help="Only write train/validation/test CSV metadata.")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode; requires max-per-language-split.")
    parser.add_argument("--max-per-language-split", type=int, default=None, help="Small subset limit for smoke tests.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_zindi_train(args.raw_dir)
    test_rows = load_zindi_test(args.raw_dir)
    train_rows = [row for row in train_rows if row["language"] in set(args.languages)]
    test_rows = [row for row in test_rows if id_language(row["ID"]) in set(args.languages)]

    metadata_rows(train_rows, test_rows, args.output_dir)
    report = {
        "raw_dir": str(args.raw_dir),
        "output_dir": str(args.output_dir),
        "languages": args.languages,
        "metadata_csvs": {
            "train": str(args.output_dir / "train.csv"),
            "validation": str(args.output_dir / "validation.csv"),
            "test": str(args.output_dir / "test.csv"),
        },
    }

    if not args.metadata_only:
        hf_report = prepare_hf_dataset(
            train_rows=train_rows,
            test_rows=test_rows,
            languages=args.languages,
            sample_rate=args.sample_rate,
            output_dir=args.output_dir,
            streaming=args.streaming,
            max_per_language_split=args.max_per_language_split,
        )
        report.update(hf_report)

    json_dump(report, args.output_dir / "prepare_report.json")
    print(f"Saved metadata and report under {args.output_dir}")
    if args.metadata_only:
        print("Metadata-only mode: Hugging Face audio was not downloaded or cached.")


if __name__ == "__main__":
    main()
