#!/usr/bin/env python3
"""Audit WAXAL train/validation speaker overlap without decoding audio.

Reads only ``id``, ``speaker_id`` and ``gender`` from cached official parquet
shards, intersects them with the challenge CSV manifests, and writes an
ID-only-compatible validation manifest containing speakers absent from train.
The output may be passed to ``run_no_metadata_pipeline.py --ids-csv``; only the
ID column is consumed by that pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ID", "Target", "language", "speaker_id", "gender", "speaker_seen_in_train"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parquet_paths(snapshot_root: Path, language: str, split: str) -> list[Path]:
    pattern = f"*/data/ASR/{language}/{language}-{split}-*.parquet"
    # A blob can be linked by multiple cached snapshots. Read each resolved
    # parquet once so counts remain stable across cache revisions.
    unique = {path.resolve(): path for path in snapshot_root.glob(pattern)}
    paths = sorted(unique.values(), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No cached parquet shards matched {snapshot_root / pattern}")
    return paths


def load_metadata(paths: list[Path], wanted_ids: set[str]) -> dict[str, tuple[str, str]]:
    import pyarrow.parquet as pq

    found: dict[str, tuple[str, str]] = {}
    for path in paths:
        table = pq.read_table(path, columns=["id", "speaker_id", "gender"])
        columns = table.to_pydict()
        for example_id, speaker_id, gender in zip(
            columns["id"], columns["speaker_id"], columns["gender"], strict=True
        ):
            if example_id not in wanted_ids:
                continue
            found[example_id] = (
                "" if speaker_id is None else str(speaker_id),
                "" if gender is None else str(gender),
            )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub/datasets--google--WaxalNLP/snapshots",
    )
    parser.add_argument("--languages", nargs="+", default=["lin", "sna"])
    parser.add_argument("--bench-csv", type=Path, default=Path("outputs/day4_h100/bench_ids.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/day4_h100/speaker_ood_bench.csv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/day4_h100/speaker_overlap_report.json"))
    args = parser.parse_args()

    languages = set(args.languages)
    train_rows = [row for row in read_csv(args.processed_dir / "train.csv") if row["language"] in languages]
    val_rows = [row for row in read_csv(args.processed_dir / "validation.csv") if row["language"] in languages]
    train_by_id = {row["ID"]: row for row in train_rows}
    val_by_id = {row["ID"]: row for row in val_rows}

    report: dict[str, object] = {"languages": {}, "bench_csv": str(args.bench_csv)}
    output_rows: list[dict[str, str]] = []
    metadata_by_id: dict[str, tuple[str, str]] = {}

    for language in args.languages:
        lang_train_ids = {row["ID"] for row in train_rows if row["language"] == language}
        lang_val_ids = {row["ID"] for row in val_rows if row["language"] == language}
        train_meta = load_metadata(
            parquet_paths(args.snapshot_root, language, "train"), lang_train_ids
        )
        val_meta = load_metadata(
            parquet_paths(args.snapshot_root, language, "validation"), lang_val_ids
        )
        metadata_by_id.update(val_meta)

        train_speakers = {speaker for speaker, _ in train_meta.values() if speaker}
        val_speakers = {speaker for speaker, _ in val_meta.values() if speaker}
        overlapping_speakers = train_speakers & val_speakers
        unique_val_speakers = val_speakers - train_speakers
        seen_rows = 0
        unseen_rows = 0
        missing_speaker_rows = 0

        for example_id in sorted(lang_val_ids):
            speaker_id, gender = val_meta.get(example_id, ("", ""))
            if not speaker_id:
                missing_speaker_rows += 1
                continue
            seen = speaker_id in train_speakers
            seen_rows += int(seen)
            unseen_rows += int(not seen)
            if not seen:
                row = val_by_id[example_id]
                output_rows.append(
                    {
                        "ID": example_id,
                        "Target": row["Target"],
                        "language": language,
                        "speaker_id": speaker_id,
                        "gender": gender,
                        "speaker_seen_in_train": "false",
                    }
                )

        report["languages"][language] = {
            "challenge_train_rows": len(lang_train_ids),
            "challenge_validation_rows": len(lang_val_ids),
            "train_rows_with_metadata": len(train_meta),
            "validation_rows_with_metadata": len(val_meta),
            "train_speakers": len(train_speakers),
            "validation_speakers": len(val_speakers),
            "overlapping_speakers": len(overlapping_speakers),
            "validation_unique_speakers": len(unique_val_speakers),
            "validation_rows_seen_speaker": seen_rows,
            "validation_rows_unseen_speaker": unseen_rows,
            "validation_rows_missing_speaker": missing_speaker_rows,
        }

    bench_status = Counter()
    if args.bench_csv.exists():
        train_speakers_by_language: dict[str, set[str]] = {}
        for language in args.languages:
            lang_train_ids = {row["ID"] for row in train_rows if row["language"] == language}
            train_meta = load_metadata(
                parquet_paths(args.snapshot_root, language, "train"), lang_train_ids
            )
            train_speakers_by_language[language] = {
                speaker for speaker, _ in train_meta.values() if speaker
            }
        for row in read_csv(args.bench_csv):
            language = row["language"]
            speaker_id = metadata_by_id.get(row["ID"], ("", ""))[0]
            if not speaker_id:
                status = "missing_speaker"
            elif speaker_id in train_speakers_by_language.get(language, set()):
                status = "seen_speaker"
            else:
                status = "unseen_speaker"
            bench_status[f"{language}:{status}"] += 1

    output_rows.sort(key=lambda row: (row["language"], row["ID"]))
    write_csv(args.output, output_rows)
    report["speaker_ood_rows"] = len(output_rows)
    report["speaker_ood_by_language"] = dict(Counter(row["language"] for row in output_rows))
    report["current_bench_speaker_status"] = dict(sorted(bench_status.items()))
    report["output"] = str(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
