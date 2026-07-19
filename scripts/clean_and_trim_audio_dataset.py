#!/usr/bin/env python3
"""Build the cleaned-audio WAXAL dataset: trim edge silence, bucket by quality,
save cleaned FLAC files, and assemble the final Hugging Face DatasetDict.

Read-only with respect to raw data: originals are never modified. Every output
row keeps a pointer back to its source dataset/split/ID plus content hashes.

Pipeline (one command):
  1. decode each example, compute original stats + hash
  2. conservatively trim leading/trailing silence (waxal.audio.trim_edges)
  3. write cleaned mono 16 kHz FLAC to data/audio_cleaned/<split>/<lang>/
  4. assign clean/medium/noisy/excluded buckets from post-trim stats
  5. save DatasetDict (train=clean, medium_train, noisy_for_review,
     excluded_for_review, validation) + metadata CSVs + impact report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.audio import extended_audio_stats, resample_if_needed, to_mono, trim_edges  # noqa: E402
from waxal.data import write_csv_rows  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import ensure_dir, json_dump  # noqa: E402

PREPROCESSING_VERSION = "clean_audio_v1"

METADATA_FIELDS = [
    "id",
    "source_id",
    "source_dataset",
    "source_split",
    "language",
    "split",
    "original_split",
    "transcription",
    "cleaned_audio_path",
    "source_pointer",
    "original_duration",
    "cleaned_duration",
    "leading_silence_removed",
    "trailing_silence_removed",
    "total_silence_removed",
    "trim_threshold_used",
    "trim_padding_ms",
    "cleaning_status",
    "quality_bucket",
    "quality_flags",
    "chars_per_second_original",
    "chars_per_second_cleaned",
    "words_per_second_original",
    "words_per_second_cleaned",
    "rms",
    "silence_ratio",
    "clipping_ratio",
    "preprocessing_version",
    "audio_hash_original",
    "audio_hash_cleaned",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed_generalization_mix"))
    parser.add_argument("--splits", nargs="*", default=["train", "validation"])
    parser.add_argument("--output-audio-dir", type=Path, default=Path("data/audio_cleaned"))
    parser.add_argument("--output-dataset", type=Path, default=Path("data/final_combined_clean_audio_dataset"))
    parser.add_argument("--quality-dir", type=Path, default=Path("data/quality"))
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/data_quality"))
    # trimming (conservative by design)
    parser.add_argument("--top-db", type=float, default=40.0)
    parser.add_argument("--pad-ms", type=float, default=200.0)
    parser.add_argument("--min-cleaned-duration", type=float, default=0.3)
    # bucket thresholds
    parser.add_argument("--silent-rms", type=float, default=0.005)
    parser.add_argument("--silent-ratio", type=float, default=0.95)
    parser.add_argument("--mismatch-duration", type=float, default=1.5)
    parser.add_argument("--mismatch-words", type=int, default=8)
    parser.add_argument("--short2-duration", type=float, default=2.0)
    parser.add_argument("--short2-chars", type=int, default=20)
    parser.add_argument("--exclude-cps", type=float, default=35.0)
    parser.add_argument("--noisy-cps", type=float, default=25.0)
    parser.add_argument("--medium-cps", type=float, default=22.0)
    parser.add_argument("--noisy-clipping", type=float, default=0.05)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument(
        "--extra-train-dataset",
        type=Path,
        action="append",
        default=None,
        help=(
            "Standardized external Dataset dir (holds hf_dataset) to concatenate into the "
            "TRAIN split before cleaning. Repeat. Lets externals be added without materializing "
            "a separate merged copy on disk (they are memory-mapped, then cleaned in one pass). "
            "Validation/test are left untouched (in-domain only)."
        ),
    )
    parser.add_argument(
        "--preprocessing-version",
        default=PREPROCESSING_VERSION,
        help="Version tag stamped on every row (bump when the source mix or rules change, e.g. clean_audio_v2 once Lingala_100hrs is added).",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assign_bucket(record: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    """Return (bucket, flags). Rules documented in docs/DATA_QUALITY_AUDIT.md."""
    flags: list[str] = []
    chars = len(record["transcription"])
    words = len(record["transcription"].split())
    dur = record["cleaned_duration"]
    cps = record["chars_per_second_cleaned"]
    status = record["cleaning_status"]

    if chars == 0:
        flags.append("empty_transcript")
    if status == "failed_decode":
        flags.append("failed_decode")
    if status == "likely_empty_audio":
        flags.append("likely_empty_audio")
    if status == "too_short_after_trim":
        flags.append("too_short_after_trim")
    if record["rms"] < args.silent_rms or record["silence_ratio"] > args.silent_ratio:
        flags.append("likely_silent")
    if dur < args.mismatch_duration and words > args.mismatch_words:
        flags.append("short_audio_long_transcript")
    if dur < args.short2_duration and chars > args.short2_chars:
        flags.append("short_audio_many_chars")
    if cps > args.exclude_cps:
        flags.append("chars_per_second_unrealistic")
    if ("likely_silent" in flags or "likely_empty_audio" in flags) and chars > 10:
        flags.append("audio_transcript_mismatch")

    excluded_flags = {
        "empty_transcript",
        "failed_decode",
        "likely_empty_audio",
        "too_short_after_trim",
        "short_audio_long_transcript",
        "chars_per_second_unrealistic",
        "audio_transcript_mismatch",
    }
    if excluded_flags & set(flags):
        return "excluded", flags

    if record["clipping_ratio"] > args.noisy_clipping:
        flags.append("heavy_clipping")
    if args.noisy_cps < cps <= args.exclude_cps:
        flags.append("chars_per_second_high")
    if "likely_silent" in flags or "heavy_clipping" in flags or "chars_per_second_high" in flags or "short_audio_many_chars" in flags:
        return "noisy", flags

    if args.medium_cps < cps <= args.noisy_cps:
        flags.append("chars_per_second_elevated")
    if record["total_silence_removed"] > 0.5 * max(record["original_duration"], 1e-6):
        flags.append("mostly_edge_silence")
    if flags:
        return "medium", flags
    return "clean", flags


def process_split(ds, *, split: str, args: argparse.Namespace) -> list[dict]:
    audio_root = ensure_dir(args.output_audio_dir / split)

    def process(batch):
        import numpy as np
        import soundfile as sf

        out = defaultdict(list)
        size = len(batch["ID"])
        for i in range(size):
            example_id = batch["ID"][i]
            language = batch["language"][i]
            transcription = normalize_text(str(batch.get("transcription", [""] * size)[i] or ""), "language_safe")
            record = {
                "id": example_id,
                "source_id": example_id,
                "source_dataset": (batch.get("source_dataset") or [""] * size)[i] or "waxal",
                "source_split": (batch.get("source_split") or [""] * size)[i] or split,
                "language": language,
                "split": split,
                "original_split": (batch.get("original_split") or [""] * size)[i] or split,
                "transcription": transcription,
                "trim_threshold_used": args.top_db,
                "trim_padding_ms": args.pad_ms,
                "preprocessing_version": args.preprocessing_version,
            }
            try:
                audio = batch["audio"][i]
                arr = to_mono(audio["array"])
                arr, sr = resample_if_needed(arr, int(audio.get("sampling_rate", 16_000)))
                orig_stats = extended_audio_stats(arr, sr)
                record["audio_hash_original"] = sha256_bytes(np.asarray(arr, dtype=np.float32).tobytes())
                trimmed, trim_meta = trim_edges(arr, sr, top_db=args.top_db, pad_ms=args.pad_ms)
                status = str(trim_meta["status"])
                cleaned = trimmed if status == "cleaned_ok" else arr
                cleaned_duration = len(cleaned) / sr
                if status != "likely_empty_audio" and cleaned_duration < args.min_cleaned_duration:
                    status = "too_short_after_trim"
                    cleaned = arr
                    cleaned_duration = len(arr) / sr
                out_path = audio_root / language / f"{example_id}.flac"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out_path, np.asarray(cleaned, dtype=np.float32), sr, format="FLAC")
                cleaned_stats = extended_audio_stats(cleaned, sr)
                chars, words = len(transcription), len(transcription.split())
                record.update(
                    {
                        "cleaned_audio_path": str(out_path),
                        "source_pointer": f"{record['source_dataset']}:{record['source_split']}:{example_id}",
                        "original_duration": round(orig_stats["duration"], 3),
                        "cleaned_duration": round(cleaned_duration, 3),
                        "leading_silence_removed": round(float(trim_meta["leading_silence_removed"]), 3) if status == "cleaned_ok" else 0.0,
                        "trailing_silence_removed": round(float(trim_meta["trailing_silence_removed"]), 3) if status == "cleaned_ok" else 0.0,
                        "cleaning_status": status,
                        "chars_per_second_original": round(chars / orig_stats["duration"], 3) if orig_stats["duration"] > 0 else 0.0,
                        "chars_per_second_cleaned": round(chars / cleaned_duration, 3) if cleaned_duration > 0 else 0.0,
                        "words_per_second_original": round(words / orig_stats["duration"], 3) if orig_stats["duration"] > 0 else 0.0,
                        "words_per_second_cleaned": round(words / cleaned_duration, 3) if cleaned_duration > 0 else 0.0,
                        "rms": round(cleaned_stats["rms"], 6),
                        "silence_ratio": round(cleaned_stats["silence_ratio"], 4),
                        "clipping_ratio": round(cleaned_stats["clipping_ratio"], 5),
                        "audio_hash_cleaned": sha256_bytes(out_path.read_bytes()),
                    }
                )
            except Exception as exc:  # failed decode must be visible, not fatal
                record.update(
                    {
                        "cleaned_audio_path": "",
                        "source_pointer": f"{record['source_dataset']}:{record['source_split']}:{example_id}",
                        "original_duration": 0.0,
                        "cleaned_duration": 0.0,
                        "leading_silence_removed": 0.0,
                        "trailing_silence_removed": 0.0,
                        "cleaning_status": "failed_decode",
                        "chars_per_second_original": 0.0,
                        "chars_per_second_cleaned": 0.0,
                        "words_per_second_original": 0.0,
                        "words_per_second_cleaned": 0.0,
                        "rms": 0.0,
                        "silence_ratio": 1.0,
                        "clipping_ratio": 0.0,
                        "audio_hash_original": "",
                        "audio_hash_cleaned": "",
                    }
                )
                print(f"WARNING: failed to process {example_id}: {type(exc).__name__}: {exc}")
            record["total_silence_removed"] = round(
                record["leading_silence_removed"] + record["trailing_silence_removed"], 3
            )
            bucket, flags = assign_bucket(record, args)
            record["quality_bucket"] = bucket
            record["quality_flags"] = "|".join(flags)
            for key in METADATA_FIELDS:
                out[key].append(record[key])
        return dict(out)

    mapped = ds.map(
        process,
        batched=True,
        batch_size=16,
        num_proc=args.num_proc if len(ds) > 256 else None,
        remove_columns=ds.column_names,
        desc=f"Cleaning {split}",
    )
    return [dict(zip(METADATA_FIELDS, values, strict=True)) for values in zip(*[mapped[c] for c in METADATA_FIELDS], strict=True)]


def rows_to_dataset(rows: list[dict]):
    from datasets import Audio, Dataset

    usable = [r for r in rows if r["cleaned_audio_path"]]
    data = {key: [r[key] for r in usable] for key in METADATA_FIELDS}
    data["audio"] = [r["cleaned_audio_path"] for r in usable]
    data["duration"] = [r["cleaned_duration"] for r in usable]
    ds = Dataset.from_dict(data)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    return ds.rename_column("id", "ID")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    from datasets import DatasetDict, load_from_disk

    dataset_dict = load_from_disk(args.dataset_dir / "hf_dataset")

    # Fold external standardized datasets into TRAIN (memory-mapped, no intermediate
    # merged copy on disk). Externals must already carry the same 11-column schema
    # (ID, audio, transcription, language, original_split, duration, source_*).
    if args.extra_train_dataset:
        from datasets import concatenate_datasets

        base_train = dataset_dict["train"]
        base_cols = base_train.column_names
        parts = [base_train]
        for extra_path in args.extra_train_dataset:
            loaded = load_from_disk(str(extra_path / "hf_dataset")) if (extra_path / "hf_dataset").exists() else load_from_disk(str(extra_path))
            if hasattr(loaded, "keys"):  # a DatasetDict -> flatten its splits
                loaded = concatenate_datasets([loaded[k] for k in loaded.keys()])
            missing = [c for c in base_cols if c not in loaded.column_names]
            if missing:
                raise SystemExit(f"{extra_path} missing columns {missing}; got {loaded.column_names}")
            loaded = loaded.select_columns(base_cols).cast(base_train.features)
            parts.append(loaded)
            print(f"  + extra train: {extra_path} (+{len(loaded)} rows)")
        dataset_dict["train"] = concatenate_datasets(parts)
        print(f"  train after externals: {len(dataset_dict['train'])} rows")

    rows_by_split: dict[str, list[dict]] = {}
    for split in args.splits:
        if split not in dataset_dict:
            print(f"Split {split} not in dataset; skipping")
            continue
        ds = dataset_dict[split]
        if args.max_samples_per_split and len(ds) > args.max_samples_per_split:
            ds = ds.select(range(args.max_samples_per_split))
        rows_by_split[split] = process_split(ds, split=split, args=args)

    ensure_dir(args.quality_dir)
    ensure_dir(args.reports_dir)
    train_rows = rows_by_split.get("train", [])
    val_rows = rows_by_split.get("validation", [])

    # ---- bucket manifests ----
    buckets = {b: [r for r in train_rows if r["quality_bucket"] == b] for b in ("clean", "medium", "noisy", "excluded")}
    manifest_fields = ["id", "source_id", "source_dataset", "language", "transcription", "quality_bucket", "quality_flags", "cleaned_audio_path", "cleaned_duration"]

    def manifest(rows):
        return [{**{k: r[k] for k in manifest_fields}, "ID": r["id"], "Target": r["transcription"]} for r in rows]

    for bucket, rows in buckets.items():
        write_csv_rows(args.quality_dir / f"final_{bucket}_train.csv", manifest(rows), ["ID", "Target", "language", *manifest_fields])
    for language in sorted({r["language"] for r in train_rows}):
        rows = [r for r in buckets["clean"] if r["language"] == language]
        write_csv_rows(args.quality_dir / f"{language}_clean_train.csv", manifest(rows), ["ID", "Target", "language", *manifest_fields])
    val_clean = [r for r in val_rows if r["quality_bucket"] == "clean"]
    val_excluded = [r for r in val_rows if r["quality_bucket"] == "excluded"]
    write_csv_rows(args.quality_dir / "final_clean_validation.csv", manifest(val_clean), ["ID", "Target", "language", *manifest_fields])
    write_csv_rows(args.quality_dir / "final_excluded_validation.csv", manifest(val_excluded), ["ID", "Target", "language", *manifest_fields])

    # ---- final DatasetDict (validation kept FULL for comparability) ----
    out = DatasetDict(
        {
            "train": rows_to_dataset(buckets["clean"]),
            "medium_train": rows_to_dataset(buckets["medium"]),
            "noisy_for_review": rows_to_dataset(buckets["noisy"]),
            "excluded_for_review": rows_to_dataset(buckets["excluded"]),
            "validation": rows_to_dataset(val_rows),
        }
    )
    ensure_dir(args.output_dataset)
    out.save_to_disk(args.output_dataset / "hf_dataset")
    write_csv_rows(args.output_dataset / "train_metadata.csv", buckets["clean"], METADATA_FIELDS)
    write_csv_rows(args.output_dataset / "medium_train_metadata.csv", buckets["medium"], METADATA_FIELDS)
    write_csv_rows(args.output_dataset / "excluded_metadata.csv", buckets["excluded"] + val_excluded, METADATA_FIELDS)
    write_csv_rows(args.output_dataset / "validation_metadata.csv", val_rows, METADATA_FIELDS)
    # training pipeline compatibility: validation.csv with ID/Target/language
    write_csv_rows(
        args.output_dataset / "validation.csv",
        [{"ID": r["id"], "Target": r["transcription"], "language": r["language"]} for r in val_rows],
        ["ID", "Target", "language"],
    )

    # ---- cleaning + impact reports ----
    def hours(rows, key):
        return round(sum(float(r[key]) for r in rows) / 3600.0, 2)

    all_rows = train_rows + val_rows
    write_csv_rows(args.reports_dir / "audio_cleaning_report.csv", all_rows, METADATA_FIELDS)
    impact = {
        "preprocessing_version": args.preprocessing_version,
        "git_commit": git_commit(),
        "source_dataset_dir": str(args.dataset_dir),
        "thresholds": {
            k: getattr(args, k)
            for k in (
                "top_db", "pad_ms", "min_cleaned_duration", "silent_rms", "silent_ratio",
                "mismatch_duration", "mismatch_words", "short2_duration", "short2_chars",
                "exclude_cps", "noisy_cps", "medium_cps", "noisy_clipping",
            )
        },
        "train": {
            "rows_before": len(train_rows),
            "bucket_counts": {b: len(r) for b, r in buckets.items()},
            "bucket_counts_by_language": {
                b: dict(sorted(Counter(x["language"] for x in r).items())) for b, r in buckets.items()
            },
            "bucket_counts_by_source": {
                b: dict(sorted(Counter(x["source_dataset"] for x in r).items())) for b, r in buckets.items()
            },
            "excluded_reasons": dict(sorted(Counter(f for r in buckets["excluded"] for f in r["quality_flags"].split("|") if f).items())),
            "original_hours": hours(train_rows, "original_duration"),
            "cleaned_hours": hours(train_rows, "cleaned_duration"),
            "silence_removed_hours": round(sum(float(r["total_silence_removed"]) for r in train_rows) / 3600.0, 2),
            "cleaning_status_counts": dict(sorted(Counter(r["cleaning_status"] for r in train_rows).items())),
        },
        "validation": {
            "rows": len(val_rows),
            "clean": len(val_clean),
            "excluded": len(val_excluded),
            "excluded_by_language": dict(sorted(Counter(r["language"] for r in val_excluded).items())),
            "original_hours": hours(val_rows, "original_duration"),
            "cleaned_hours": hours(val_rows, "cleaned_duration"),
            "note": "validation split kept FULL in the DatasetDict for comparable evaluation; excluded list is analysis-only",
        },
    }
    json_dump(impact, args.reports_dir / "cleaning_impact_report.json")
    json_dump(impact, args.reports_dir / "audio_cleaning_summary.json")
    print(json.dumps(impact["train"]["bucket_counts"], indent=2))
    print(f"Saved cleaned DatasetDict to {args.output_dataset / 'hf_dataset'}")
    print(f"Impact report: {args.reports_dir / 'cleaning_impact_report.json'}")


if __name__ == "__main__":
    main()
