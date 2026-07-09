#!/usr/bin/env python3
"""Full audio/transcript consistency audit across prepared datasets.

Read-only: decodes every example, computes audio stats and transcript/audio
rate features, and flags likely mismatches (e.g. 1-second clips with 20-word
transcripts). Run it over data/processed and data/processed_generalization_mix.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.audio import extended_audio_stats, to_mono  # noqa: E402
from waxal.data import write_csv_rows  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402

AUDIT_FIELDS = [
    "id",
    "source_id",
    "source_dataset",
    "language",
    "split",
    "transcript",
    "transcript_chars",
    "transcript_words",
    "duration",
    "sample_rate",
    "channels",
    "rms",
    "peak",
    "silence_ratio",
    "clipping_ratio",
    "leading_silence",
    "trailing_silence",
    "chars_per_second",
    "words_per_second",
    "flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        required=True,
        help="Prepared dataset dir containing hf_dataset. Repeat for multiple sources.",
    )
    parser.add_argument("--splits", nargs="*", default=["train", "validation"])
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=8)
    # flag thresholds (documented in docs/DATA_QUALITY_AUDIT.md)
    parser.add_argument("--short-audio-duration", type=float, default=1.5)
    parser.add_argument("--short-audio-max-words", type=int, default=8)
    parser.add_argument("--short2-duration", type=float, default=2.0)
    parser.add_argument("--short2-chars", type=int, default=20)
    parser.add_argument("--silent-rms", type=float, default=0.005)
    parser.add_argument("--silent-ratio", type=float, default=0.95)
    parser.add_argument("--max-chars-per-second", type=float, default=35.0)
    parser.add_argument("--min-chars-per-second", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/data_quality/full_audio_text_audit.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/data_quality/full_audio_text_audit_summary.json"))
    return parser.parse_args()


def compute_flags(record: dict, args: argparse.Namespace) -> list[str]:
    flags = []
    duration = record["duration"]
    chars = record["transcript_chars"]
    words = record["transcript_words"]
    cps = record["chars_per_second"]
    if chars == 0:
        flags.append("empty_transcript")
    elif record["transcript"].strip() and set(record["transcript"].strip()) <= set(".,!?;:-"):
        flags.append("punctuation_only_transcript")
    if duration <= 0.05 or record["rms"] == 0.0:
        flags.append("empty_audio")
    if record["rms"] < args.silent_rms or record["silence_ratio"] > args.silent_ratio:
        flags.append("likely_silent")
    if duration < args.short_audio_duration and words > args.short_audio_max_words:
        flags.append("short_audio_long_transcript")
    if duration < args.short2_duration and chars > args.short2_chars:
        flags.append("short_audio_many_chars")
    if cps > args.max_chars_per_second:
        flags.append("chars_per_second_too_high")
    if 0 < cps < args.min_chars_per_second and duration > 4.0:
        flags.append("chars_per_second_too_low")
    if record["clipping_ratio"] > 0.05:
        flags.append("heavy_clipping")
    if "likely_silent" in flags and chars > 10:
        flags.append("audio_transcript_mismatch")
    if "short_audio_long_transcript" in flags or "chars_per_second_too_high" in flags:
        flags.append("audio_transcript_mismatch")
    return sorted(set(flags))


def audit_split(ds, *, source_name: str, split: str, args: argparse.Namespace) -> list[dict]:
    def process(batch):
        out = defaultdict(list)
        size = len(batch["ID"])
        for i in range(size):
            audio = batch["audio"][i]
            arr = to_mono(audio["array"])
            sr = int(audio.get("sampling_rate", 16_000))
            stats = extended_audio_stats(arr, sr)
            transcript = normalize_text(str(batch.get("transcription", [""] * size)[i] or ""), "language_safe")
            duration = stats["duration"]
            record = {
                "id": batch["ID"][i],
                "source_id": batch["ID"][i],
                "source_dataset": (batch.get("source_dataset") or [source_name] * size)[i] or source_name,
                "language": batch["language"][i],
                "split": split,
                "transcript": transcript[:300],
                "transcript_chars": len(transcript),
                "transcript_words": len(transcript.split()),
                "duration": round(duration, 3),
                "sample_rate": sr,
                "channels": 1,
                "rms": round(stats["rms"], 6),
                "peak": round(stats["peak"], 4),
                "silence_ratio": round(stats["silence_ratio"], 4),
                "clipping_ratio": round(stats["clipping_ratio"], 5),
                "leading_silence": round(stats["leading_silence"], 3),
                "trailing_silence": round(stats["trailing_silence"], 3),
                "chars_per_second": round(len(transcript) / duration, 3) if duration > 0 else 0.0,
                "words_per_second": round(len(transcript.split()) / duration, 3) if duration > 0 else 0.0,
            }
            record["flags"] = "|".join(compute_flags(record, args))
            for key, value in record.items():
                out[key].append(value)
        return dict(out)

    keep = [c for c in AUDIT_FIELDS if c in ds.column_names]
    mapped = ds.map(
        process,
        batched=True,
        batch_size=16,
        num_proc=args.num_proc if len(ds) > 256 else None,
        remove_columns=[c for c in ds.column_names if c not in keep],
        desc=f"Auditing {source_name}/{split}",
    )
    return [dict(zip(mapped.column_names, values, strict=True)) for values in zip(*[mapped[c] for c in AUDIT_FIELDS], strict=True)]


def main() -> None:
    args = parse_args()
    from datasets import load_from_disk

    all_rows: list[dict] = []
    for dataset_dir in args.dataset_dir:
        dataset_dict = load_from_disk(dataset_dir / "hf_dataset")
        for split in args.splits:
            if split not in dataset_dict:
                continue
            ds = dataset_dict[split]
            if "transcription" not in ds.column_names:
                print(f"Skipping {dataset_dir}/{split}: no transcription column (never audit test labels).")
                continue
            if args.max_samples_per_split and len(ds) > args.max_samples_per_split:
                ds = ds.select(range(args.max_samples_per_split))
            all_rows.extend(audit_split(ds, source_name=str(dataset_dir), split=split, args=args))

    # de-duplicate on (id, split): generalization mix overlaps data/processed
    seen: set[tuple[str, str]] = set()
    rows = []
    for row in all_rows:
        key = (row["id"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    write_csv_rows(args.output, rows, AUDIT_FIELDS)

    flag_counts: Counter[str] = Counter()
    flag_by_language: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for flag in filter(None, row["flags"].split("|")):
            flag_counts[flag] += 1
            flag_by_language[row["language"]][flag] += 1
    flagged_rows = [r for r in rows if r["flags"]]
    summary = {
        "total_rows": len(rows),
        "flagged_rows": len(flagged_rows),
        "flag_counts": dict(sorted(flag_counts.items())),
        "flags_by_language": {k: dict(sorted(v.items())) for k, v in sorted(flag_by_language.items())},
        "by_language_rows": dict(sorted(Counter(r["language"] for r in rows).items())),
        "by_source_rows": dict(sorted(Counter(r["source_dataset"] for r in rows).items())),
        "mismatch_rows": sum(1 for r in rows if "audio_transcript_mismatch" in r["flags"]),
        "total_hours": round(sum(r["duration"] for r in rows) / 3600.0, 2),
        "thresholds": {
            "short_audio_duration": args.short_audio_duration,
            "short_audio_max_words": args.short_audio_max_words,
            "short2_duration": args.short2_duration,
            "short2_chars": args.short2_chars,
            "silent_rms": args.silent_rms,
            "silent_ratio": args.silent_ratio,
            "max_chars_per_second": args.max_chars_per_second,
            "min_chars_per_second": args.min_chars_per_second,
        },
    }
    json_dump(summary, args.summary_output)
    print(f"Audited {len(rows)} rows -> {args.output}")
    print(f"Flagged {len(flagged_rows)} rows; mismatches: {summary['mismatch_rows']}")


if __name__ == "__main__":
    main()
