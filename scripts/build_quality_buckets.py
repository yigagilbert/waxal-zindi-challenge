#!/usr/bin/env python3
"""Build WAXAL clean/medium/noisy/excluded training buckets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_csv_dicts  # noqa: E402
from waxal.scoring import compute_group_metrics  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


BASE_FIELDS = [
    "ID",
    "Target",
    "language",
    "original_split",
    "quality_bucket",
    "bucket_reasons",
    "duration_seconds",
    "transcript_chars",
    "transcript_words",
    "chars_per_second",
    "words_per_second",
    "audio_rms",
    "peak_amplitude",
    "clipping_ratio",
    "approximate_silence_ratio",
    "quality_flags",
    "teacher_count",
    "teacher_mean_combined",
    "teacher_max_combined",
    "teacher_min_combined",
    "teacher_best_model",
    "teacher_scores_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-quality", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--teacher-predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/quality/quality_bucket_summary.json"))
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--clean-teacher-max-combined", type=float, default=0.55)
    parser.add_argument("--medium-teacher-max-combined", type=float, default=0.85)
    parser.add_argument("--severe-clipping-ratio", type=float, default=0.05)
    parser.add_argument("--severe-silence-ratio", type=float, default=0.95)
    parser.add_argument("--severe-max-chars-per-second", type=float, default=70.0)
    parser.add_argument("--severe-max-words-per-second", type=float, default=12.0)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    columns, rows, bad = read_csv_dicts(path)
    if bad:
        raise ValueError(f"{path} has malformed rows with extra fields: {bad[:3]}")
    if not columns:
        raise ValueError(f"{path} has no header")
    return rows


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}; got {sorted(columns)}")


def to_bool(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def metadata_target(row: dict[str, str]) -> str:
    return row.get("Target") or row.get("transcription") or ""


def load_teacher_predictions(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        rows = load_rows(path)
        require_columns(path, rows, {"ID", "Target"})
        for row in rows:
            row = dict(row)
            row.setdefault("teacher_model", path.stem)
            by_id[row["ID"]].append(row)
    return by_id


def score_teachers(
    reference: str,
    teacher_rows: list[dict[str, str]],
    *,
    normalization: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for teacher in teacher_rows:
        prediction = teacher.get("Target", "")
        metrics = compute_group_metrics([reference], [prediction], normalization=normalization)
        scores.append(
            {
                "teacher_model": teacher.get("teacher_model") or "unknown_teacher",
                "combined": metrics["combined"],
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "prediction_chars": len(normalize_text(prediction, "raw")),
            }
        )
    if not scores:
        return [], {
            "teacher_count": 0,
            "teacher_mean_combined": "",
            "teacher_max_combined": "",
            "teacher_min_combined": "",
            "teacher_best_model": "",
        }
    combined = [float(item["combined"]) for item in scores]
    best = min(scores, key=lambda item: float(item["combined"]))
    return scores, {
        "teacher_count": len(scores),
        "teacher_mean_combined": sum(combined) / len(combined),
        "teacher_max_combined": max(combined),
        "teacher_min_combined": min(combined),
        "teacher_best_model": best["teacher_model"],
    }


def choose_bucket(
    metadata_row: dict[str, str],
    quality_row: dict[str, str] | None,
    teacher_summary: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, list[str]]:
    target = normalize_text(metadata_target(metadata_row), "raw")
    reasons: list[str] = []
    if not target:
        return "excluded", ["empty_transcript"]

    if quality_row is None:
        reasons.append("missing_audio_quality")
        quality_flags: set[str] = set()
        duration = 0.0
        clipping_ratio = 0.0
        silence_ratio = 0.0
        chars_per_second = 0.0
        words_per_second = 0.0
    else:
        quality_flags = {flag for flag in quality_row.get("quality_flags", "").split(";") if flag}
        duration = to_float(quality_row.get("duration_seconds"))
        clipping_ratio = to_float(quality_row.get("clipping_ratio"))
        silence_ratio = to_float(quality_row.get("approximate_silence_ratio"))
        chars_per_second = to_float(quality_row.get("chars_per_second"))
        words_per_second = to_float(quality_row.get("words_per_second"))
        reasons.extend(sorted(quality_flags))

    if "audio_error" in ";".join(reasons):
        return "excluded", reasons
    if "too_short" in quality_flags or duration <= 0.05:
        return "excluded", reasons or ["too_short"]
    if "low_energy" in quality_flags and silence_ratio >= args.severe_silence_ratio:
        return "excluded", reasons or ["near_silent"]
    if clipping_ratio >= args.severe_clipping_ratio:
        return "excluded", reasons or ["severe_clipping"]
    if chars_per_second >= args.severe_max_chars_per_second or words_per_second >= args.severe_max_words_per_second:
        return "excluded", reasons or ["severe_text_audio_rate"]

    teacher_count = int(teacher_summary["teacher_count"])
    teacher_max = to_float(teacher_summary.get("teacher_max_combined"), default=0.0)
    teacher_mean = to_float(teacher_summary.get("teacher_mean_combined"), default=0.0)
    if teacher_count and teacher_max >= args.medium_teacher_max_combined:
        reasons.append("extreme_teacher_disagreement")
        return "noisy", reasons

    noisy_flags = {"too_long", "text_rate_outlier"}
    medium_flags = {"low_energy", "clipped"}
    if quality_flags & noisy_flags:
        return "noisy", reasons
    if teacher_count and teacher_mean >= args.clean_teacher_max_combined:
        reasons.append("moderate_teacher_disagreement")
        return "medium", reasons
    if quality_flags & medium_flags or "missing_audio_quality" in reasons:
        return "medium", reasons

    return "clean", reasons


def format_float(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.6f}"


def output_row(
    metadata_row: dict[str, str],
    quality_row: dict[str, str] | None,
    bucket: str,
    reasons: list[str],
    teacher_scores: list[dict[str, Any]],
    teacher_summary: dict[str, Any],
) -> dict[str, str]:
    quality_row = quality_row or {}
    return {
        "ID": metadata_row.get("ID") or metadata_row.get("id") or "",
        "Target": metadata_target(metadata_row),
        "language": metadata_row.get("language") or "",
        "original_split": metadata_row.get("original_split") or "",
        "quality_bucket": bucket,
        "bucket_reasons": ";".join(dict.fromkeys(reason for reason in reasons if reason)),
        "duration_seconds": quality_row.get("duration_seconds", ""),
        "transcript_chars": quality_row.get("transcript_chars", ""),
        "transcript_words": quality_row.get("transcript_words", ""),
        "chars_per_second": quality_row.get("chars_per_second", ""),
        "words_per_second": quality_row.get("words_per_second", ""),
        "audio_rms": quality_row.get("audio_rms", ""),
        "peak_amplitude": quality_row.get("peak_amplitude", ""),
        "clipping_ratio": quality_row.get("clipping_ratio", ""),
        "approximate_silence_ratio": quality_row.get("approximate_silence_ratio", ""),
        "quality_flags": quality_row.get("quality_flags", ""),
        "teacher_count": str(teacher_summary["teacher_count"]),
        "teacher_mean_combined": format_float(teacher_summary["teacher_mean_combined"]),
        "teacher_max_combined": format_float(teacher_summary["teacher_max_combined"]),
        "teacher_min_combined": format_float(teacher_summary["teacher_min_combined"]),
        "teacher_best_model": str(teacher_summary["teacher_best_model"]),
        "teacher_scores_json": json.dumps(teacher_scores, ensure_ascii=False, sort_keys=True),
    }


def write_bucket(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata_rows = load_rows(args.metadata)
    quality_rows = load_rows(args.audio_quality)
    require_columns(args.metadata, metadata_rows, {"ID", "Target", "language"})
    require_columns(args.audio_quality, quality_rows, {"ID", "quality_flags"})

    quality_by_id = {row["ID"]: row for row in quality_rows}
    teacher_by_id = load_teacher_predictions(args.teacher_predictions)

    bucket_rows: dict[str, list[dict[str, str]]] = {
        "clean": [],
        "medium": [],
        "noisy": [],
        "excluded": [],
    }
    teacher_missing = 0
    for row in metadata_rows:
        example_id = row["ID"]
        teacher_scores, teacher_summary = score_teachers(
            metadata_target(row),
            teacher_by_id.get(example_id, []),
            normalization=args.normalization,
        )
        if not teacher_scores:
            teacher_missing += 1
        bucket, reasons = choose_bucket(row, quality_by_id.get(example_id), teacher_summary, args)
        bucket_rows[bucket].append(output_row(row, quality_by_id.get(example_id), bucket, reasons, teacher_scores, teacher_summary))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for bucket, rows in bucket_rows.items():
        out = args.output_dir / f"{bucket}_{args.split_name}.csv"
        write_bucket(out, rows)
        written[bucket] = str(out)

    by_language_bucket: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reason_counts: Counter[str] = Counter()
    for bucket, rows in bucket_rows.items():
        for row in rows:
            by_language_bucket[row["language"]][bucket] += 1
            for reason in row["bucket_reasons"].split(";"):
                if reason:
                    reason_counts[reason] += 1

    summary = {
        "metadata": str(args.metadata),
        "audio_quality": str(args.audio_quality),
        "teacher_predictions": [str(path) for path in args.teacher_predictions],
        "normalization": args.normalization,
        "thresholds": {
            "clean_teacher_max_combined": args.clean_teacher_max_combined,
            "medium_teacher_max_combined": args.medium_teacher_max_combined,
            "severe_clipping_ratio": args.severe_clipping_ratio,
            "severe_silence_ratio": args.severe_silence_ratio,
            "severe_max_chars_per_second": args.severe_max_chars_per_second,
            "severe_max_words_per_second": args.severe_max_words_per_second,
        },
        "total_examples": len(metadata_rows),
        "teacher_missing_examples": teacher_missing,
        "bucket_counts": {bucket: len(rows) for bucket, rows in bucket_rows.items()},
        "by_language_bucket": {
            language: dict(sorted(counts.items()))
            for language, counts in sorted(by_language_bucket.items())
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "outputs": written,
    }
    json_dump(summary, args.summary_output)
    print(f"Wrote quality buckets under {args.output_dir}")
    print(f"Wrote bucket summary to {args.summary_output}")
    print(json.dumps(summary["bucket_counts"], indent=2, sort_keys=True))
    for language, counts in summary["by_language_bucket"].items():
        print(f"{language}: {counts}")


if __name__ == "__main__":
    main()
