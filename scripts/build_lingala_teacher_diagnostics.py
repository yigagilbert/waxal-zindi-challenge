#!/usr/bin/env python3
"""Build Lingala teacher-disagreement diagnostics from WAXAL labels and ASR predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_csv_dicts  # noqa: E402
from waxal.scoring import compute_group_metrics  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


OUTPUT_FIELDS = [
    "ID",
    "language",
    "original_transcript",
    "teacher_prediction",
    "teacher_model",
    "wer_original_vs_teacher",
    "cer_original_vs_teacher",
    "combined_original_vs_teacher",
    "duration_seconds",
    "chars_per_second",
    "words_per_second",
    "audio_quality_flags",
    "original_chars",
    "original_words",
    "teacher_chars",
    "teacher_words",
    "teacher_empty",
    "teacher_dot_only",
    "teacher_very_short",
    "original_suspicious",
    "teacher_suspicious",
    "recommendation",
    "reasons",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--audio-quality", type=Path, default=Path("outputs/quality/audio_quality_train.csv"))
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lingala_teacher/alvin_teacher_disagreement_lingala_train.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/lingala_teacher/alvin_teacher_disagreement_lingala_train.summary.json"),
    )
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--agree-threshold", type=float, default=0.35)
    parser.add_argument("--review-threshold", type=float, default=0.65)
    parser.add_argument("--correction-threshold", type=float, default=0.85)
    parser.add_argument("--min-duration", type=float, default=0.30)
    parser.add_argument("--max-duration", type=float, default=50.0)
    parser.add_argument("--max-chars-per-second", type=float, default=70.0)
    parser.add_argument("--max-words-per-second", type=float, default=12.0)
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


def to_float(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def target_from_metadata(row: dict[str, str]) -> str:
    return row.get("Target") or row.get("transcription") or ""


def text_flags(text: str) -> dict[str, Any]:
    normalized = normalize_text(text, "raw")
    stripped = normalized.strip()
    words = normalized.split()
    return {
        "chars": len(normalized),
        "words": len(words),
        "empty": stripped == "",
        "dot_only": stripped in {".", "...", ",", "?", "!", ";", ":"},
        "very_short": len(normalized) < 12 or len(words) < 3,
    }


def score_pair(original: str, teacher: str, normalization: str) -> dict[str, float]:
    metrics = compute_group_metrics([original], [teacher], normalization=normalization)
    return {
        "wer": float(metrics["wer"]),
        "cer": float(metrics["cer"]),
        "combined": float(metrics["combined"]),
    }


def choose_recommendation(
    original: str,
    teacher: str,
    quality_row: dict[str, str] | None,
    score: dict[str, float],
    args: argparse.Namespace,
) -> tuple[str, list[str], bool, bool]:
    original_info = text_flags(original)
    teacher_info = text_flags(teacher)
    reasons: list[str] = []
    quality_flags = set()
    duration = 0.0
    chars_per_second = 0.0
    words_per_second = 0.0
    if quality_row:
        duration = to_float(quality_row.get("duration_seconds"))
        chars_per_second = to_float(quality_row.get("chars_per_second"))
        words_per_second = to_float(quality_row.get("words_per_second"))
        quality_flags = {flag for flag in quality_row.get("quality_flags", "").split(";") if flag}
    else:
        reasons.append("missing_audio_quality")

    original_suspicious = False
    teacher_suspicious = bool(teacher_info["empty"] or teacher_info["dot_only"] or teacher_info["very_short"])
    if original_info["empty"]:
        original_suspicious = True
        reasons.append("empty_original")
    if duration < args.min_duration:
        original_suspicious = True
        reasons.append("too_short_audio")
    if duration > args.max_duration:
        reasons.append("long_audio")
    if "low_energy" in quality_flags:
        reasons.append("low_energy")
    if "clipped" in quality_flags:
        reasons.append("clipped")
    if chars_per_second > args.max_chars_per_second or words_per_second > args.max_words_per_second:
        original_suspicious = True
        reasons.append("original_text_rate_extreme")
    if teacher_suspicious:
        reasons.append("teacher_blank_or_short")

    combined = score["combined"]
    if teacher_info["empty"] or teacher_info["dot_only"]:
        return "keep_original", reasons, original_suspicious, teacher_suspicious
    if original_suspicious and not teacher_suspicious:
        return "manual_review", reasons + ["original_suspicious_teacher_plausible"], True, teacher_suspicious
    if combined <= args.agree_threshold:
        return "keep_original", reasons + ["teacher_agrees"], original_suspicious, teacher_suspicious
    if combined >= args.correction_threshold and not teacher_suspicious:
        return "potential_teacher_correction", reasons + ["extreme_teacher_disagreement"], original_suspicious, teacher_suspicious
    if combined >= args.review_threshold:
        return "suspicious_label", reasons + ["high_teacher_disagreement"], original_suspicious, teacher_suspicious
    return "manual_review", reasons + ["moderate_teacher_disagreement"], original_suspicious, teacher_suspicious


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metadata_rows = load_rows(args.metadata)
    quality_rows = load_rows(args.audio_quality)
    teacher_rows = load_rows(args.teacher_predictions)
    require_columns(args.metadata, metadata_rows, {"ID", "Target", "language"})
    require_columns(args.audio_quality, quality_rows, {"ID"})
    require_columns(args.teacher_predictions, teacher_rows, {"ID", "Target"})

    quality_by_id = {row["ID"]: row for row in quality_rows}
    teacher_by_id = {row["ID"]: row for row in teacher_rows}
    lingala_rows = [row for row in metadata_rows if row.get("language") == "lin"]

    output_rows: list[dict[str, str]] = []
    recommendation_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    missing_teacher = 0
    score_values: list[float] = []

    for row in lingala_rows:
        example_id = row["ID"]
        original = target_from_metadata(row)
        teacher_row = teacher_by_id.get(example_id)
        if teacher_row is None:
            missing_teacher += 1
            teacher_prediction = ""
            teacher_model = ""
        else:
            teacher_prediction = teacher_row.get("Target", "")
            teacher_model = teacher_row.get("teacher_model") or teacher_row.get("model") or ""
        score = score_pair(original, teacher_prediction, args.normalization)
        score_values.append(score["combined"])
        quality_row = quality_by_id.get(example_id)
        recommendation, reasons, original_suspicious, teacher_suspicious = choose_recommendation(
            original,
            teacher_prediction,
            quality_row,
            score,
            args,
        )
        recommendation_counts[recommendation] += 1
        reason_counts.update(reasons)
        original_info = text_flags(original)
        teacher_info = text_flags(teacher_prediction)
        quality_row = quality_row or {}
        output_rows.append(
            {
                "ID": example_id,
                "language": "lin",
                "original_transcript": original,
                "teacher_prediction": teacher_prediction,
                "teacher_model": teacher_model,
                "wer_original_vs_teacher": f"{score['wer']:.6f}",
                "cer_original_vs_teacher": f"{score['cer']:.6f}",
                "combined_original_vs_teacher": f"{score['combined']:.6f}",
                "duration_seconds": quality_row.get("duration_seconds", ""),
                "chars_per_second": quality_row.get("chars_per_second", ""),
                "words_per_second": quality_row.get("words_per_second", ""),
                "audio_quality_flags": quality_row.get("quality_flags", ""),
                "original_chars": str(original_info["chars"]),
                "original_words": str(original_info["words"]),
                "teacher_chars": str(teacher_info["chars"]),
                "teacher_words": str(teacher_info["words"]),
                "teacher_empty": "true" if teacher_info["empty"] else "false",
                "teacher_dot_only": "true" if teacher_info["dot_only"] else "false",
                "teacher_very_short": "true" if teacher_info["very_short"] else "false",
                "original_suspicious": "true" if original_suspicious else "false",
                "teacher_suspicious": "true" if teacher_suspicious else "false",
                "recommendation": recommendation,
                "reasons": ";".join(dict.fromkeys(reasons)),
            }
        )

    write_csv(args.output, output_rows)
    summary = {
        "metadata": str(args.metadata),
        "audio_quality": str(args.audio_quality),
        "teacher_predictions": str(args.teacher_predictions),
        "normalization": args.normalization,
        "lingala_total": len(lingala_rows),
        "teacher_predictions_matched": len(lingala_rows) - missing_teacher,
        "missing_teacher": missing_teacher,
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "mean_teacher_disagreement": sum(score_values) / len(score_values) if score_values else None,
        "outputs": {
            "diagnostics": str(args.output),
        },
    }
    json_dump(summary, args.summary_output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
