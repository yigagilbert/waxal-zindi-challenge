#!/usr/bin/env python3
"""Create Luganda training manifests cleaned with a trusted Sunbird teacher."""

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
    "Target",
    "language",
    "original_split",
    "quality_bucket",
    "action",
    "label_source",
    "cleaning_reasons",
    "original_target",
    "teacher_target",
    "teacher_model",
    "teacher_combined",
    "teacher_wer",
    "teacher_cer",
    "teacher_plausible",
    "duration_seconds",
    "chars_per_second",
    "words_per_second",
    "audio_rms",
    "peak_amplitude",
    "approximate_silence_ratio",
    "quality_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--audio-quality", type=Path, default=Path("outputs/quality/audio_quality_train.csv"))
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--base-clean-manifest", type=Path, default=Path("data/quality/clean_train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/quality/luganda_teacher_cleaning_summary.json"))
    parser.add_argument("--normalization", default="no_punct_lower")
    parser.add_argument(
        "--teacher-label-mode",
        choices=["agreement_only", "high_disagreement", "all_plausible"],
        default="high_disagreement",
        help=(
            "agreement_only keeps original labels; high_disagreement uses teacher labels only for strong "
            "disagreements; all_plausible distills Sunbird labels for all plausible Luganda examples."
        ),
    )
    parser.add_argument("--keep-threshold", type=float, default=0.35)
    parser.add_argument("--replace-threshold", type=float, default=0.65)
    parser.add_argument("--min-teacher-chars-per-second", type=float, default=2.0)
    parser.add_argument("--max-teacher-chars-per-second", type=float, default=25.0)
    parser.add_argument("--min-teacher-words-per-second", type=float, default=0.25)
    parser.add_argument("--max-teacher-words-per-second", type=float, default=4.0)
    parser.add_argument("--near-silence-ratio", type=float, default=0.95)
    parser.add_argument("--min-duration", type=float, default=0.30)
    parser.add_argument("--max-keep-duration", type=float, default=50.0)
    parser.add_argument(
        "--include-review-original",
        action="store_true",
        help="Include review rows in the combined training manifest with original labels.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    columns, rows, bad = read_csv_dicts(path)
    if bad:
        raise ValueError(f"{path} has malformed rows with extra fields: {bad[:3]}")
    if not columns:
        raise ValueError(f"{path} has no CSV header")
    return rows


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}; got {sorted(columns)}")


def to_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def target_from_metadata(row: dict[str, str]) -> str:
    return row.get("Target") or row.get("transcription") or ""


def text_rate(text: str, duration: float) -> tuple[float, float]:
    normalized = normalize_text(text, "raw")
    if duration <= 0:
        return 0.0, 0.0
    return len(normalized) / duration, len(normalized.split()) / duration


def teacher_metrics(original: str, teacher: str, normalization: str) -> dict[str, float]:
    metrics = compute_group_metrics([original], [teacher], normalization=normalization)
    return {
        "combined": float(metrics["combined"]),
        "wer": float(metrics["wer"]),
        "cer": float(metrics["cer"]),
    }


def plausible_teacher(text: str, duration: float, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    normalized = normalize_text(text, "raw")
    if not normalized:
        return False, ["empty_teacher"]
    if len(normalized) < 3:
        reasons.append("teacher_too_short")
    cps, wps = text_rate(normalized, duration)
    if duration > 0:
        if cps < args.min_teacher_chars_per_second:
            reasons.append("teacher_cps_low")
        if cps > args.max_teacher_chars_per_second:
            reasons.append("teacher_cps_high")
        if wps < args.min_teacher_words_per_second:
            reasons.append("teacher_wps_low")
        if wps > args.max_teacher_words_per_second:
            reasons.append("teacher_wps_high")
    return not reasons, reasons


def severe_audio_or_label_issue(
    metadata_row: dict[str, str],
    quality_row: dict[str, str] | None,
    args: argparse.Namespace,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    original = normalize_text(target_from_metadata(metadata_row), "raw")
    if not original:
        reasons.append("empty_original")
    if quality_row is None:
        return False, ["missing_audio_quality"]

    flags = {flag for flag in quality_row.get("quality_flags", "").split(";") if flag}
    duration = to_float(quality_row.get("duration_seconds"))
    silence = to_float(quality_row.get("approximate_silence_ratio"))
    if duration < args.min_duration:
        reasons.append("too_short")
    if duration > args.max_keep_duration:
        reasons.append("beyond_max_keep_duration")
    if "low_energy" in flags and silence >= args.near_silence_ratio:
        reasons.append("near_silent")
    if any(flag.startswith("audio_error") for flag in flags):
        reasons.append("audio_error")
    return bool(reasons), reasons


def choose_luganda_action(
    metadata_row: dict[str, str],
    quality_row: dict[str, str] | None,
    teacher_row: dict[str, str] | None,
    args: argparse.Namespace,
) -> tuple[str, str, str, list[str], dict[str, Any]]:
    """Return (bucket, action, selected_target, reasons, diagnostics)."""
    original = target_from_metadata(metadata_row)
    teacher = teacher_row.get("Target", "") if teacher_row else ""
    teacher_model = teacher_row.get("teacher_model", "") if teacher_row else ""
    duration = to_float(quality_row.get("duration_seconds")) if quality_row else 0.0
    quality_flags = {flag for flag in (quality_row or {}).get("quality_flags", "").split(";") if flag}

    severe, severe_reasons = severe_audio_or_label_issue(metadata_row, quality_row, args)
    teacher_ok, teacher_reasons = plausible_teacher(teacher, duration, args)
    if not teacher_row:
        teacher_reasons.append("missing_teacher")

    if severe and not (not normalize_text(original, "raw") and teacher_ok):
        return "excluded", "exclude", "", severe_reasons + teacher_reasons, {
            "teacher_model": teacher_model,
            "teacher_plausible": teacher_ok,
            "teacher_combined": "",
            "teacher_wer": "",
            "teacher_cer": "",
        }

    if not teacher_ok:
        if normalize_text(original, "raw") and not severe:
            return "review", "keep_original_review_teacher_implausible", original, teacher_reasons, {
                "teacher_model": teacher_model,
                "teacher_plausible": teacher_ok,
                "teacher_combined": "",
                "teacher_wer": "",
                "teacher_cer": "",
            }
        return "excluded", "exclude", "", severe_reasons + teacher_reasons, {
            "teacher_model": teacher_model,
            "teacher_plausible": teacher_ok,
            "teacher_combined": "",
            "teacher_wer": "",
            "teacher_cer": "",
        }

    if not normalize_text(original, "raw"):
        return "clean", "replace_empty_original_with_teacher", teacher, severe_reasons, {
            "teacher_model": teacher_model,
            "teacher_plausible": teacher_ok,
            "teacher_combined": "",
            "teacher_wer": "",
            "teacher_cer": "",
        }

    metrics = teacher_metrics(original, teacher, args.normalization)
    combined = metrics["combined"]
    reasons: list[str] = []
    if "too_long" in quality_flags:
        reasons.append("long_luganda_kept")
    if "low_energy" in quality_flags:
        reasons.append("low_energy_check")
    if "text_rate_outlier" in quality_flags:
        reasons.append("original_text_rate_outlier")

    diagnostics = {
        "teacher_model": teacher_model,
        "teacher_plausible": teacher_ok,
        "teacher_combined": metrics["combined"],
        "teacher_wer": metrics["wer"],
        "teacher_cer": metrics["cer"],
    }

    if args.teacher_label_mode == "all_plausible":
        action = "replace_with_teacher_all_plausible"
        if combined <= args.keep_threshold:
            reasons.append("teacher_agrees")
        else:
            reasons.append("teacher_distillation")
        return "clean", action, teacher, reasons, diagnostics

    if combined <= args.keep_threshold:
        return "clean", "keep_original_teacher_agrees", original, reasons + ["teacher_agrees"], diagnostics

    if args.teacher_label_mode == "high_disagreement" and combined >= args.replace_threshold:
        return "clean", "replace_with_teacher_high_disagreement", teacher, reasons + ["teacher_disagrees_strongly"], diagnostics

    if combined >= args.replace_threshold:
        return "review", "review_high_disagreement", original, reasons + ["teacher_disagrees_strongly"], diagnostics

    return "review", "review_moderate_disagreement", original, reasons + ["teacher_disagrees_moderately"], diagnostics


def output_row(
    metadata_row: dict[str, str],
    quality_row: dict[str, str] | None,
    teacher_row: dict[str, str] | None,
    bucket: str,
    action: str,
    target: str,
    reasons: list[str],
    diagnostics: dict[str, Any],
) -> dict[str, str]:
    quality_row = quality_row or {}
    teacher_target = teacher_row.get("Target", "") if teacher_row else ""
    return {
        "ID": metadata_row["ID"],
        "Target": target,
        "language": "lug",
        "original_split": metadata_row.get("original_split", "train"),
        "quality_bucket": bucket,
        "action": action,
        "label_source": "teacher" if action.startswith("replace") else "original",
        "cleaning_reasons": ";".join(dict.fromkeys(reason for reason in reasons if reason)),
        "original_target": target_from_metadata(metadata_row),
        "teacher_target": teacher_target,
        "teacher_model": str(diagnostics.get("teacher_model", "")),
        "teacher_combined": format_float(diagnostics.get("teacher_combined", "")),
        "teacher_wer": format_float(diagnostics.get("teacher_wer", "")),
        "teacher_cer": format_float(diagnostics.get("teacher_cer", "")),
        "teacher_plausible": "true" if diagnostics.get("teacher_plausible") else "false",
        "duration_seconds": quality_row.get("duration_seconds", ""),
        "chars_per_second": quality_row.get("chars_per_second", ""),
        "words_per_second": quality_row.get("words_per_second", ""),
        "audio_rms": quality_row.get("audio_rms", ""),
        "peak_amplitude": quality_row.get("peak_amplitude", ""),
        "approximate_silence_ratio": quality_row.get("approximate_silence_ratio", ""),
        "quality_flags": quality_row.get("quality_flags", ""),
    }


def format_float(value: Any) -> str:
    if value in {"", None}:
        return ""
    return f"{float(value):.6f}"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def combine_with_non_luganda_clean(base_manifest: Path, luganda_rows: list[dict[str, str]], output_path: Path) -> list[dict[str, str]]:
    base_rows = load_rows(base_manifest)
    require_columns(base_manifest, base_rows, {"ID", "Target", "language"})
    combined = [row for row in base_rows if row.get("language") != "lug"]
    combined.extend(luganda_rows)
    write_csv(output_path, combined, OUTPUT_FIELDS)
    return combined


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
    lug_rows = [row for row in metadata_rows if row.get("language") == "lug"]

    clean_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for row in lug_rows:
        bucket, action, target, reasons, diagnostics = choose_luganda_action(
            row,
            quality_by_id.get(row["ID"]),
            teacher_by_id.get(row["ID"]),
            args,
        )
        out_row = output_row(
            row,
            quality_by_id.get(row["ID"]),
            teacher_by_id.get(row["ID"]),
            bucket,
            action,
            target,
            reasons,
            diagnostics,
        )
        action_counts[action] += 1
        for reason in out_row["cleaning_reasons"].split(";"):
            if reason:
                reason_counts[reason] += 1
        if bucket == "clean":
            clean_rows.append(out_row)
        elif bucket == "review":
            review_rows.append(out_row)
        else:
            excluded_rows.append(out_row)

    combined_lug_rows = list(clean_rows)
    if args.include_review_original:
        combined_lug_rows.extend(review_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = args.output_dir / "lug_sunbird_clean_train.csv"
    review_path = args.output_dir / "lug_sunbird_review_train.csv"
    excluded_path = args.output_dir / "lug_sunbird_excluded_train.csv"
    combined_path = args.output_dir / "clean_train_sunbird_lug.csv"
    write_csv(clean_path, clean_rows, OUTPUT_FIELDS)
    write_csv(review_path, review_rows, OUTPUT_FIELDS)
    write_csv(excluded_path, excluded_rows, OUTPUT_FIELDS)
    combined_rows = combine_with_non_luganda_clean(args.base_clean_manifest, combined_lug_rows, combined_path)

    summary = {
        "metadata": str(args.metadata),
        "audio_quality": str(args.audio_quality),
        "teacher_predictions": str(args.teacher_predictions),
        "base_clean_manifest": str(args.base_clean_manifest),
        "normalization": args.normalization,
        "teacher_label_mode": args.teacher_label_mode,
        "thresholds": {
            "keep_threshold": args.keep_threshold,
            "replace_threshold": args.replace_threshold,
            "min_teacher_chars_per_second": args.min_teacher_chars_per_second,
            "max_teacher_chars_per_second": args.max_teacher_chars_per_second,
            "min_teacher_words_per_second": args.min_teacher_words_per_second,
            "max_teacher_words_per_second": args.max_teacher_words_per_second,
            "max_keep_duration": args.max_keep_duration,
        },
        "luganda_total": len(lug_rows),
        "teacher_predictions_matched": sum(1 for row in lug_rows if row["ID"] in teacher_by_id),
        "clean_luganda_rows": len(clean_rows),
        "review_luganda_rows": len(review_rows),
        "excluded_luganda_rows": len(excluded_rows),
        "combined_manifest_rows": len(combined_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "outputs": {
            "lug_clean": str(clean_path),
            "lug_review": str(review_path),
            "lug_excluded": str(excluded_path),
            "combined_manifest": str(combined_path),
        },
    }
    json_dump(summary, args.summary_output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
