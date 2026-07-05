#!/usr/bin/env python3
"""Build a WAXAL training manifest with Alvin-assisted Lingala filtering."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_csv_dicts  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


FIELDS = [
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
    "duration_seconds",
    "chars_per_second",
    "words_per_second",
    "quality_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-clean-manifest", type=Path, default=Path("data/quality/clean_train_sunbird_lug.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument(
        "--teacher-diagnostics",
        type=Path,
        default=Path("outputs/lingala_teacher/alvin_teacher_disagreement_lingala_train.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/lingala_teacher/alvin_lingala_bucket_summary.json"),
    )
    parser.add_argument(
        "--include-review-original",
        action="store_true",
        help="Include manual_review Lingala rows with original labels in the combined manifest.",
    )
    parser.add_argument(
        "--allow-teacher-corrections",
        action="store_true",
        help="Use teacher labels for potential_teacher_correction rows. Off by default for safety.",
    )
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


def metadata_by_id(path: Path) -> dict[str, dict[str, str]]:
    rows = load_rows(path)
    require_columns(path, rows, {"ID", "Target", "language"})
    return {row["ID"]: row for row in rows}


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manifest_row(diag: dict[str, str], meta: dict[str, str], *, target: str, bucket: str, action: str, label_source: str) -> dict[str, str]:
    return {
        "ID": diag["ID"],
        "Target": target,
        "language": "lin",
        "original_split": meta.get("original_split", "train"),
        "quality_bucket": bucket,
        "action": action,
        "label_source": label_source,
        "cleaning_reasons": diag.get("reasons", ""),
        "original_target": diag.get("original_transcript", ""),
        "teacher_target": diag.get("teacher_prediction", ""),
        "teacher_model": diag.get("teacher_model", ""),
        "teacher_combined": diag.get("combined_original_vs_teacher", ""),
        "teacher_wer": diag.get("wer_original_vs_teacher", ""),
        "teacher_cer": diag.get("cer_original_vs_teacher", ""),
        "duration_seconds": diag.get("duration_seconds", ""),
        "chars_per_second": diag.get("chars_per_second", ""),
        "words_per_second": diag.get("words_per_second", ""),
        "quality_flags": diag.get("audio_quality_flags", ""),
    }


def main() -> None:
    args = parse_args()
    base_rows = load_rows(args.base_clean_manifest)
    require_columns(args.base_clean_manifest, base_rows, {"ID", "Target", "language"})
    metadata = metadata_by_id(args.metadata)
    diagnostics = load_rows(args.teacher_diagnostics)
    require_columns(
        args.teacher_diagnostics,
        diagnostics,
        {"ID", "original_transcript", "teacher_prediction", "recommendation"},
    )

    clean_lingala: list[dict[str, str]] = []
    review_lingala: list[dict[str, str]] = []
    suspicious_lingala: list[dict[str, str]] = []
    excluded_lingala: list[dict[str, str]] = []
    action_counts: Counter[str] = Counter()
    label_source_counts: Counter[str] = Counter()

    for diag in diagnostics:
        example_id = diag["ID"]
        meta = metadata.get(example_id, {})
        recommendation = diag.get("recommendation", "")
        original = diag.get("original_transcript", "")
        teacher = diag.get("teacher_prediction", "")
        action_counts[recommendation] += 1

        if recommendation == "keep_original":
            row = manifest_row(diag, meta, target=original, bucket="clean", action=recommendation, label_source="original")
            clean_lingala.append(row)
            label_source_counts["original"] += 1
        elif recommendation == "manual_review":
            row = manifest_row(diag, meta, target=original, bucket="review", action=recommendation, label_source="original")
            review_lingala.append(row)
            if args.include_review_original:
                clean_lingala.append({**row, "quality_bucket": "clean_review_included"})
                label_source_counts["original_review_included"] += 1
        elif recommendation == "potential_teacher_correction":
            if args.allow_teacher_corrections:
                row = manifest_row(diag, meta, target=teacher, bucket="clean", action=recommendation, label_source="teacher")
                clean_lingala.append(row)
                label_source_counts["teacher"] += 1
            else:
                row = manifest_row(diag, meta, target=original, bucket="suspicious", action=recommendation, label_source="original")
                suspicious_lingala.append(row)
        elif recommendation == "suspicious_label":
            row = manifest_row(diag, meta, target=original, bucket="suspicious", action=recommendation, label_source="original")
            suspicious_lingala.append(row)
        else:
            row = manifest_row(diag, meta, target=original, bucket="excluded", action=recommendation or "unknown", label_source="original")
            excluded_lingala.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = args.output_dir / "lingala_alvin_clean.csv"
    suspicious_path = args.output_dir / "lingala_alvin_suspicious.csv"
    review_path = args.output_dir / "lingala_alvin_review.csv"
    excluded_path = args.output_dir / "lingala_alvin_excluded.csv"
    combined_path = args.output_dir / "clean_train_alvin_lingala_v1.csv"
    write_csv(clean_path, clean_lingala, FIELDS)
    write_csv(suspicious_path, suspicious_lingala, FIELDS)
    write_csv(review_path, review_lingala, FIELDS)
    write_csv(excluded_path, excluded_lingala, FIELDS)

    combined = [row for row in base_rows if row.get("language") != "lin"]
    combined.extend(clean_lingala)
    write_csv(combined_path, combined, FIELDS)

    summary = {
        "base_clean_manifest": str(args.base_clean_manifest),
        "metadata": str(args.metadata),
        "teacher_diagnostics": str(args.teacher_diagnostics),
        "include_review_original": args.include_review_original,
        "allow_teacher_corrections": args.allow_teacher_corrections,
        "base_rows": len(base_rows),
        "base_non_lingala_rows": len([row for row in base_rows if row.get("language") != "lin"]),
        "lingala_diagnostic_rows": len(diagnostics),
        "clean_lingala_rows": len(clean_lingala),
        "review_lingala_rows": len(review_lingala),
        "suspicious_lingala_rows": len(suspicious_lingala),
        "excluded_lingala_rows": len(excluded_lingala),
        "combined_manifest_rows": len(combined),
        "action_counts": dict(sorted(action_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "outputs": {
            "clean_lingala": str(clean_path),
            "suspicious_lingala": str(suspicious_path),
            "review_lingala": str(review_path),
            "excluded_lingala": str(excluded_path),
            "combined_manifest": str(combined_path),
        },
    }
    json_dump(summary, args.summary_output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
