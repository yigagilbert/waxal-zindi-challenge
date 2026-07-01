"""Data loading helpers for the WAXAL Zindi files and Hugging Face dataset."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

TARGET_LANGUAGES = ("lin", "sna", "lug")
LANGUAGE_CONFIGS = {
    "lin": "lin_asr",
    "sna": "sna_asr",
    "lug": "lug_asr",
}

KNOWN_DOWNLOAD_DIR = Path(
    "/Users/sunbird/Downloads/google-waxal-asr-challenge20260630-10570-elxebu"
)


def default_raw_dir() -> Path:
    """Return the raw-data directory, preferring env var then known download path."""
    env = os.environ.get("WAXAL_RAW_DIR")
    if env:
        return Path(env).expanduser()
    if KNOWN_DOWNLOAD_DIR.exists():
        return KNOWN_DOWNLOAD_DIR
    return Path("data/raw")


def read_csv_dicts(path: str | Path) -> tuple[list[str], list[dict[str, str]], list[dict]]:
    """Read a CSV using robust quoting and backslash escaping.

    The Zindi Train.csv contains commas, quotes, apostrophes, and newlines in
    transcriptions. Python's csv module handles this cleanly when configured
    explicitly.
    """
    rows: list[dict[str, str]] = []
    bad_rows: list[dict] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(
            f,
            quotechar='"',
            escapechar="\\",
            doublequote=True,
            strict=False,
        )
        fieldnames = reader.fieldnames or []
        for line_number, row in enumerate(reader, start=2):
            extras = row.pop(None, None)
            if extras:
                bad_rows.append({"line_number": line_number, "extra_fields": extras})
            rows.append({k: "" if v is None else v for k, v in row.items()})
    return fieldnames, rows, bad_rows


def require_columns(name: str, columns: Sequence[str], required: Sequence[str]) -> None:
    """Raise if a table is missing expected columns."""
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}; got {columns}")


def load_zindi_train(raw_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Load Train.csv."""
    base = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    columns, rows, bad = read_csv_dicts(base / "Train.csv")
    require_columns("Train.csv", columns, ["id", "transcription", "language", "original_split"])
    if bad:
        raise ValueError(f"Train.csv has rows with extra fields: {bad[:3]}")
    return rows


def load_zindi_test(raw_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Load Test.csv."""
    base = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    columns, rows, bad = read_csv_dicts(base / "Test.csv")
    require_columns("Test.csv", columns, ["ID"])
    if bad:
        raise ValueError(f"Test.csv has rows with extra fields: {bad[:3]}")
    return rows


def load_sample_submission(raw_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Load SampleSubmission.csv."""
    base = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    columns, rows, bad = read_csv_dicts(base / "SampleSubmission.csv")
    require_columns("SampleSubmission.csv", columns, ["ID", "Target"])
    if bad:
        raise ValueError(f"SampleSubmission.csv has rows with extra fields: {bad[:3]}")
    return rows


def id_language(example_id: str) -> str:
    """Infer language prefix from a WAXAL ID such as lug_96114."""
    return example_id.split("_", 1)[0]


def split_rows(
    rows: Iterable[dict[str, str]],
    *,
    languages: Sequence[str] | None = None,
    original_split: str | None = None,
) -> list[dict[str, str]]:
    """Filter Zindi train rows by language and/or original split."""
    language_set = set(languages) if languages else None
    out: list[dict[str, str]] = []
    for row in rows:
        if language_set is not None and row["language"] not in language_set:
            continue
        if original_split is not None and row["original_split"] != original_split:
            continue
        out.append(row)
    return out


def count_by(rows: Iterable[dict[str, str]], key: str) -> dict[str, int]:
    """Return sorted counts for a row key."""
    return dict(sorted(Counter(row[key] for row in rows).items()))


def id_set(rows: Iterable[dict[str, str]], key: str = "id") -> set[str]:
    """Collect IDs from rows."""
    return {row[key] for row in rows}


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, str]], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_prediction_csv(path: str | Path) -> list[dict[str, str]]:
    """Read an ID,Target prediction file."""
    columns, rows, bad = read_csv_dicts(path)
    require_columns(Path(path).name, columns, ["ID", "Target"])
    if bad:
        raise ValueError(f"{path} has rows with extra fields: {bad[:3]}")
    return rows


def references_from_validation_csv(path: str | Path) -> list[dict[str, str]]:
    """Load validation references from a processed or raw CSV."""
    columns, rows, bad = read_csv_dicts(path)
    if bad:
        raise ValueError(f"{path} has rows with extra fields: {bad[:3]}")
    if "ID" in columns and "Target" in columns:
        return rows
    if "id" in columns and "transcription" in columns:
        converted = []
        for row in rows:
            converted.append(
                {
                    "ID": row["id"],
                    "Target": row["transcription"],
                    "language": row.get("language", id_language(row["id"])),
                }
            )
        return converted
    raise ValueError(
        f"{path} must contain either ID,Target or id,transcription columns; got {columns}"
    )

