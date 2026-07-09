#!/usr/bin/env python3
"""Audit KasuleTrevor/Lingala_100hrs before any training use."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_csv_dicts  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


SAFE_TEXT_RE = re.compile(r"^[\w\s'.,!?;:()\\/\-]+$", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="KasuleTrevor/Lingala_100hrs")
    parser.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--test-ids", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lingala_external/lingala_100hrs_audit.json"))
    parser.add_argument(
        "--source-stats-output",
        type=Path,
        default=Path("outputs/lingala_external/lingala_100hrs_source_stats.csv"),
    )
    return parser.parse_args()


def q(values: list[float], prob: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = prob * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p05": q(values, 0.05),
        "p95": q(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def read_optional_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    _, rows, bad = read_csv_dicts(path)
    if bad:
        raise ValueError(f"{path} has malformed rows with extra fields: {bad[:3]}")
    return rows


def text_value(row: dict[str, Any]) -> str:
    for key in ("text", "sentence", "transcription", "Target", "transcript"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def duration_seconds(row: dict[str, Any]) -> float | None:
    audio = row.get("audio")
    if isinstance(audio, dict):
        if audio.get("array") is not None and audio.get("sampling_rate"):
            return len(audio["array"]) / float(audio["sampling_rate"])
        if audio.get("duration") is not None:
            return float(audio["duration"])
    for key in ("duration", "duration_seconds", "audio_duration"):
        if row.get(key) not in {None, ""}:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def source_value(row: dict[str, Any]) -> str:
    for key in ("source", "dataset", "origin", "corpus"):
        if row.get(key) not in {None, ""}:
            return str(row[key])
    return "unknown"


def row_id_value(row: dict[str, Any]) -> str:
    for key in ("ID", "id", "path", "audio_id", "file"):
        if row.get(key) not in {None, ""}:
            return str(row[key])
    audio = row.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return ""


def weird_chars(text: str) -> list[str]:
    normalized = normalize_text(text, "raw")
    return sorted({char for char in normalized if not SAFE_TEXT_RE.match(char)})


def write_source_stats(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "source",
        "rows",
        "empty_text",
        "mean_duration",
        "median_duration",
        "mean_text_chars",
        "median_text_chars",
        "duplicate_texts",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
    except ImportError as exc:
        raise RuntimeError("Install datasets[audio] before auditing external datasets.") from exc

    dataset_info: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "license_from_dataset_info": "",
        "description_present": False,
        "configs": [],
        "safety_decision": "audit_only_until_license_and_source_overlap_are_manually_confirmed",
    }
    try:
        builder = load_dataset_builder(args.dataset_name)
        dataset_info["license_from_dataset_info"] = getattr(builder.info, "license", "") or ""
        dataset_info["description_present"] = bool(getattr(builder.info, "description", "") or "")
    except Exception as exc:
        dataset_info["builder_error"] = f"{type(exc).__name__}: {exc}"
    try:
        dataset_info["configs"] = get_dataset_config_names(args.dataset_name)
    except Exception as exc:
        dataset_info["configs_error"] = f"{type(exc).__name__}: {exc}"

    waxal_rows = read_optional_rows(args.metadata)
    waxal_texts = {
        normalize_text(row.get("Target") or row.get("transcription") or "", "language_safe")
        for row in waxal_rows
        if row.get("language") == "lin"
    }
    test_rows = read_optional_rows(args.test_ids)
    waxal_test_ids = {row.get("ID", "") for row in test_rows}

    split_reports: dict[str, Any] = {}
    source_buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "empty_text": 0,
            "durations": [],
            "text_chars": [],
            "texts": Counter(),
        }
    )
    all_texts: Counter[str] = Counter()
    possible_text_overlap = 0
    possible_id_overlap = 0

    for split in args.splits:
        ds = load_dataset(args.dataset_name, split=split, streaming=args.streaming)
        rows_seen = 0
        durations: list[float] = []
        text_chars: list[int] = []
        text_words: list[int] = []
        empty_text = 0
        weird_counter: Counter[str] = Counter()
        source_counter: Counter[str] = Counter()
        split_texts: Counter[str] = Counter()
        sample_rate_counter: Counter[int] = Counter()
        observed_columns: list[str] = []

        iterator = iter(ds)
        for row in iterator:
            if args.max_rows_per_split is not None and rows_seen >= args.max_rows_per_split:
                break
            rows_seen += 1
            if not observed_columns:
                observed_columns = sorted(row.keys())
            audio_value = row.get("audio")
            if isinstance(audio_value, dict) and audio_value.get("sampling_rate"):
                sample_rate_counter[int(audio_value["sampling_rate"])] += 1
            text = normalize_text(text_value(row), "raw")
            source = source_value(row)
            source_counter[source] += 1
            duration = duration_seconds(row)
            if duration is not None and math.isfinite(duration):
                durations.append(duration)
                source_buckets[(split, source)]["durations"].append(duration)
            if not text:
                empty_text += 1
                source_buckets[(split, source)]["empty_text"] += 1
            chars = len(text)
            words = len(text.split())
            text_chars.append(chars)
            text_words.append(words)
            source_buckets[(split, source)]["rows"] += 1
            source_buckets[(split, source)]["text_chars"].append(chars)
            normalized_for_overlap = normalize_text(text, "language_safe")
            if normalized_for_overlap:
                split_texts[normalized_for_overlap] += 1
                all_texts[normalized_for_overlap] += 1
                source_buckets[(split, source)]["texts"][normalized_for_overlap] += 1
                if normalized_for_overlap in waxal_texts:
                    possible_text_overlap += 1
            weird_counter.update(weird_chars(text))
            row_id = row_id_value(row)
            if row_id in waxal_test_ids:
                possible_id_overlap += 1

        split_reports[split] = {
            "rows_seen": rows_seen,
            "observed_columns": observed_columns,
            "observed_sampling_rates": dict(sorted(sample_rate_counter.items())),
            "total_hours": round(sum(durations) / 3600.0, 2) if durations else None,
            "source_counts": dict(sorted(source_counter.items())),
            "duration_seconds": summary(durations),
            "text_chars": summary(text_chars),
            "text_words": summary(text_words),
            "empty_text": empty_text,
            "duplicate_text_count": sum(count - 1 for count in split_texts.values() if count > 1),
            "weird_char_counts": dict(sorted(weird_counter.items())),
            "note": "Full row count only if --max-rows-per-split is omitted.",
        }

    source_rows = []
    for (split, source), bucket in sorted(source_buckets.items()):
        source_rows.append(
            {
                "split": split,
                "source": source,
                "rows": str(bucket["rows"]),
                "empty_text": str(bucket["empty_text"]),
                "mean_duration": "" if not bucket["durations"] else f"{mean(bucket['durations']):.6f}",
                "median_duration": "" if not bucket["durations"] else f"{median(bucket['durations']):.6f}",
                "mean_text_chars": "" if not bucket["text_chars"] else f"{mean(bucket['text_chars']):.6f}",
                "median_text_chars": "" if not bucket["text_chars"] else f"{median(bucket['text_chars']):.6f}",
                "duplicate_texts": str(sum(count - 1 for count in bucket["texts"].values() if count > 1)),
            }
        )
    write_source_stats(args.source_stats_output, source_rows)

    sources_by_text: dict[str, set[str]] = defaultdict(set)
    for (split, source), bucket in source_buckets.items():
        for text in bucket["texts"]:
            sources_by_text[text].add(source)
    payload = {
        "dataset_info": dataset_info,
        "splits": split_reports,
        "overall": {
            "duplicate_text_count": sum(count - 1 for count in all_texts.values() if count > 1),
            "texts_shared_across_sources": sum(1 for sources in sources_by_text.values() if len(sources) > 1),
            "possible_text_overlap_with_waxal_lingala_train": possible_text_overlap,
            "possible_id_overlap_with_waxal_test": possible_id_overlap,
        },
        "source_stats_output": str(args.source_stats_output),
        "manual_license_notes": {
            "dataset_card_status": "Hugging Face page exposes rows/sources, but source-level licensing must be manually verified before training.",
            "expected_sources_from_model_name": ["FLEURS", "AMMI", "AFRIVOICE", "LRSC"],
            "decision": "Do not train on this dataset until license and Phase 1 leakage checks are signed off.",
        },
    }
    json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
