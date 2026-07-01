#!/usr/bin/env python3
"""Audit the official Zindi WAXAL CSV files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import (  # noqa: E402
    default_raw_dir,
    id_language,
    load_sample_submission,
    load_zindi_test,
    load_zindi_train,
)
from waxal.utils import json_dump  # noqa: E402


def quantile(values: list[int], q: float) -> int:
    """Return an inclusive nearest-rank quantile from sorted-ish values."""
    if not values:
        return 0
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1))
    return ordered[index]


def length_stats(texts: list[str]) -> dict:
    """Compute transcript length statistics."""
    chars = [len(t) for t in texts]
    words = [len(t.split()) for t in texts]
    return {
        "chars": {
            "min": min(chars) if chars else 0,
            "median": statistics.median(chars) if chars else 0,
            "p95": quantile(chars, 0.95),
            "max": max(chars) if chars else 0,
        },
        "words": {
            "min": min(words) if words else 0,
            "median": statistics.median(words) if words else 0,
            "p95": quantile(words, 0.95),
            "max": max(words) if words else 0,
        },
    }


def char_inventory(rows: list[dict[str, str]]) -> dict:
    """Build character inventory per language."""
    by_lang: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_lang[row["language"]].update(row["transcription"])
    report = {}
    for lang, counter in sorted(by_lang.items()):
        chars = sorted(counter)
        report[lang] = {
            "num_unique": len(chars),
            "characters": "".join(chars),
            "specials": "".join(c for c in chars if not (c.isalnum() or c.isspace())),
            "non_ascii": "".join(c for c in chars if ord(c) > 127),
            "top": counter.most_common(40),
        }
    return report


def row_examples(rows: list[dict[str, str]]) -> dict:
    """Collect examples that need human inspection."""
    categories: dict[str, list[dict[str, str]]] = {
        "punctuation": [],
        "quotes": [],
        "apostrophes": [],
        "numbers": [],
        "unusual_characters": [],
        "long_transcripts": [],
        "empty_after_strip": [],
        "newlines": [],
    }

    def add(name: str, row: dict[str, str], limit: int = 8) -> None:
        if len(categories[name]) < limit:
            categories[name].append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "original_split": row["original_split"],
                    "transcription": row["transcription"][:500],
                }
            )

    sorted_by_len = sorted(rows, key=lambda r: len(r["transcription"]), reverse=True)
    for row in sorted_by_len[:8]:
        add("long_transcripts", row)

    for row in rows:
        text = row["transcription"]
        if any(unicodedata.category(c).startswith("P") for c in text):
            add("punctuation", row)
        if '"' in text or "“" in text or "”" in text:
            add("quotes", row)
        if "'" in text or "’" in text or "`" in text:
            add("apostrophes", row)
        if any(c.isdigit() for c in text):
            add("numbers", row)
        if any((not c.isalnum()) and (not c.isspace()) and not unicodedata.category(c).startswith("P") for c in text):
            add("unusual_characters", row)
        if not text.strip():
            add("empty_after_strip", row)
        if "\n" in text or "\r" in text:
            add("newlines", row)
    return categories


def build_report(raw_dir: Path) -> dict:
    """Load files and build an audit report."""
    train = load_zindi_train(raw_dir)
    test = load_zindi_test(raw_dir)
    sample = load_sample_submission(raw_dir)

    train_ids = [row["id"] for row in train]
    test_ids = [row["ID"] for row in test]
    sample_ids = [row["ID"] for row in sample]

    missing_values = {
        "train": {
            key: sum(1 for row in train if row.get(key, "") == "")
            for key in ["id", "transcription", "language", "original_split"]
        },
        "test": {"ID": sum(1 for row in test if row.get("ID", "") == "")},
        "sample_submission": {
            key: sum(1 for row in sample if row.get(key, "") == "")
            for key in ["ID", "Target"]
        },
    }
    duplicates = {
        "train_ids": [k for k, v in Counter(train_ids).items() if v > 1],
        "test_ids": [k for k, v in Counter(test_ids).items() if v > 1],
        "sample_ids": [k for k, v in Counter(sample_ids).items() if v > 1],
    }
    by_language = Counter(row["language"] for row in train)
    by_split = Counter(row["original_split"] for row in train)
    language_split = defaultdict(Counter)
    for row in train:
        language_split[row["language"]][row["original_split"]] += 1

    text_by_lang = defaultdict(list)
    text_by_lang_split = defaultdict(list)
    for row in train:
        text_by_lang[row["language"]].append(row["transcription"])
        text_by_lang_split[(row["language"], row["original_split"])].append(row["transcription"])

    report = {
        "raw_dir": str(raw_dir),
        "files": {
            "train_rows": len(train),
            "test_rows": len(test),
            "sample_submission_rows": len(sample),
            "train_columns": ["id", "transcription", "language", "original_split"],
            "test_columns": ["ID"],
            "sample_submission_columns": ["ID", "Target"],
        },
        "missing_values": missing_values,
        "duplicate_ids": {k: {"count": len(v), "examples": v[:20]} for k, v in duplicates.items()},
        "train_test_id_overlap": {
            "count": len(set(train_ids) & set(test_ids)),
            "examples": sorted(set(train_ids) & set(test_ids))[:20],
        },
        "sample_submission_alignment": {
            "same_ids_same_order": test_ids == sample_ids,
            "test_minus_sample": sorted(set(test_ids) - set(sample_ids))[:20],
            "sample_minus_test": sorted(set(sample_ids) - set(test_ids))[:20],
        },
        "language_distribution": dict(sorted(by_language.items())),
        "split_distribution": dict(sorted(by_split.items())),
        "language_split_distribution": {
            lang: dict(sorted(counts.items())) for lang, counts in sorted(language_split.items())
        },
        "test_prefix_distribution": dict(sorted(Counter(id_language(x) for x in test_ids).items())),
        "transcription_length_stats": {
            lang: length_stats(texts) for lang, texts in sorted(text_by_lang.items())
        },
        "transcription_length_stats_by_split": {
            f"{lang}_{split}": length_stats(texts)
            for (lang, split), texts in sorted(text_by_lang_split.items())
        },
        "text_quirks": {
            "leading_trailing_whitespace_rows": sum(
                1 for row in train if row["transcription"] != row["transcription"].strip()
            ),
            "newline_rows": sum(1 for row in train if "\n" in row["transcription"] or "\r" in row["transcription"]),
            "uppercase_rows": sum(1 for row in train if any(c.isupper() for c in row["transcription"])),
            "apostrophe_rows": sum(1 for row in train if "'" in row["transcription"] or "’" in row["transcription"]),
            "non_ascii_rows": sum(1 for row in train if any(ord(c) > 127 for c in row["transcription"])),
            "digit_rows": sum(1 for row in train if any(c.isdigit() for c in row["transcription"])),
            "empty_after_strip_rows": sum(1 for row in train if not row["transcription"].strip()),
        },
        "character_inventory": char_inventory(train),
        "examples": row_examples(train),
    }
    return report


def print_summary(report: dict) -> None:
    """Print a compact human-readable audit summary."""
    print(json.dumps(report["files"], indent=2))
    print("\nLanguage distribution:", report["language_distribution"])
    print("Split distribution:", report["split_distribution"])
    print("Language x split:", report["language_split_distribution"])
    print("Test prefix distribution:", report["test_prefix_distribution"])
    print("Missing values:", report["missing_values"])
    print("Duplicate ID counts:", {k: v["count"] for k, v in report["duplicate_ids"].items()})
    print("Train/test overlap:", report["train_test_id_overlap"]["count"])
    print("Sample alignment:", report["sample_submission_alignment"])
    print("Text quirks:", report["text_quirks"])
    print("\nLength stats by language:")
    for lang, stats in report["transcription_length_stats"].items():
        print(f"  {lang}: {stats}")
    print("\nCharacter specials by language:")
    for lang, inv in report["character_inventory"].items():
        print(f"  {lang}: unique={inv['num_unique']} specials={inv['specials']!r} non_ascii={inv['non_ascii']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    parser.add_argument("--output", type=Path, default=Path("outputs/data_audit.json"))
    args = parser.parse_args()

    report = build_report(args.raw_dir)
    print_summary(report)
    json_dump(report, args.output)
    print(f"\nSaved audit report to {args.output}")


if __name__ == "__main__":
    main()

