#!/usr/bin/env python3
"""Analyze ASR prediction distribution sanity across checkpoints/splits."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import id_language, read_prediction_csv, references_from_validation_csv  # noqa: E402
from waxal.scoring import score_records  # noqa: E402
from waxal.text_normalization import POLICIES, normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


SAFE_TEXT_RE = re.compile(r"^[\w\s'.,!?;:()\\/\-]+$", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--normalization", choices=[*POLICIES, "all"], default="no_punct_lower")
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/prediction_distribution_analysis.json"))
    parser.add_argument("--short-word-threshold", type=int, default=3)
    parser.add_argument("--short-char-threshold", type=int, default=12)
    parser.add_argument("--long-word-threshold", type=int, default=90)
    parser.add_argument("--long-char-threshold", type=int, default=650)
    parser.add_argument("--max-examples", type=int, default=25)
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
    values = sorted(values)

    def q(prob: float) -> float:
        if len(values) == 1:
            return float(values[0])
        pos = prob * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(values[lo])
        return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))

    return {
        "min": float(values[0]),
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": float(values[-1]),
        "mean": float(mean(values)),
    }


def max_repeated_run(words: list[str]) -> int:
    best = 0
    current = 0
    previous = None
    for word in words:
        if word and word == previous:
            current += 1
        else:
            previous = word
            current = 1 if word else 0
        best = max(best, current)
    return best


def has_repeated_ngram(words: list[str], n: int = 3) -> bool:
    if len(words) < 2 * n:
        return False
    for idx in range(len(words) - 2 * n + 1):
        if words[idx : idx + n] == words[idx + n : idx + 2 * n]:
            return True
    return False


def row_stats(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    text = normalize_text(row.get("Target", ""), "raw")
    words = text.split()
    unusual_chars = sorted({char for char in text if not SAFE_TEXT_RE.match(char)})
    return {
        "ID": row["ID"],
        "language": row.get("language") or row.get("Language") or id_language(row["ID"]),
        "text": text,
        "chars": len(text),
        "words": len(words),
        "empty": len(text) == 0,
        "very_short": len(words) < args.short_word_threshold or len(text) < args.short_char_threshold,
        "very_long": len(words) > args.long_word_threshold or len(text) > args.long_char_threshold,
        "max_repeated_run": max_repeated_run([word.lower() for word in words]),
        "repeated_ngram": has_repeated_ngram([word.lower() for word in words]),
        "unusual_chars": unusual_chars,
    }


def summarize_group(stats_rows: list[dict[str, Any]]) -> dict[str, Any]:
    unusual_counter: Counter[str] = Counter()
    for row in stats_rows:
        unusual_counter.update(row["unusual_chars"])
    return {
        "num_examples": len(stats_rows),
        "char_length": quantiles([row["chars"] for row in stats_rows]),
        "word_length": quantiles([row["words"] for row in stats_rows]),
        "empty_count": sum(row["empty"] for row in stats_rows),
        "very_short_count": sum(row["very_short"] for row in stats_rows),
        "very_long_count": sum(row["very_long"] for row in stats_rows),
        "repeated_run_ge3_count": sum(row["max_repeated_run"] >= 3 for row in stats_rows),
        "repeated_ngram_count": sum(row["repeated_ngram"] for row in stats_rows),
        "unusual_char_counts": dict(sorted(unusual_counter.items())),
    }


def analyze_prediction_file(path: Path, name: str, args: argparse.Namespace) -> dict[str, Any]:
    rows = read_prediction_csv(path)
    stats_rows = [row_stats(row, args) for row in rows]
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stats_rows:
        by_language[row["language"]].append(row)

    examples = {
        "empty": [row["ID"] for row in stats_rows if row["empty"]][: args.max_examples],
        "very_short": [row["ID"] for row in stats_rows if row["very_short"]][: args.max_examples],
        "very_long": [row["ID"] for row in stats_rows if row["very_long"]][: args.max_examples],
        "repeated": [row["ID"] for row in stats_rows if row["max_repeated_run"] >= 3 or row["repeated_ngram"]][: args.max_examples],
        "unusual_chars": [
            {"ID": row["ID"], "language": row["language"], "chars": row["unusual_chars"], "text": row["text"][:220]}
            for row in stats_rows
            if row["unusual_chars"]
        ][: args.max_examples],
    }

    return {
        "name": name,
        "path": str(path),
        "overall": summarize_group(stats_rows),
        "by_language": {lang: summarize_group(group) for lang, group in sorted(by_language.items())},
        "examples": examples,
    }


def compare_pair(path_a: Path, name_a: str, path_b: Path, name_b: str, args: argparse.Namespace) -> dict[str, Any]:
    rows_a = {row["ID"]: normalize_text(row.get("Target", ""), "raw") for row in read_prediction_csv(path_a)}
    rows_b = {row["ID"]: normalize_text(row.get("Target", ""), "raw") for row in read_prediction_csv(path_b)}
    common = sorted(set(rows_a) & set(rows_b))
    changed = []
    strong = []
    for example_id in common:
        a = rows_a[example_id]
        b = rows_b[example_id]
        if a == b:
            continue
        len_a = len(a)
        len_b = len(b)
        ratio = (len_b + 1) / (len_a + 1)
        record = {
            "ID": example_id,
            "language": id_language(example_id),
            "chars_a": len_a,
            "chars_b": len_b,
            "length_ratio_b_over_a": ratio,
            "a": a[:220],
            "b": b[:220],
        }
        changed.append(record)
        if ratio < 0.65 or ratio > 1.55 or abs(len_b - len_a) > 160:
            strong.append(record)

    by_language = defaultdict(lambda: {"changed": 0, "strong": 0, "total": 0})
    for example_id in common:
        by_language[id_language(example_id)]["total"] += 1
    for row in changed:
        by_language[row["language"]]["changed"] += 1
    for row in strong:
        by_language[row["language"]]["strong"] += 1

    return {
        "a": name_a,
        "b": name_b,
        "common_ids": len(common),
        "changed_count": len(changed),
        "strong_difference_count": len(strong),
        "by_language": dict(sorted((lang, dict(values)) for lang, values in by_language.items())),
        "strong_examples": strong[: args.max_examples],
    }


def main() -> None:
    args = parse_args()
    if args.names and len(args.names) != len(args.predictions):
        raise ValueError("--names must have the same number of values as --predictions")
    names = args.names or [path.stem for path in args.predictions]

    payload: dict[str, Any] = {
        "predictions": [analyze_prediction_file(path, name, args) for path, name in zip(args.predictions, names, strict=True)],
        "pairwise": [],
    }

    if len(args.predictions) > 1:
        for idx in range(len(args.predictions) - 1):
            payload["pairwise"].append(
                compare_pair(args.predictions[idx], names[idx], args.predictions[idx + 1], names[idx + 1], args)
            )

    if args.references:
        references = references_from_validation_csv(args.references)
        policies = POLICIES if args.normalization == "all" else (args.normalization,)
        payload["validation_metrics"] = {}
        for path, name in zip(args.predictions, names, strict=True):
            predictions = read_prediction_csv(path)
            payload["validation_metrics"][name] = {
                policy: score_records(references, predictions, normalization=policy)
                for policy in policies
            }

    json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
