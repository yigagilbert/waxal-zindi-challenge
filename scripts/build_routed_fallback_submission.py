#!/usr/bin/env python3
"""Build and evaluate routed fallback ASR predictions.

This script is intentionally conservative: it only replaces predictions that
match explicit failure patterns, and it keeps original WAXAL IDs/order intact.
Use it first on validation with references, then on test only if the validation
simulation and sanity report look better.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import (  # noqa: E402
    default_raw_dir,
    id_language,
    load_sample_submission,
    read_csv_dicts,
    read_prediction_csv,
    references_from_validation_csv,
    write_csv_rows,
)
from waxal.scoring import compute_group_metrics, score_records  # noqa: E402
from waxal.text_normalization import POLICIES  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


PUNCT_ONLY_RE = re.compile(r"^[\s\.,!?;:'\"()\[\]{}\-_/\\|]+$")
WORD_RE = re.compile(r"\S+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/analysis/routed_fallback_report.json"))
    parser.add_argument("--sample-order", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--alvin-lingala", type=Path, default=None)
    parser.add_argument("--shona-fallback", type=Path, default=None)
    parser.add_argument(
        "--comparison-predictions",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Extra prediction file to score/compare on validation. May be repeated.",
    )
    parser.add_argument("--normalization", choices=POLICIES, default="language_safe")
    parser.add_argument("--lingala-short-chars", type=int, default=12)
    parser.add_argument("--lingala-short-words", type=int, default=2)
    parser.add_argument("--shona-short-chars", type=int, default=12)
    parser.add_argument("--shona-short-words", type=int, default=2)
    parser.add_argument("--min-reference-char-ratio", type=float, default=0.20)
    parser.add_argument("--repeated-run-threshold", type=int, default=3)
    parser.add_argument("--char-collapse-threshold", type=int, default=12)
    parser.add_argument("--max-examples", type=int, default=40)
    return parser.parse_args()


def load_order(path: Path | None, raw_dir: Path | None, base_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if path is not None:
        columns, rows, bad = read_csv_dicts(path)
        if bad:
            raise ValueError(f"{path} has rows with extra fields: {bad[:3]}")
        if "ID" in columns:
            return [{"ID": row["ID"], "language": row.get("language") or row.get("Language") or id_language(row["ID"])} for row in rows]
        if "id" in columns:
            return [{"ID": row["id"], "language": row.get("language") or id_language(row["id"])} for row in rows]
        raise ValueError(f"{path} must contain ID or id column; got {columns}")

    if raw_dir is not None:
        return [{"ID": row["ID"], "language": id_language(row["ID"])} for row in load_sample_submission(raw_dir)]

    default_raw = default_raw_dir()
    sample = default_raw / "SampleSubmission.csv"
    if sample.exists():
        return [{"ID": row["ID"], "language": id_language(row["ID"])} for row in load_sample_submission(default_raw)]

    return [{"ID": row["ID"], "language": row.get("language") or id_language(row["ID"])} for row in base_rows]


def parse_named_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--comparison-predictions must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty comparison name in {value!r}")
    return name, Path(path)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.strip())


def max_repeated_run(tokens: list[str]) -> int:
    best = 0
    current = 0
    previous = None
    for token in tokens:
        token = token.lower()
        if token and token == previous:
            current += 1
        else:
            current = 1 if token else 0
            previous = token
        best = max(best, current)
    return best


def max_char_run(text: str) -> int:
    best = 0
    current = 0
    previous = None
    for char in text:
        if char.isspace():
            previous = None
            current = 0
            continue
        if char == previous:
            current += 1
        else:
            current = 1
            previous = char
        best = max(best, current)
    return best


def text_flags(
    text: str,
    language: str,
    args: argparse.Namespace,
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    raw = "" if text is None else text
    stripped = raw.strip()
    token_list = words(stripped)
    char_count = len(stripped)
    word_count = len(token_list)
    dot_only = bool(stripped) and set(stripped) <= {"."}
    punctuation_only = bool(stripped) and PUNCT_ONLY_RE.fullmatch(stripped) is not None
    repeated_run = max_repeated_run(token_list)
    char_run = max_char_run(stripped)
    reference_char_ratio = None
    if reference is not None:
        ref_len = len(reference.strip())
        reference_char_ratio = (char_count / ref_len) if ref_len else None

    flags = {
        "empty": char_count == 0,
        "dot_only": dot_only,
        "length_le_3": char_count <= 3,
        "punctuation_only": punctuation_only,
        "repeated_token_run_ge3": repeated_run >= args.repeated_run_threshold,
        "ctc_char_collapse": char_run >= args.char_collapse_threshold,
        "very_short_lingala": False,
        "very_short_shona": False,
        "very_short_vs_reference": False,
    }
    if language == "lin":
        flags["very_short_lingala"] = char_count < args.lingala_short_chars or word_count <= args.lingala_short_words
    if language == "sna":
        flags["very_short_shona"] = char_count < args.shona_short_chars or word_count <= args.shona_short_words
    if reference_char_ratio is not None and language in {"lin", "sna"}:
        flags["very_short_vs_reference"] = reference_char_ratio < args.min_reference_char_ratio

    severe = any(
        flags[name]
        for name in (
            "empty",
            "dot_only",
            "length_le_3",
            "punctuation_only",
            "repeated_token_run_ge3",
            "ctc_char_collapse",
        )
    )
    language_short = bool(flags["very_short_lingala"] or flags["very_short_shona"] or flags["very_short_vs_reference"])
    return {
        "chars": char_count,
        "words": word_count,
        "max_repeated_run": repeated_run,
        "max_char_run": char_run,
        "reference_char_ratio": reference_char_ratio,
        "flags": flags,
        "is_bad": severe or language_short,
        "is_severe_bad": severe,
        "bad_reasons": [name for name, value in flags.items() if value],
    }


def prediction_by_id(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {row["ID"]: row.get("Target", "") for row in read_prediction_csv(path)}


def refs_by_id(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    return {row["ID"]: row for row in references_from_validation_csv(path)}


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "p95": None, "max": None}
    values = sorted(values)

    def q(prob: float) -> float:
        if len(values) == 1:
            return float(values[0])
        pos = prob * (len(values) - 1)
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return float(values[low])
        return float(values[low] * (high - pos) + values[high] * (pos - low))

    return {
        "min": float(values[0]),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "p95": q(0.95),
        "max": float(values[-1]),
    }


def summarize_rows(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    references: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    references = references or {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        example_id = row["ID"]
        language = row.get("language") or id_language(example_id)
        reference = references.get(example_id, {}).get("Target")
        stats = text_flags(row.get("Target", ""), language, args, reference=reference)
        grouped[language].append({"ID": example_id, **stats})

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        reason_counts: Counter[str] = Counter()
        for item in group:
            reason_counts.update(item["bad_reasons"])
        return {
            "num_examples": len(group),
            "empty_count": sum(item["flags"]["empty"] for item in group),
            "dot_only_count": sum(item["flags"]["dot_only"] for item in group),
            "length_le_3_count": sum(item["flags"]["length_le_3"] for item in group),
            "punctuation_only_count": sum(item["flags"]["punctuation_only"] for item in group),
            "very_short_count": sum(item["is_bad"] for item in group),
            "severe_bad_count": sum(item["is_severe_bad"] for item in group),
            "repeated_token_run_ge3_count": sum(item["flags"]["repeated_token_run_ge3"] for item in group),
            "ctc_char_collapse_count": sum(item["flags"]["ctc_char_collapse"] for item in group),
            "char_length": quantiles([item["chars"] for item in group]),
            "word_length": quantiles([item["words"] for item in group]),
            "bad_reason_counts": dict(sorted(reason_counts.items())),
        }

    all_items = [item for group in grouped.values() for item in group]
    return {
        "overall": summarize_group(all_items),
        "by_language": {lang: summarize_group(group) for lang, group in sorted(grouped.items())},
    }


def candidate_is_usable(
    candidate: str | None,
    language: str,
    args: argparse.Namespace,
    reference: str | None,
) -> tuple[bool, dict[str, Any]]:
    if candidate is None:
        return False, {"bad_reasons": ["missing_candidate"], "is_bad": True, "is_severe_bad": True}
    stats = text_flags(candidate, language, args, reference=reference)
    return not stats["is_bad"], stats


def build_routed_rows(
    order_rows: list[dict[str, str]],
    base_by_id: dict[str, str],
    alvin_by_id: dict[str, str],
    shona_by_id: dict[str, str],
    references: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    output_rows: list[dict[str, str]] = []
    action_counts: Counter[str] = Counter()
    action_counts_by_language: dict[str, Counter[str]] = defaultdict(Counter)
    replaced_examples: list[dict[str, Any]] = []
    rejected_examples: list[dict[str, Any]] = []
    missing_base: list[str] = []

    for order_row in order_rows:
        example_id = order_row["ID"]
        language = order_row.get("language") or id_language(example_id)
        reference = references.get(example_id, {}).get("Target")
        base_text = base_by_id.get(example_id, "")
        if example_id not in base_by_id:
            missing_base.append(example_id)

        base_stats = text_flags(base_text, language, args, reference=reference)
        selected_text = base_text
        action = "keep_base"
        candidate_text: str | None = None
        candidate_name = ""
        candidate_stats: dict[str, Any] | None = None

        if language == "lin" and base_stats["is_bad"] and alvin_by_id:
            candidate_name = "alvin_lingala"
            candidate_text = alvin_by_id.get(example_id)
            usable, candidate_stats = candidate_is_usable(candidate_text, language, args, reference)
            if usable:
                selected_text = candidate_text or ""
                action = "replace_lingala_with_alvin"
            else:
                action = "keep_base_alvin_unusable"
        elif language == "sna" and base_stats["is_bad"] and shona_by_id:
            candidate_name = "shona_fallback"
            candidate_text = shona_by_id.get(example_id)
            usable, candidate_stats = candidate_is_usable(candidate_text, language, args, reference)
            if usable:
                selected_text = candidate_text or ""
                action = "replace_shona_with_fallback"
            else:
                action = "keep_base_shona_fallback_unusable"
        elif base_stats["is_bad"]:
            action = f"keep_base_bad_{language}"

        action_counts[action] += 1
        action_counts_by_language[language][action] += 1

        if action.startswith("replace_"):
            replaced_examples.append(
                {
                    "ID": example_id,
                    "language": language,
                    "action": action,
                    "base_reasons": base_stats["bad_reasons"],
                    "base": base_text[:260],
                    "candidate_name": candidate_name,
                    "candidate": selected_text[:260],
                }
            )
        elif action.startswith("keep_base_") and action != "keep_base":
            rejected_examples.append(
                {
                    "ID": example_id,
                    "language": language,
                    "action": action,
                    "base_reasons": base_stats["bad_reasons"],
                    "candidate_name": candidate_name,
                    "candidate_reasons": (candidate_stats or {}).get("bad_reasons", []),
                    "base": base_text[:260],
                    "candidate": (candidate_text or "")[:260],
                }
            )

        output_rows.append({"ID": example_id, "Target": selected_text.strip() or "."})

    diagnostics = {
        "action_counts": dict(sorted(action_counts.items())),
        "action_counts_by_language": {
            lang: dict(sorted(counter.items())) for lang, counter in sorted(action_counts_by_language.items())
        },
        "missing_base_predictions": missing_base,
        "replaced_examples": replaced_examples[: args.max_examples],
        "rejected_examples": rejected_examples[: args.max_examples],
        "num_replaced": len(replaced_examples),
        "num_rejected_bad_base": len(rejected_examples),
    }
    return output_rows, diagnostics


def per_example_combined(ref: str, pred: str, normalization: str) -> float:
    metrics = compute_group_metrics([ref], [pred], normalization=normalization)
    return float(metrics["combined"])


def compare_with_references(
    base_by_id: dict[str, str],
    comparisons: dict[str, dict[str, str]],
    references: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, comp_by_id in comparisons.items():
        worse: list[dict[str, Any]] = []
        better: list[dict[str, Any]] = []
        for example_id, ref_row in references.items():
            if example_id not in base_by_id or example_id not in comp_by_id:
                continue
            ref = ref_row["Target"]
            base_text = base_by_id[example_id]
            comp_text = comp_by_id[example_id]
            base_score = per_example_combined(ref, base_text, args.normalization)
            comp_score = per_example_combined(ref, comp_text, args.normalization)
            delta = base_score - comp_score
            record = {
                "ID": example_id,
                "language": ref_row.get("language") or id_language(example_id),
                "base_combined": base_score,
                "comparison_combined": comp_score,
                "base_minus_comparison": delta,
                "reference": ref[:220],
                "base": base_text[:220],
                "comparison": comp_text[:220],
            }
            if delta > 0.05:
                worse.append(record)
            elif delta < -0.05:
                better.append(record)
        worse.sort(key=lambda row: row["base_minus_comparison"], reverse=True)
        better.sort(key=lambda row: row["base_minus_comparison"])
        output[name] = {
            "ids_where_base_is_worse_count": len(worse),
            "ids_where_base_is_better_count": len(better),
            "base_worse_examples": worse[: args.max_examples],
            "base_better_examples": better[: args.max_examples],
        }
    return output


def validate_output(
    rows: list[dict[str, str]],
    order_rows: list[dict[str, str]],
    base_rows: list[dict[str, str]],
) -> dict[str, Any]:
    ids = [row["ID"] for row in rows]
    order_ids = [row["ID"] for row in order_rows]
    duplicate_ids = sorted(example_id for example_id, count in Counter(ids).items() if count > 1)
    missing_ids = sorted(set(order_ids) - set(ids))
    extra_ids = sorted(set(ids) - set(order_ids))
    empty_targets = [row["ID"] for row in rows if not row.get("Target", "").strip()]
    aligned = ids == order_ids
    return {
        "row_count": len(rows),
        "expected_row_count": len(order_rows),
        "base_row_count": len(base_rows),
        "aligned_to_order": aligned,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "empty_targets": empty_targets,
    }


def main() -> None:
    args = parse_args()
    base_rows = read_prediction_csv(args.base_predictions)
    base_by_id = {row["ID"]: row.get("Target", "") for row in base_rows}
    order_rows = load_order(args.sample_order, args.raw_dir, base_rows)
    references = refs_by_id(args.references)
    reference_rows = list(references.values())

    alvin_by_id = prediction_by_id(args.alvin_lingala)
    shona_by_id = prediction_by_id(args.shona_fallback)
    comparison_paths = dict(parse_named_prediction(value) for value in args.comparison_predictions)
    comparison_predictions = {name: prediction_by_id(path) for name, path in comparison_paths.items()}
    if args.shona_fallback is not None and "shona_fallback" not in comparison_predictions:
        comparison_predictions["shona_fallback"] = shona_by_id

    routed_rows, routing_diagnostics = build_routed_rows(
        order_rows,
        base_by_id,
        alvin_by_id,
        shona_by_id,
        references,
        args,
    )
    write_csv_rows(args.output, routed_rows, ["ID", "Target"])

    base_ordered_rows = [{"ID": row["ID"], "Target": base_by_id.get(row["ID"], "")} for row in order_rows]
    report: dict[str, Any] = {
        "inputs": {
            "base_predictions": str(args.base_predictions),
            "sample_order": str(args.sample_order) if args.sample_order else None,
            "raw_dir": str(args.raw_dir) if args.raw_dir else None,
            "references": str(args.references) if args.references else None,
            "alvin_lingala": str(args.alvin_lingala) if args.alvin_lingala else None,
            "shona_fallback": str(args.shona_fallback) if args.shona_fallback else None,
            "comparison_predictions": {name: str(path) for name, path in comparison_paths.items()},
        },
        "thresholds": {
            "lingala_short_chars": args.lingala_short_chars,
            "lingala_short_words": args.lingala_short_words,
            "shona_short_chars": args.shona_short_chars,
            "shona_short_words": args.shona_short_words,
            "min_reference_char_ratio": args.min_reference_char_ratio,
            "repeated_run_threshold": args.repeated_run_threshold,
            "char_collapse_threshold": args.char_collapse_threshold,
        },
        "validation": validate_output(routed_rows, order_rows, base_rows),
        "routing": routing_diagnostics,
        "sanity_before": summarize_rows(base_ordered_rows, args, references),
        "sanity_after": summarize_rows(routed_rows, args, references),
    }

    if reference_rows:
        report["validation_metrics"] = {
            "base": score_records(reference_rows, base_ordered_rows, normalization=args.normalization),
            "routed_fallback": score_records(reference_rows, routed_rows, normalization=args.normalization),
        }
        for name, comp_by_id in comparison_predictions.items():
            comp_rows = [{"ID": row["ID"], "Target": comp_by_id.get(row["ID"], "")} for row in order_rows]
            report["validation_metrics"][name] = score_records(reference_rows, comp_rows, normalization=args.normalization)
        if alvin_by_id:
            lin_refs = [row for row in reference_rows if (row.get("language") or id_language(row["ID"])) == "lin"]
            lin_alvin_rows = [{"ID": row["ID"], "Target": alvin_by_id.get(row["ID"], "")} for row in lin_refs]
            report["validation_metrics"]["alvin_lingala_only_on_lingala"] = score_records(
                lin_refs,
                lin_alvin_rows,
                normalization=args.normalization,
            )
        report["base_vs_comparisons"] = compare_with_references(base_by_id, comparison_predictions, references, args)

    json_dump(report, args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote routed predictions to {args.output}")
    print(f"Wrote routing report to {args.report}")


if __name__ == "__main__":
    main()
