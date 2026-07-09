#!/usr/bin/env python3
"""Validation-driven ASR model-zoo comparison and routed submission builder.

This script is for post-processing and routing only. It does not train models.
It compares available validation predictions, estimates oracle routing ceilings,
then applies the same conservative-but-less-timid fallback rules to test.
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
    read_prediction_csv,
    references_from_validation_csv,
    write_csv_rows,
)
from waxal.scoring import compute_group_metrics, edit_distance, safe_rate, score_records  # noqa: E402
from waxal.text_normalization import POLICIES, normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


PUNCT_ONLY_RE = re.compile(r"^[\s\.,!?;:'\"()\[\]{}\-_/\\|]+$")
WORD_RE = re.compile(r"\S+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument(
        "--validation-prediction",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Validation prediction CSV. Repeat for each model.",
    )
    parser.add_argument(
        "--test-prediction",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Test prediction CSV for routed submission. Repeat for each model.",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    parser.add_argument("--normalization", choices=POLICIES, default="language_safe")
    parser.add_argument("--model-zoo-output", type=Path, default=Path("outputs/analysis/model_zoo_validation_comparison.json"))
    parser.add_argument("--oracle-output", type=Path, default=Path("outputs/analysis/oracle_routing_validation.json"))
    parser.add_argument("--sanity-output", type=Path, default=Path("outputs/analysis/fallback_before_after_sanity.json"))
    parser.add_argument("--submission-output", type=Path, default=Path("outputs/submissions/submission_model_zoo_routed_v1.csv"))
    parser.add_argument(
        "--fallback-priority",
        action="append",
        default=[],
        metavar="LANG=MODEL1,MODEL2",
        help="Fallback priority for production routing, e.g. lin=alvin,noirlab.",
    )
    parser.add_argument("--min-words", type=int, default=2, help="Bad if prediction has fewer words than this.")
    parser.add_argument("--short-char-threshold", type=int, default=4)
    parser.add_argument("--repeated-run-threshold", type=int, default=3)
    parser.add_argument("--char-collapse-threshold", type=int, default=12)
    parser.add_argument("--low-reference-ratio", type=float, default=0.20)
    parser.add_argument("--low-median-ratio", type=float, default=0.15)
    parser.add_argument("--min-route-gain", type=float, default=0.01)
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Empty name in {value!r}")
    return name, Path(path)


def parse_priority(values: list[str]) -> dict[str, list[str]]:
    priorities: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LANG=MODEL1,MODEL2, got {value!r}")
        lang, names = value.split("=", 1)
        lang = lang.strip()
        priorities[lang] = [name.strip() for name in names.split(",") if name.strip()]
    return priorities


def load_predictions(named_values: list[str]) -> dict[str, list[dict[str, str]]]:
    payload = {}
    for value in named_values:
        name, path = parse_named_path(value)
        if not path.exists():
            raise FileNotFoundError(f"Prediction file for {name!r} not found: {path}")
        payload[name] = read_prediction_csv(path)
    return payload


def by_id(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["ID"]: row.get("Target", "") for row in rows}


def ref_language(row: dict[str, str]) -> str:
    return row.get("language") or row.get("Language") or id_language(row["ID"])


def token_words(text: str) -> list[str]:
    return WORD_RE.findall((text or "").strip())


def max_repeated_run(tokens: list[str]) -> int:
    best = 0
    current = 0
    previous = None
    for token in tokens:
        token = token.lower()
        if token and token == previous:
            current += 1
        else:
            previous = token
            current = 1 if token else 0
        best = max(best, current)
    return best


def has_repeated_ngram(tokens: list[str], n: int = 3) -> bool:
    lowered = [token.lower() for token in tokens if token]
    if len(lowered) < 2 * n:
        return False
    for idx in range(len(lowered) - 2 * n + 1):
        if lowered[idx : idx + n] == lowered[idx + n : idx + 2 * n]:
            return True
    return False


def max_char_run(text: str) -> int:
    best = 0
    current = 0
    previous = None
    for char in text or "":
        if char.isspace():
            current = 0
            previous = None
            continue
        if char == previous:
            current += 1
        else:
            current = 1
            previous = char
        best = max(best, current)
    return best


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "p95": None, "max": None}
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
        "median": float(median(values)),
        "mean": float(mean(values)),
        "p95": q(0.95),
        "max": float(values[-1]),
    }


def language_medians(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[id_language(row["ID"])].append(len((row.get("Target") or "").strip()))
    return {lang: float(median(values)) for lang, values in grouped.items() if values}


def prediction_flags(
    text: str,
    language: str,
    args: argparse.Namespace,
    *,
    reference: str | None = None,
    median_chars: float | None = None,
) -> dict[str, Any]:
    stripped = (text or "").strip()
    tokens = token_words(stripped)
    char_count = len(stripped)
    word_count = len(tokens)
    reference_ratio = None
    if reference is not None:
        ref_len = len(reference.strip())
        reference_ratio = char_count / ref_len if ref_len else None
    median_ratio = None
    if median_chars:
        median_ratio = char_count / median_chars

    flags = {
        "empty": char_count == 0,
        "dot_only": bool(stripped) and set(stripped) <= {"."},
        "length_le_threshold": char_count <= args.short_char_threshold,
        "fewer_than_min_words": word_count < args.min_words,
        "punctuation_only": bool(stripped) and PUNCT_ONLY_RE.fullmatch(stripped) is not None,
        "repeated_token_collapse": max_repeated_run(tokens) >= args.repeated_run_threshold,
        "repeated_ngram": has_repeated_ngram(tokens),
        "ctc_char_collapse": max_char_run(stripped) >= args.char_collapse_threshold,
        "low_reference_ratio": reference_ratio is not None and reference_ratio < args.low_reference_ratio,
        "low_language_median_ratio": median_ratio is not None and median_ratio < args.low_median_ratio,
    }
    is_bad = any(flags.values())
    return {
        "chars": char_count,
        "words": word_count,
        "reference_ratio": reference_ratio,
        "median_ratio": median_ratio,
        "flags": flags,
        "bad_reasons": [name for name, value in flags.items() if value],
        "is_bad": is_bad,
        "is_healthy": not is_bad,
    }


def summarize_predictions(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    *,
    references: dict[str, dict[str, str]] | None = None,
    median_by_language: dict[str, float] | None = None,
) -> dict[str, Any]:
    references = references or {}
    median_by_language = median_by_language or language_medians(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        example_id = row["ID"]
        language = id_language(example_id)
        stats = prediction_flags(
            row.get("Target", ""),
            language,
            args,
            reference=references.get(example_id, {}).get("Target"),
            median_chars=median_by_language.get(language),
        )
        grouped[language].append(stats)

    def group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        reason_counts: Counter[str] = Counter()
        for item in items:
            reason_counts.update(item["bad_reasons"])
        return {
            "num_examples": len(items),
            "empty_count": sum(item["flags"]["empty"] for item in items),
            "dot_only_count": sum(item["flags"]["dot_only"] for item in items),
            "very_short_count": sum(item["is_bad"] for item in items),
            "repeated_token_count": sum(item["flags"]["repeated_token_collapse"] for item in items),
            "repeated_ngram_count": sum(item["flags"]["repeated_ngram"] for item in items),
            "ctc_char_collapse_count": sum(item["flags"]["ctc_char_collapse"] for item in items),
            "mean_chars": mean([item["chars"] for item in items]) if items else 0.0,
            "median_chars": median([item["chars"] for item in items]) if items else 0.0,
            "char_length": quantiles([item["chars"] for item in items]),
            "word_length": quantiles([item["words"] for item in items]),
            "bad_reason_counts": dict(sorted(reason_counts.items())),
        }

    all_items = [item for values in grouped.values() for item in values]
    return {
        "overall": group_summary(all_items),
        "by_language": {lang: group_summary(items) for lang, items in sorted(grouped.items())},
    }


def sequence_distance(ref: list[str], hyp: list[str]) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(ref, hyp))
    except Exception:
        return edit_distance(ref, hyp)


def score_pair(reference: str, prediction: str, normalization: str) -> dict[str, float | int]:
    ref = normalize_text(reference, normalization)
    hyp = normalize_text(prediction, normalization)
    ref_words = ref.split()
    hyp_words = hyp.split()
    word_errors = sequence_distance(ref_words, hyp_words)
    char_errors = sequence_distance(list(ref), list(hyp))
    wer = safe_rate(word_errors, len(ref_words))
    cer = safe_rate(char_errors, len(ref))
    return {
        "word_errors": word_errors,
        "reference_words": len(ref_words),
        "char_errors": char_errors,
        "reference_chars": len(ref),
        "wer": wer,
        "cer": cer,
        "combined": 0.5 * wer + 0.5 * cer,
    }


def precompute_pair_scores(
    refs: list[dict[str, str]],
    predictions_by_name: dict[str, dict[str, str]],
    normalization: str,
) -> dict[tuple[str, str], dict[str, float | int]]:
    scores: dict[tuple[str, str], dict[str, float | int]] = {}
    for ref in refs:
        example_id = ref["ID"]
        reference = ref["Target"]
        for name, pred_map in predictions_by_name.items():
            if example_id in pred_map:
                scores[(name, example_id)] = score_pair(reference, pred_map[example_id], normalization)
    return scores


def aggregate_pair_metrics(
    refs: list[dict[str, str]],
    selections: dict[str, str],
    pair_scores: dict[tuple[str, str], dict[str, float | int]],
    *,
    missing_predictions: list[str] | None = None,
    extra_predictions: list[str] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    overall_items = []
    for ref in refs:
        example_id = ref["ID"]
        model_name = selections.get(example_id)
        if not model_name:
            continue
        score = pair_scores.get((model_name, example_id))
        if score is None:
            continue
        overall_items.append(score)
        grouped[ref_language(ref)].append(score)

    def aggregate(items: list[dict[str, float | int]]) -> dict[str, float | int]:
        word_errors = sum(int(item["word_errors"]) for item in items)
        reference_words = sum(int(item["reference_words"]) for item in items)
        char_errors = sum(int(item["char_errors"]) for item in items)
        reference_chars = sum(int(item["reference_chars"]) for item in items)
        wer = safe_rate(word_errors, reference_words)
        cer = safe_rate(char_errors, reference_chars)
        return {
            "num_examples": len(items),
            "word_errors": word_errors,
            "reference_words": reference_words,
            "char_errors": char_errors,
            "reference_chars": reference_chars,
            "wer": wer,
            "cer": cer,
            "combined": 0.5 * wer + 0.5 * cer,
        }

    by_language = {lang: aggregate(items) for lang, items in sorted(grouped.items())}
    if by_language:
        macro = {
            "wer": sum(float(v["wer"]) for v in by_language.values()) / len(by_language),
            "cer": sum(float(v["cer"]) for v in by_language.values()) / len(by_language),
            "combined": sum(float(v["combined"]) for v in by_language.values()) / len(by_language),
            "num_languages": len(by_language),
        }
    else:
        macro = {"wer": 0.0, "cer": 0.0, "combined": 0.0, "num_languages": 0}
    return {
        "overall_weighted": aggregate(overall_items),
        "macro_by_language": macro,
        "by_language": by_language,
        "missing_predictions": missing_predictions or [],
        "extra_predictions": extra_predictions or [],
    }


def score_rows(name: str, refs: list[dict[str, str]], rows: list[dict[str, str]], normalization: str) -> dict[str, Any]:
    metrics = score_records(refs, rows, normalization=normalization)
    return {
        "name": name,
        "num_prediction_rows": len(rows),
        "coverage": {
            "common_ids": metrics["overall_weighted"]["num_examples"],
            "missing_predictions": len(metrics["missing_predictions"]),
            "extra_predictions": len(metrics["extra_predictions"]),
        },
        "metrics": metrics,
    }


def build_model_zoo_report(
    refs: list[dict[str, str]],
    predictions: dict[str, list[dict[str, str]]],
    args: argparse.Namespace,
    pair_scores: dict[tuple[str, str], dict[str, float | int]],
) -> dict[str, Any]:
    refs_map = {row["ID"]: row for row in refs}
    ref_ids = {row["ID"] for row in refs}
    report: dict[str, Any] = {
        "normalization": args.normalization,
        "models": {},
    }
    for name, rows in sorted(predictions.items()):
        pred_map = by_id(rows)
        pred_ids = set(pred_map)
        common_ids = ref_ids & pred_ids
        selections = {example_id: name for example_id in common_ids}
        metrics = aggregate_pair_metrics(
            refs,
            selections,
            pair_scores,
            missing_predictions=sorted(ref_ids - pred_ids),
            extra_predictions=sorted(pred_ids - ref_ids),
        )
        report["models"][name] = {
            "name": name,
            "num_prediction_rows": len(rows),
            "coverage": {
                "common_ids": metrics["overall_weighted"]["num_examples"],
                "missing_predictions": len(metrics["missing_predictions"]),
                "extra_predictions": len(metrics["extra_predictions"]),
            },
            "metrics": metrics,
            "sanity": summarize_predictions(rows, args, references=refs_map),
        }
    return report


def rows_from_map(order_refs: list[dict[str, str]], pred_map: dict[str, str]) -> list[dict[str, str]]:
    return [{"ID": row["ID"], "Target": pred_map.get(row["ID"], "")} for row in order_refs]


def selections_from_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["ID"]: row["model"] for row in rows if row.get("model")}


def best_model_per_language(
    refs: list[dict[str, str]],
    predictions_by_name: dict[str, dict[str, str]],
    args: argparse.Namespace,
    pair_scores: dict[tuple[str, str], dict[str, float | int]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    languages = sorted({ref_language(row) for row in refs})
    selected: dict[str, str] = {}
    for language in languages:
        lang_refs = [row for row in refs if ref_language(row) == language]
        best_name = None
        best_score = float("inf")
        for name, pred_map in predictions_by_name.items():
            selections = {row["ID"]: name for row in lang_refs if row["ID"] in pred_map}
            if not selections:
                continue
            result = aggregate_pair_metrics(lang_refs, selections, pair_scores)
            score = result["overall_weighted"]["combined"]
            if score < best_score:
                best_score = score
                best_name = name
        if best_name:
            selected[language] = best_name

    routed = []
    for ref in refs:
        language = ref_language(ref)
        model_name = selected.get(language)
        routed.append(
            {
                "ID": ref["ID"],
                "Target": predictions_by_name.get(model_name, {}).get(ref["ID"], ""),
                "model": model_name or "",
            }
        )
    return routed, selected


def best_per_sample(
    refs: list[dict[str, str]],
    predictions_by_name: dict[str, dict[str, str]],
    args: argparse.Namespace,
    *,
    base_name: str | None = None,
    only_languages: set[str] | None = None,
    only_bad_base: bool = False,
    use_cer_only: bool = True,
    pair_scores: dict[tuple[str, str], dict[str, float | int]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    base_map = predictions_by_name.get(base_name or "", {})
    selected_rows = []
    counts: Counter[str] = Counter()
    examples = []
    median_by_language = language_medians(rows_from_map(refs, base_map)) if base_map else {}

    for ref in refs:
        example_id = ref["ID"]
        language = ref_language(ref)
        reference = ref["Target"]
        if only_languages and language not in only_languages:
            selected = base_map.get(example_id, "")
            selected_name = base_name or ""
        else:
            if only_bad_base and base_name:
                base_text = base_map.get(example_id, "")
                base_flags = prediction_flags(
                    base_text,
                    language,
                    args,
                    reference=reference,
                    median_chars=median_by_language.get(language),
                )
                if not base_flags["is_bad"]:
                    selected_rows.append({"ID": example_id, "Target": base_text, "model": base_name or ""})
                    counts[f"keep_{base_name}"] += 1
                    continue
            best_name = None
            best_text = ""
            best_score = float("inf")
            for name, pred_map in predictions_by_name.items():
                if example_id not in pred_map:
                    continue
                pred = pred_map[example_id]
                metrics = (
                    pair_scores.get((name, example_id))
                    if pair_scores is not None
                    else score_pair(reference, pred, args.normalization)
                )
                score = metrics["cer"] if use_cer_only else metrics["combined"]
                if score < best_score:
                    best_score = float(score)
                    best_name = name
                    best_text = pred
            selected = best_text
            selected_name = best_name or ""
            if base_name and selected_name and selected_name != base_name and len(examples) < args.max_examples:
                examples.append(
                    {
                        "ID": example_id,
                        "language": language,
                        "selected": selected_name,
                        "reference": reference[:220],
                        "base": base_map.get(example_id, "")[:220],
                        "selected_prediction": selected[:220],
                    }
                )
        counts[selected_name or "missing"] += 1
        selected_rows.append({"ID": example_id, "Target": selected, "model": selected_name})

    return selected_rows, {"selection_counts": dict(sorted(counts.items())), "examples": examples}


def production_route(
    refs_or_order: list[dict[str, str]],
    predictions_by_name: dict[str, dict[str, str]],
    priorities: dict[str, list[str]],
    base_name: str,
    args: argparse.Namespace,
    *,
    references_by_id: dict[str, dict[str, str]] | None = None,
    median_by_language: dict[str, float] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    references_by_id = references_by_id or {}
    base_map = predictions_by_name[base_name]
    base_rows = rows_from_map(refs_or_order, base_map)
    median_by_language = median_by_language or language_medians(base_rows)
    output = []
    counts: Counter[str] = Counter()
    replacement_examples = []
    rejected_examples = []

    for row in refs_or_order:
        example_id = row["ID"]
        language = row.get("language") or row.get("Language") or id_language(example_id)
        reference = references_by_id.get(example_id, {}).get("Target")
        base_text = base_map.get(example_id, "")
        base_flags = prediction_flags(
            base_text,
            language,
            args,
            reference=reference,
            median_chars=median_by_language.get(language),
        )
        selected = base_text
        selected_model = base_name
        if base_flags["is_bad"]:
            for candidate_name in priorities.get(language, []):
                candidate_map = predictions_by_name.get(candidate_name, {})
                candidate_text = candidate_map.get(example_id)
                if candidate_text is None:
                    continue
                candidate_flags = prediction_flags(
                    candidate_text,
                    language,
                    args,
                    reference=reference,
                    median_chars=median_by_language.get(language),
                )
                if candidate_flags["is_healthy"]:
                    selected = candidate_text
                    selected_model = candidate_name
                    if len(replacement_examples) < args.max_examples:
                        replacement_examples.append(
                            {
                                "ID": example_id,
                                "language": language,
                                "from": base_name,
                                "to": candidate_name,
                                "base_reasons": base_flags["bad_reasons"],
                                "base": base_text[:220],
                                "replacement": candidate_text[:220],
                            }
                        )
                    break
            if selected_model == base_name and len(rejected_examples) < args.max_examples:
                rejected_examples.append(
                    {
                        "ID": example_id,
                        "language": language,
                        "base_reasons": base_flags["bad_reasons"],
                        "base": base_text[:220],
                    }
                )
        counts[f"{language}:{selected_model}"] += 1
        output.append({"ID": example_id, "Target": selected.strip() or ".", "model": selected_model})

    return output, {
        "selection_counts": dict(sorted(counts.items())),
        "replacement_examples": replacement_examples,
        "unreplaced_bad_base_examples": rejected_examples,
    }


def oracle_report(
    refs: list[dict[str, str]],
    validation_predictions: dict[str, list[dict[str, str]]],
    base_name: str,
    priorities: dict[str, list[str]],
    args: argparse.Namespace,
    pair_scores: dict[tuple[str, str], dict[str, float | int]],
) -> dict[str, Any]:
    pred_maps = {name: by_id(rows) for name, rows in validation_predictions.items()}
    refs_map = {row["ID"]: row for row in refs}
    report: dict[str, Any] = {"normalization": args.normalization, "base_model": base_name, "strategies": {}}

    base_selections = {row["ID"]: base_name for row in refs if row["ID"] in pred_maps[base_name]}
    report["strategies"]["base"] = aggregate_pair_metrics(refs, base_selections, pair_scores)

    best_lang_rows, selected_by_lang = best_model_per_language(refs, pred_maps, args, pair_scores)
    report["strategies"]["best_single_model_per_language"] = {
        "selected_by_language": selected_by_lang,
        "metrics": aggregate_pair_metrics(refs, selections_from_rows(best_lang_rows), pair_scores),
    }

    sample_rows, sample_info = best_per_sample(refs, pred_maps, args, use_cer_only=True, pair_scores=pair_scores)
    report["strategies"]["oracle_best_model_per_sample_by_cer"] = {
        **sample_info,
        "metrics": aggregate_pair_metrics(refs, selections_from_rows(sample_rows), pair_scores),
    }

    collapsed_rows, collapsed_info = best_per_sample(
        refs,
        pred_maps,
        args,
        base_name=base_name,
        only_bad_base=True,
        use_cer_only=False,
        pair_scores=pair_scores,
    )
    report["strategies"]["oracle_replace_only_bad_base_outputs"] = {
        **collapsed_info,
        "metrics": aggregate_pair_metrics(refs, selections_from_rows(collapsed_rows), pair_scores),
    }

    for language in ("lin", "sna", "lug"):
        rows, info = best_per_sample(
            refs,
            pred_maps,
            args,
            base_name=base_name,
            only_languages={language},
            use_cer_only=False,
            pair_scores=pair_scores,
        )
        report["strategies"][f"oracle_replace_only_{language}"] = {
            **info,
            "metrics": aggregate_pair_metrics(refs, selections_from_rows(rows), pair_scores),
        }

    production_rows, production_info = production_route(
        refs,
        pred_maps,
        priorities,
        base_name,
        args,
        references_by_id=refs_map,
    )
    production_metrics = aggregate_pair_metrics(refs, selections_from_rows(production_rows), pair_scores)
    base_combined = report["strategies"]["base"]["overall_weighted"]["combined"]
    production_combined = production_metrics["overall_weighted"]["combined"]
    report["strategies"]["production_priority_route"] = {
        **production_info,
        "metrics": production_metrics,
        "absolute_combined_gain_vs_base": base_combined - production_combined,
        "recommend_submit_if_same_behavior_on_test": (base_combined - production_combined) >= args.min_route_gain,
    }
    return report


def read_sample_order(raw_dir: Path) -> list[dict[str, str]]:
    rows = load_sample_submission(raw_dir)
    return [{"ID": row["ID"], "language": id_language(row["ID"])} for row in rows]


def build_test_submission(
    test_predictions: dict[str, list[dict[str, str]]],
    base_name: str,
    priorities: dict[str, list[str]],
    raw_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if base_name not in test_predictions:
        raise ValueError(f"Base model {base_name!r} missing from --test-prediction inputs.")
    pred_maps = {name: by_id(rows) for name, rows in test_predictions.items()}
    sample_order = read_sample_order(raw_dir)
    base_rows = rows_from_map(sample_order, pred_maps[base_name])
    medians = language_medians(base_rows)
    return production_route(
        sample_order,
        pred_maps,
        priorities,
        base_name,
        args,
        references_by_id=None,
        median_by_language=medians,
    )


def main() -> None:
    args = parse_args()
    priorities = parse_priority(args.fallback_priority)
    validation_predictions = load_predictions(args.validation_prediction)
    if args.base_model not in validation_predictions:
        raise ValueError(f"Base model {args.base_model!r} missing from validation predictions.")

    refs = references_from_validation_csv(args.references)
    refs_by_id = {row["ID"]: row for row in refs}
    validation_pred_maps = {name: by_id(rows) for name, rows in validation_predictions.items()}
    pair_scores = precompute_pair_scores(refs, validation_pred_maps, args.normalization)

    model_zoo = build_model_zoo_report(refs, validation_predictions, args, pair_scores)
    json_dump(model_zoo, args.model_zoo_output)

    oracle = oracle_report(refs, validation_predictions, args.base_model, priorities, args, pair_scores)
    json_dump(oracle, args.oracle_output)

    base_validation_rows = validation_predictions[args.base_model]
    production_validation_maps = {name: by_id(rows) for name, rows in validation_predictions.items()}
    production_validation_rows, production_validation_info = production_route(
        refs,
        production_validation_maps,
        priorities,
        args.base_model,
        args,
        references_by_id=refs_by_id,
    )
    sanity: dict[str, Any] = {
        "validation": {
            "before": summarize_predictions(base_validation_rows, args, references=refs_by_id),
            "after": summarize_predictions(production_validation_rows, args, references=refs_by_id),
            "route_info": production_validation_info,
            "metrics_before": aggregate_pair_metrics(
                refs,
                {row["ID"]: args.base_model for row in refs if row["ID"] in validation_pred_maps[args.base_model]},
                pair_scores,
            ),
            "metrics_after": aggregate_pair_metrics(
                refs,
                selections_from_rows(production_validation_rows),
                pair_scores,
            ),
        }
    }

    test_predictions = load_predictions(args.test_prediction) if args.test_prediction else {}
    if test_predictions:
        routed_test_rows, test_route_info = build_test_submission(
            test_predictions,
            args.base_model,
            priorities,
            args.raw_dir,
            args,
        )
        write_csv_rows(args.submission_output, routed_test_rows, ["ID", "Target"])
        base_test_rows = test_predictions[args.base_model]
        sanity["test"] = {
            "before": summarize_predictions(base_test_rows, args),
            "after": summarize_predictions(routed_test_rows, args),
            "route_info": test_route_info,
            "submission_output": str(args.submission_output),
            "row_count": len(routed_test_rows),
            "duplicate_ids": sorted(
                example_id for example_id, count in Counter(row["ID"] for row in routed_test_rows).items() if count > 1
            ),
            "empty_targets": [row["ID"] for row in routed_test_rows if not row.get("Target", "").strip()],
        }

    base_combined = sanity["validation"]["metrics_before"]["overall_weighted"]["combined"]
    routed_combined = sanity["validation"]["metrics_after"]["overall_weighted"]["combined"]
    sanity["recommendation"] = {
        "base_combined": base_combined,
        "routed_combined": routed_combined,
        "absolute_combined_gain": base_combined - routed_combined,
        "submit": (base_combined - routed_combined) >= args.min_route_gain,
        "minimum_required_gain": args.min_route_gain,
    }
    json_dump(sanity, args.sanity_output)

    print(json.dumps(sanity["recommendation"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.model_zoo_output}")
    print(f"Wrote {args.oracle_output}")
    print(f"Wrote {args.sanity_output}")
    if test_predictions:
        print(f"Wrote {args.submission_output}")


if __name__ == "__main__":
    main()
