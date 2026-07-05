#!/usr/bin/env python3
"""Compare Lingala ASR prediction files against WAXAL references or each other."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import id_language, read_prediction_csv, references_from_validation_csv  # noqa: E402
from waxal.scoring import compute_group_metrics, score_records  # noqa: E402
from waxal.text_normalization import POLICIES, normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


SAFE_TEXT_RE = re.compile(r"^[\w\s'.,!?;:()\\/\-]+$", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--audio-quality", type=Path, default=None)
    parser.add_argument("--normalization", choices=[*POLICIES, "all"], default="language_safe")
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/lingala_models/lingala_model_comparison_metrics.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/LINGALA_MODEL_COMPARISON.md"))
    parser.add_argument("--max-examples", type=int, default=25)
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


def text_stats(text: str) -> dict[str, Any]:
    raw = normalize_text(text, "raw")
    words = raw.split()
    unusual_chars = sorted({char for char in raw if not SAFE_TEXT_RE.match(char)})
    lower_words = [word.lower() for word in words]
    repeated_run = 0
    current = 0
    previous = None
    for word in lower_words:
        if word == previous:
            current += 1
        else:
            previous = word
            current = 1
        repeated_run = max(repeated_run, current)
    repeated_ngram = False
    for n in (2, 3, 4):
        for idx in range(len(lower_words) - 2 * n + 1):
            if lower_words[idx : idx + n] == lower_words[idx + n : idx + 2 * n]:
                repeated_ngram = True
                break
        if repeated_ngram:
            break
    stripped = raw.strip()
    return {
        "chars": len(raw),
        "words": len(words),
        "empty": stripped == "",
        "dot_only": stripped in {".", "...", ",", "?", "!", ";", ":"},
        "very_short": len(raw) < 12 or len(words) < 3,
        "very_long": len(raw) > 650 or len(words) > 90,
        "repeated_run": repeated_run,
        "repeated_ngram": repeated_ngram,
        "unusual_chars": unusual_chars,
    }


def summarize_predictions(rows: list[dict[str, str]], refs_by_id: dict[str, str] | None) -> dict[str, Any]:
    rows = [row for row in rows if id_language(row["ID"]) == "lin"]
    stats = [text_stats(row.get("Target", "")) for row in rows]
    chars = [item["chars"] for item in stats]
    words = [item["words"] for item in stats]
    ratios = []
    if refs_by_id:
        for row, item in zip(rows, stats, strict=True):
            ref = normalize_text(refs_by_id.get(row["ID"], ""), "raw")
            ref_len = len(ref)
            if ref_len:
                ratios.append(item["chars"] / ref_len)
    return {
        "num_examples": len(rows),
        "empty_count": sum(item["empty"] for item in stats),
        "dot_only_count": sum(item["dot_only"] for item in stats),
        "very_short_count": sum(item["very_short"] for item in stats),
        "very_long_count": sum(item["very_long"] for item in stats),
        "repeated_ngram_count": sum(item["repeated_ngram"] for item in stats),
        "repeated_run_ge3_count": sum(item["repeated_run"] >= 3 for item in stats),
        "unusual_char_count": sum(bool(item["unusual_chars"]) for item in stats),
        "char_length": {
            "mean": mean(chars) if chars else None,
            "median": median(chars) if chars else None,
            "p05": q(chars, 0.05),
            "p95": q(chars, 0.95),
        },
        "word_length": {
            "mean": mean(words) if words else None,
            "median": median(words) if words else None,
            "p05": q(words, 0.05),
            "p95": q(words, 0.95),
        },
        "prediction_reference_char_ratio": {
            "mean": mean(ratios) if ratios else None,
            "median": median(ratios) if ratios else None,
            "p05": q(ratios, 0.05),
            "p95": q(ratios, 0.95),
        },
    }


def per_example_score(ref: str, pred: str, normalization: str) -> dict[str, float]:
    metrics = compute_group_metrics([ref], [pred], normalization=normalization)
    return {
        "wer": float(metrics["wer"]),
        "cer": float(metrics["cer"]),
        "combined": float(metrics["combined"]),
    }


def load_audio_quality(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    import csv

    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["ID"]: row for row in csv.DictReader(f)}


def collect_examples(
    rows: list[dict[str, str]],
    refs_by_id: dict[str, str] | None,
    normalization: str,
    max_examples: int,
) -> dict[str, list[dict[str, Any]]]:
    if not refs_by_id:
        short = []
        for row in rows:
            if id_language(row["ID"]) != "lin":
                continue
            st = text_stats(row.get("Target", ""))
            if st["dot_only"] or st["very_short"] or st["repeated_ngram"]:
                short.append({"ID": row["ID"], "prediction": row.get("Target", "")[:240], **st})
            if len(short) >= max_examples:
                break
        return {"suspicious_predictions": short}

    scored = []
    for row in rows:
        if id_language(row["ID"]) != "lin" or row["ID"] not in refs_by_id:
            continue
        pred = row.get("Target", "")
        ref = refs_by_id[row["ID"]]
        score = per_example_score(ref, pred, normalization)
        scored.append(
            {
                "ID": row["ID"],
                "combined": score["combined"],
                "wer": score["wer"],
                "cer": score["cer"],
                "reference": ref[:300],
                "prediction": pred[:300],
                **text_stats(pred),
            }
        )
    scored.sort(key=lambda item: item["combined"])
    return {
        "best_examples": scored[:max_examples],
        "worst_examples": scored[-max_examples:],
        "short_or_dot_examples": [
            item for item in scored if item["dot_only"] or item["very_short"] or item["repeated_ngram"]
        ][:max_examples],
    }


def compare_against_baseline(
    predictions_by_name: dict[str, list[dict[str, str]]],
    refs_by_id: dict[str, str],
    baseline_name: str | None,
    normalization: str,
    max_examples: int,
) -> dict[str, Any]:
    if not baseline_name or baseline_name not in predictions_by_name:
        return {}
    baseline = {row["ID"]: row.get("Target", "") for row in predictions_by_name[baseline_name]}
    comparisons: dict[str, Any] = {}
    for name, rows in predictions_by_name.items():
        if name == baseline_name:
            continue
        improved = []
        worsened = []
        current = {row["ID"]: row.get("Target", "") for row in rows}
        for example_id, baseline_pred in baseline.items():
            if id_language(example_id) != "lin" or example_id not in current or example_id not in refs_by_id:
                continue
            ref = refs_by_id[example_id]
            baseline_score = per_example_score(ref, baseline_pred, normalization)["combined"]
            current_score = per_example_score(ref, current[example_id], normalization)["combined"]
            delta = baseline_score - current_score
            record = {
                "ID": example_id,
                "delta_baseline_minus_current": delta,
                "reference": ref[:260],
                "baseline_prediction": baseline_pred[:260],
                "current_prediction": current[example_id][:260],
                "baseline_combined": baseline_score,
                "current_combined": current_score,
            }
            if delta > 0.15:
                improved.append(record)
            elif delta < -0.15:
                worsened.append(record)
        improved.sort(key=lambda item: item["delta_baseline_minus_current"], reverse=True)
        worsened.sort(key=lambda item: item["delta_baseline_minus_current"])
        comparisons[name] = {
            "strongly_improved_count": len(improved),
            "strongly_worsened_count": len(worsened),
            "strongly_improved_examples": improved[:max_examples],
            "strongly_worsened_examples": worsened[:max_examples],
        }
    return comparisons


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lingala Model Comparison",
        "",
        "This report is generated from local WAXAL Lingala validation/test predictions.",
        "",
        "## Models",
        "",
    ]
    for name, result in payload["models"].items():
        sanity = result["sanity"]
        metrics = result.get("metrics", {})
        weighted = metrics.get("overall_weighted", {}) if metrics else {}
        lines.append(f"### {name}")
        lines.append("")
        if weighted:
            zindi_score = 1.0 - float(weighted["combined"])
            lines.append(
                f"- WER: {float(weighted['wer']):.6f}; CER: {float(weighted['cer']):.6f}; "
                f"combined error: {float(weighted['combined']):.6f}; Zindi-style score: {zindi_score:.6f}"
            )
        lines.append(
            f"- Empty: {sanity['empty_count']}; dot-only: {sanity['dot_only_count']}; "
            f"very short: {sanity['very_short_count']}; repeated ngram: {sanity['repeated_ngram_count']}"
        )
        lines.append(
            f"- Mean chars: {sanity['char_length']['mean']}; median chars: {sanity['char_length']['median']}"
        )
        lines.append("")
    if payload.get("baseline_comparison"):
        lines.extend(["## Baseline Comparison", ""])
        for name, result in payload["baseline_comparison"].items():
            lines.append(
                f"- `{name}`: strongly improved {result['strongly_improved_count']}; "
                f"strongly worsened {result['strongly_worsened_count']}."
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.names and len(args.names) != len(args.predictions):
        raise ValueError("--names must match number of --predictions files")
    names = args.names or [path.stem for path in args.predictions]
    predictions_by_name = {
        name: read_prediction_csv(path)
        for name, path in zip(names, args.predictions, strict=True)
    }
    refs = references_from_validation_csv(args.references) if args.references else []
    refs = [row for row in refs if id_language(row["ID"]) == "lin"]
    refs_by_id = {row["ID"]: row["Target"] for row in refs} if refs else None
    policies = POLICIES if args.normalization == "all" else (args.normalization,)
    audio_quality = load_audio_quality(args.audio_quality)

    payload: dict[str, Any] = {
        "references": str(args.references) if args.references else "",
        "audio_quality": str(args.audio_quality) if args.audio_quality else "",
        "normalization": args.normalization,
        "models": {},
        "baseline_comparison": {},
    }
    for name, rows in predictions_by_name.items():
        model_result: dict[str, Any] = {
            "path": str(args.predictions[names.index(name)]),
            "sanity": summarize_predictions(rows, refs_by_id),
            "examples": collect_examples(
                rows,
                refs_by_id,
                policies[0] if args.normalization != "all" else "language_safe",
                args.max_examples,
            ),
        }
        if refs:
            model_result["metrics"] = {
                policy: score_records(refs, rows, normalization=policy)
                for policy in policies
            }
            if args.normalization != "all":
                model_result["metrics"] = model_result["metrics"][args.normalization]
        if audio_quality:
            suspicious_ids = {
                row["ID"]
                for row in rows
                if id_language(row["ID"]) == "lin"
                and (text_stats(row.get("Target", ""))["dot_only"] or text_stats(row.get("Target", ""))["very_short"])
            }
            grouped = defaultdict(int)
            for example_id in suspicious_ids:
                flags = audio_quality.get(example_id, {}).get("quality_flags", "")
                grouped[flags or "no_flags"] += 1
            model_result["suspicious_audio_quality_flag_counts"] = dict(sorted(grouped.items()))
        payload["models"][name] = model_result

    if refs_by_id:
        payload["baseline_comparison"] = compare_against_baseline(
            predictions_by_name,
            refs_by_id,
            args.baseline_name,
            policies[0] if args.normalization != "all" else "language_safe",
            args.max_examples,
        )

    json_dump(payload, args.output)
    write_markdown(args.markdown_output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
