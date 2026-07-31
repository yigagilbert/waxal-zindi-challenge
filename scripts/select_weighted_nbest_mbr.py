#!/usr/bin/env python3
"""Frequency/posterior-weighted n-best MBR with nested validation tuning.

The original selector collapsed duplicate samples and then gave every remaining
hypothesis equal weight. That overweights one-off variants. This selector keeps
sampling frequency, optionally incorporates average token log-probabilities,
and gives the deterministic anchor a tunable prior weight.

Tune mode requires references and reports an out-of-fold result. Apply mode
loads the frozen final parameters from a tune report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_prediction_csv, references_from_validation_csv  # noqa: E402
from waxal.scoring import edit_distance, score_records  # noqa: E402
from waxal.text_normalization import POLICIES, normalize_text  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein as FastLevenshtein
except ImportError:
    FastLevenshtein = None


def pair_cost(left: str, right: str) -> float:
    left_words, right_words = left.split(), right.split()
    word_denom = max(len(left_words), len(right_words), 1)
    char_denom = max(len(left), len(right), 1)
    if FastLevenshtein is not None:
        word_distance = FastLevenshtein.distance(left_words, right_words)
        char_distance = FastLevenshtein.distance(left, right)
    else:
        word_distance = edit_distance(left_words, right_words)
        char_distance = edit_distance(list(left), list(right))
    return 0.5 * word_distance / word_denom + 0.5 * char_distance / char_denom


def load_predictions(path: Path) -> tuple[list[str], dict[str, str]]:
    rows = read_prediction_csv(path)
    return [row["ID"] for row in rows], {row["ID"]: row["Target"] for row in rows}


def stable_fold(example_id: str, folds: int) -> int:
    digest = hashlib.sha1(example_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds


def load_nbest(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[row["ID"]].append(row)
    return groups


def candidate_risks(
    rows: list[dict[str, str]],
    anchor_text: str,
    normalization: str,
    anchor_weight: float,
    score_temperature: float | None,
) -> tuple[list[dict], str]:
    anchor_normalized = normalize_text(anchor_text, normalization)
    raw = []
    finite_scores = []
    for row in rows:
        normalized = normalize_text(row["Target"], normalization)
        score_text = row.get("sequence_score", "")
        try:
            score = float(score_text)
        except (TypeError, ValueError):
            score = 0.0
        if math.isfinite(score):
            finite_scores.append(score)
        raw.append((row["Target"], normalized, score))
    best_score = max(finite_scores, default=0.0)

    grouped: dict[str, dict] = {}
    order: list[str] = []
    for surface, normalized, score in raw:
        if normalized not in grouped:
            grouped[normalized] = {
                "normalized": normalized,
                "surface": surface,
                "sample_count": 0,
                "sample_weight": 0.0,
                "max_score": score,
            }
            order.append(normalized)
        item = grouped[normalized]
        item["sample_count"] += 1
        item["max_score"] = max(item["max_score"], score)
        if score_temperature is None or score_temperature <= 0 or not math.isfinite(score):
            item["sample_weight"] += 1.0
        else:
            item["sample_weight"] += math.exp(
                max((score - best_score) / score_temperature, -50.0)
            )

    if anchor_normalized not in grouped:
        grouped[anchor_normalized] = {
            "normalized": anchor_normalized,
            "surface": anchor_text,
            "sample_count": 0,
            "sample_weight": 0.0,
            "max_score": float("-inf"),
        }
        order.insert(0, anchor_normalized)
    else:
        # Always emit the trusted deterministic surface form when its normalized
        # form wins.
        grouped[anchor_normalized]["surface"] = anchor_text
        order.remove(anchor_normalized)
        order.insert(0, anchor_normalized)

    grouped[anchor_normalized]["anchor_weight"] = anchor_weight
    candidates = [grouped[key] for key in order]
    for item in candidates:
        item.setdefault("anchor_weight", 0.0)
        item["weight"] = item["sample_weight"] + item["anchor_weight"]

    total_weight = sum(item["weight"] for item in candidates)
    for item in candidates:
        item["risk"] = (
            sum(
                other["weight"] * pair_cost(item["normalized"], other["normalized"])
                for other in candidates
            )
            / max(total_weight, 1e-12)
        )
    return candidates, anchor_normalized


def select_one(
    rows: list[dict[str, str]],
    anchor_text: str,
    normalization: str,
    anchor_weight: float,
    score_temperature: float | None,
    margin: float,
) -> dict:
    candidates, anchor_normalized = candidate_risks(
        rows,
        anchor_text,
        normalization,
        anchor_weight,
        score_temperature,
    )
    anchor = next(item for item in candidates if item["normalized"] == anchor_normalized)
    best = min(
        candidates,
        key=lambda item: (
            item["risk"],
            item["normalized"] != anchor_normalized,
            -item["weight"],
        ),
    )
    advantage = anchor["risk"] - best["risk"]
    use_best = best["normalized"] != anchor_normalized and advantage > margin
    selected = best if use_best else anchor
    return {
        "Target": selected["surface"],
        "selected_alternative": use_best,
        "anchor_risk": anchor["risk"],
        "selected_risk": selected["risk"],
        "anchor_advantage": advantage,
        "sample_count": selected["sample_count"],
        "sample_weight": selected["sample_weight"],
    }


def parameter_grid(
    anchor_weights: list[float],
    margins: list[float],
    score_temperatures: list[float],
    has_real_scores: bool,
) -> list[tuple[float, float, float | None]]:
    temperatures: list[float | None] = [None]
    if has_real_scores:
        temperatures.extend(score_temperatures)
    return [
        (anchor_weight, margin, temperature)
        for anchor_weight in anchor_weights
        for margin in margins
        for temperature in temperatures
    ]


def corpus_error(
    ids: list[str],
    predictions: dict[str, str],
    references: dict[str, str],
) -> float:
    total_words = sum(len(references[row_id].split()) for row_id in ids)
    total_chars = sum(len(references[row_id]) for row_id in ids)
    word_errors = 0
    char_errors = 0
    for row_id in ids:
        ref = references[row_id]
        hyp = predictions[row_id]
        if FastLevenshtein is not None:
            word_errors += FastLevenshtein.distance(ref.split(), hyp.split())
            char_errors += FastLevenshtein.distance(ref, hyp)
        else:
            word_errors += edit_distance(ref.split(), hyp.split())
            char_errors += edit_distance(list(ref), list(hyp))
    return (
        0.5 * word_errors / max(total_words, 1)
        + 0.5 * char_errors / max(total_chars, 1)
    )


def encode_temperature(value: float | None) -> str:
    return "frequency" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--primary", type=Path, default=None)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--tune-report", type=Path, default=None)
    parser.add_argument("--normalization", choices=POLICIES, default="raw")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--anchor-weights",
        type=float,
        nargs="*",
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument(
        "--margins",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    )
    parser.add_argument(
        "--score-temperatures",
        type=float,
        nargs="*",
        default=[0.025, 0.05, 0.1, 0.2, 0.5],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.references is None and args.tune_report is None:
        parser.error("Apply mode requires --tune-report when references are absent")

    nbest = load_nbest(args.nbest)
    order, anchor = load_predictions(args.anchor)
    if args.primary:
        _, primary = load_predictions(args.primary)
    else:
        primary = dict(anchor)
    ids = [row_id for row_id in order if row_id in nbest and row_id in anchor]

    observed_scores = []
    for rows in nbest.values():
        for row in rows:
            try:
                observed_scores.append(float(row.get("sequence_score", "0")))
            except ValueError:
                pass
    has_real_scores = any(math.isfinite(score) and abs(score) > 1e-12 for score in observed_scores)

    references: dict[str, str] = {}
    if args.references:
        references = {
            row["ID"]: normalize_text(row["Target"], args.normalization)
            for row in references_from_validation_csv(args.references)
            if row["ID"] in ids
        }
        ids = [row_id for row_id in ids if row_id in references]

    report: dict = {
        "num_rows": len(ids),
        "has_real_scores": has_real_scores,
        "normalization": args.normalization,
    }
    if references:
        grid = parameter_grid(
            args.anchor_weights,
            args.margins,
            args.score_temperatures,
            has_real_scores,
        )
        cache: dict[tuple[float, float | None], dict[str, dict]] = {}
        for anchor_weight, _, temperature in grid:
            key = (anchor_weight, temperature)
            if key not in cache:
                cache[key] = {
                    row_id: select_one(
                        nbest[row_id],
                        anchor[row_id],
                        args.normalization,
                        anchor_weight,
                        temperature,
                        margin=-1.0,
                    )
                    for row_id in ids
                }

        def predictions_for(
            subset: list[str],
            params: tuple[float, float, float | None],
        ) -> dict[str, str]:
            anchor_weight, margin, temperature = params
            selections = cache[(anchor_weight, temperature)]
            return {
                row_id: (
                    selections[row_id]["Target"]
                    if selections[row_id]["selected_alternative"]
                    and selections[row_id]["anchor_advantage"] > margin
                    and primary[row_id] == anchor[row_id]
                    else primary[row_id]
                )
                for row_id in subset
            }

        folds = {
            fold: [row_id for row_id in ids if stable_fold(row_id, args.folds) == fold]
            for fold in range(args.folds)
        }
        oof_predictions = dict(primary)
        fold_reports = []
        for fold, held_ids in folds.items():
            train_ids = [row_id for row_id in ids if stable_fold(row_id, args.folds) != fold]
            scored = []
            for params in grid:
                preds = predictions_for(train_ids, params)
                error = corpus_error(train_ids, preds, references)
                switches = sum(preds[row_id] != primary[row_id] for row_id in train_ids)
                scored.append((error, switches, params))
            _, _, best_params = min(scored, key=lambda item: (item[0], item[1]))
            held_predictions = predictions_for(held_ids, best_params)
            oof_predictions.update(held_predictions)
            fold_reports.append(
                {
                    "fold": fold,
                    "num_rows": len(held_ids),
                    "anchor_weight": best_params[0],
                    "margin": best_params[1],
                    "score_temperature": encode_temperature(best_params[2]),
                    "num_switches": sum(
                        held_predictions[row_id] != primary[row_id] for row_id in held_ids
                    ),
                    "primary_error": corpus_error(
                        held_ids,
                        {row_id: primary[row_id] for row_id in held_ids},
                        references,
                    ),
                    "selected_error": corpus_error(
                        held_ids,
                        held_predictions,
                        references,
                    ),
                }
            )

        full_grid = []
        for params in grid:
            predictions = predictions_for(ids, params)
            full_grid.append(
                {
                    "anchor_weight": params[0],
                    "margin": params[1],
                    "score_temperature": encode_temperature(params[2]),
                    "error": corpus_error(ids, predictions, references),
                    "num_switches": sum(
                        predictions[row_id] != primary[row_id] for row_id in ids
                    ),
                    "_params": params,
                }
            )
        best = min(full_grid, key=lambda row: (row["error"], row["num_switches"]))
        final_params = best.pop("_params")
        for row in full_grid:
            row.pop("_params", None)
        final_predictions = predictions_for(ids, final_params)
        oof_error = corpus_error(ids, oof_predictions, references)
        primary_error = corpus_error(
            ids,
            {row_id: primary[row_id] for row_id in ids},
            references,
        )
        report.update(
            {
                "mode": "tune",
                "folds": fold_reports,
                "oof_primary_error": primary_error,
                "oof_selected_error": oof_error,
                "oof_gain": primary_error - oof_error,
                "oof_num_switches": sum(
                    oof_predictions[row_id] != primary[row_id] for row_id in ids
                ),
                "final_params": {
                    "anchor_weight": final_params[0],
                    "margin": final_params[1],
                    "score_temperature": encode_temperature(final_params[2]),
                },
                "full_validation_best": best,
                "grid": sorted(full_grid, key=lambda row: row["error"]),
            }
        )
        output_predictions = final_predictions
    else:
        tuned = json.loads(args.tune_report.read_text(encoding="utf-8"))
        final = tuned["final_params"]
        temperature = (
            None
            if final["score_temperature"] == "frequency"
            else float(final["score_temperature"])
        )
        output_predictions = {}
        switches = []
        for row_id in ids:
            selected = select_one(
                nbest[row_id],
                anchor[row_id],
                args.normalization,
                float(final["anchor_weight"]),
                temperature,
                float(final["margin"]),
            )
            protected = primary[row_id] != anchor[row_id]
            use_alternative = selected["selected_alternative"] and not protected
            output_predictions[row_id] = (
                selected["Target"] if use_alternative else primary[row_id]
            )
            if use_alternative:
                switches.append(row_id)
        report.update(
            {
                "mode": "apply",
                "final_params": final,
                "num_switches": len(switches),
                "switches": switches,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(
            {"ID": row_id, "Target": output_predictions.get(row_id, primary[row_id])}
            for row_id in order
        )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "grid"}, indent=2))


if __name__ == "__main__":
    main()
