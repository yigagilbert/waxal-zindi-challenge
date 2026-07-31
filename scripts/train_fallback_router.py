#!/usr/bin/env python3
"""Cross-fit a conservative reference-free router between two ASR systems.

The router is trained only on official validation references.  At application
time it uses transcript-shape, model-disagreement, and LID-confidence features;
it never uses a reference or Phase-2 metadata.

Model/routing-rate selection is nested inside each outer validation fold.  This
prevents the reported OOF result from reusing the same labels for both tuning
and evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.scoring import edit_distance  # noqa: E402


TOKEN_EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
LANGUAGES = ("ach", "myx", "nyn", "xog")
RATES = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--lid", type=Path, default=None)
    parser.add_argument(
        "--extra-features-validation",
        type=Path,
        default=None,
        help="Optional ID-keyed CSV of additional numeric validation features.",
    )
    parser.add_argument("--apply-primary", type=Path, required=True)
    parser.add_argument("--apply-fallback", type=Path, required=True)
    parser.add_argument("--apply-lid", type=Path, default=None)
    parser.add_argument(
        "--extra-features-apply",
        type=Path,
        default=None,
        help="Optional ID-keyed CSV matching --extra-features-validation columns.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def load_csv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [row["ID"] for row in rows], {row["ID"]: row for row in rows}


def load_extra_features(
    path: Path | None,
    expected_columns: list[str] | None = None,
) -> tuple[dict[str, list[float]], list[str]]:
    if path is None:
        return {}, expected_columns or []
    order, rows = load_csv(path)
    del order
    if not rows:
        return {}, expected_columns or []
    first = next(iter(rows.values()))
    columns = expected_columns or sorted(column for column in first if column != "ID")
    missing = [column for column in columns if column not in first]
    if missing:
        raise ValueError(f"{path} is missing extra feature columns: {missing}")
    features = {
        row_id: [float(row.get(column, 0.0) or 0.0) for column in columns]
        for row_id, row in rows.items()
    }
    return features, columns


def normalized_tokens(text: str) -> list[str]:
    tokens = [TOKEN_EDGE.sub("", token.lower()) for token in text.split()]
    return [token for token in tokens if token]


def max_ngram_count(tokens: list[str], order: int) -> int:
    if len(tokens) < order:
        return 0
    return max(
        Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)).values(),
        default=0,
    )


def repeated_ngram_fraction(tokens: list[str], order: int) -> float:
    if len(tokens) < order:
        return 0.0
    grams = [tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1.0)


def language_for(row_id: str, reference_row: dict[str, str] | None = None) -> str:
    if reference_row:
        value = reference_row.get("language", "")
        if value in LANGUAGES:
            return value
        if value == "mas":
            return "myx"
        if value == "sog":
            return "xog"
    prefix = row_id.split("_", 1)[0]
    return {"mas": "myx", "sog": "xog"}.get(prefix, prefix)


def text_features(
    primary: str,
    fallback: str,
    language: str,
    lid_row: dict[str, str] | None,
) -> list[float]:
    p_tokens = normalized_tokens(primary)
    f_tokens = normalized_tokens(fallback)
    p_chars = len(primary)
    f_chars = len(fallback)
    p_words = len(p_tokens)
    f_words = len(f_tokens)
    p_set = set(p_tokens)
    f_set = set(f_tokens)
    union = p_set | f_set

    features: list[float] = [
        math.log1p(p_chars),
        math.log1p(f_chars),
        math.log((p_chars + 1) / (f_chars + 1)),
        math.log1p(p_words),
        math.log1p(f_words),
        math.log((p_words + 1) / (f_words + 1)),
        safe_ratio(len(p_set), p_words),
        safe_ratio(len(f_set), f_words),
        safe_ratio(len(p_set & f_set), len(union)),
        safe_ratio(edit_distance(p_tokens, f_tokens), max(p_words, f_words)),
        safe_ratio(edit_distance(list(primary.lower()), list(fallback.lower())), max(p_chars, f_chars)),
        safe_ratio(len(zlib.compress(primary.encode("utf-8"))), p_chars),
        safe_ratio(len(zlib.compress(fallback.encode("utf-8"))), f_chars),
        safe_ratio(sum(not char.isalnum() and not char.isspace() for char in primary), p_chars),
        safe_ratio(sum(not char.isalnum() and not char.isspace() for char in fallback), f_chars),
        safe_ratio(sum(len(token) for token in p_tokens), p_words),
        safe_ratio(sum(len(token) for token in f_tokens), f_words),
        float(max((len(token) for token in p_tokens), default=0)),
        float(max((len(token) for token in f_tokens), default=0)),
    ]
    for order in range(1, 6):
        features.extend(
            [
                float(max_ngram_count(p_tokens, order)),
                float(max_ngram_count(f_tokens, order)),
                repeated_ngram_fraction(p_tokens, order),
                repeated_ngram_fraction(f_tokens, order),
            ]
        )
    confidence = float((lid_row or {}).get("confidence", 0.0) or 0.0)
    margin = float((lid_row or {}).get("margin", 0.0) or 0.0)
    features.extend([confidence, margin])
    features.extend(float(language == candidate) for candidate in LANGUAGES)
    return features


def prediction_map(path: Path) -> tuple[list[str], dict[str, str]]:
    order, rows = load_csv(path)
    return order, {row_id: row.get("Target", row.get("prediction", "")) for row_id, row in rows.items()}


def build_model(name: str, seed: int):
    if name == "hist3":
        return HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=3,
            min_samples_leaf=60,
            l2_regularization=10.0,
            random_state=seed,
        )
    if name == "hist7":
        return HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=60,
            l2_regularization=10.0,
            random_state=seed,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=30,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "extra":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=30,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(name)


MODEL_NAMES = ("hist3", "hist7", "rf", "extra")


def strata(languages: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{language}_{'win' if gain > 0 else 'lose'}" for language, gain in zip(languages, gains)]
    )


def selected_indices(scores: np.ndarray, rate: float) -> np.ndarray:
    count = min(len(scores), int(round(rate * len(scores))))
    if count <= 0:
        return np.asarray([], dtype=int)
    return np.argsort(scores, kind="stable")[-count:]


def choose_model_and_rate(
    x: np.ndarray,
    gains: np.ndarray,
    languages: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[str, float, dict]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    split_list = list(splitter.split(x, strata(languages, gains)))
    candidates = []
    for model_name in MODEL_NAMES:
        predictions = np.zeros(len(x), dtype=float)
        fold_index = np.zeros(len(x), dtype=int)
        for fold, (train_idx, valid_idx) in enumerate(split_list):
            model = build_model(model_name, seed + fold)
            model.fit(x[train_idx], gains[train_idx])
            predictions[valid_idx] = model.predict(x[valid_idx])
            fold_index[valid_idx] = fold
        for rate in RATES:
            selected = []
            fold_gains = []
            for fold in range(folds):
                valid_idx = np.flatnonzero(fold_index == fold)
                local = selected_indices(predictions[valid_idx], rate)
                chosen = valid_idx[local]
                selected.extend(chosen.tolist())
                fold_gains.append(float(gains[chosen].sum()))
            total_gain = float(gains[np.asarray(selected, dtype=int)].sum()) if selected else 0.0
            stable_folds = sum(gain >= 0.0 for gain in fold_gains)
            candidates.append(
                {
                    "model": model_name,
                    "rate": rate,
                    "gain": total_gain,
                    "fold_gains": fold_gains,
                    "stable_folds": stable_folds,
                    "num_switches": len(selected),
                }
            )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["stable_folds"] >= folds - 1 and min(candidate["fold_gains"]) >= -0.00025
    ]
    if not eligible:
        best = next(candidate for candidate in candidates if candidate["rate"] == 0.0)
    else:
        best = max(eligible, key=lambda candidate: (candidate["gain"], -candidate["rate"]))
    return best["model"], best["rate"], {"best": best, "candidates": candidates}


def write_predictions(path: Path, order: list[str], values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Target"])
        writer.writerows((row_id, values[row_id]) for row_id in order)


def main() -> None:
    args = parse_args()
    order, primary = prediction_map(args.primary)
    _, fallback = prediction_map(args.fallback)
    _, references = load_csv(args.references)
    _, lid = load_csv(args.lid) if args.lid else ([], {})
    validation_extra, extra_columns = load_extra_features(args.extra_features_validation)

    ids = [
        row_id
        for row_id in order
        if row_id in primary and row_id in fallback and row_id in references
    ]
    total_words = sum(len(references[row_id]["Target"].split()) for row_id in ids)
    total_chars = sum(len(references[row_id]["Target"]) for row_id in ids)

    x_rows: list[list[float]] = []
    gains: list[float] = []
    languages: list[str] = []
    row_details = []
    for row_id in ids:
        ref = " ".join(references[row_id]["Target"].split())
        p = " ".join(primary[row_id].split())
        f = " ".join(fallback[row_id].split())
        language = language_for(row_id, references[row_id])
        p_word_errors = edit_distance(ref.split(), p.split())
        f_word_errors = edit_distance(ref.split(), f.split())
        p_char_errors = edit_distance(list(ref), list(p))
        f_char_errors = edit_distance(list(ref), list(f))
        exact_gain = (
            0.5 * (p_word_errors - f_word_errors) / total_words
            + 0.5 * (p_char_errors - f_char_errors) / total_chars
        )
        x_rows.append(
            text_features(p, f, language, lid.get(row_id))
            + validation_extra.get(row_id, [0.0] * len(extra_columns))
        )
        gains.append(exact_gain)
        languages.append(language)
        row_details.append(
            {
                "ID": row_id,
                "language": language,
                "primary_word_errors": p_word_errors,
                "fallback_word_errors": f_word_errors,
                "primary_char_errors": p_char_errors,
                "fallback_char_errors": f_char_errors,
                "exact_gain": exact_gain,
            }
        )

    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(gains, dtype=float)
    language_array = np.asarray(languages)
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    outer_splits = list(outer.split(x, strata(language_array, y)))
    oof_scores = np.zeros(len(ids), dtype=float)
    oof_selected = np.zeros(len(ids), dtype=bool)
    fold_reports = []

    for fold, (train_idx, valid_idx) in enumerate(outer_splits):
        model_name, rate, inner_report = choose_model_and_rate(
            x[train_idx],
            y[train_idx],
            language_array[train_idx],
            folds=max(args.folds - 1, 3),
            seed=args.seed + 100 * (fold + 1),
        )
        model = build_model(model_name, args.seed + fold)
        model.fit(x[train_idx], y[train_idx])
        scores = model.predict(x[valid_idx])
        chosen_local = selected_indices(scores, rate)
        chosen = valid_idx[chosen_local]
        oof_scores[valid_idx] = scores
        oof_selected[chosen] = True
        fold_reports.append(
            {
                "fold": fold,
                "model": model_name,
                "rate": rate,
                "num_switches": int(len(chosen)),
                "gain": float(y[chosen].sum()),
                "inner_best": inner_report["best"],
            }
        )

    oof_values = dict(primary)
    for index, row_id in enumerate(ids):
        if oof_selected[index]:
            oof_values[row_id] = fallback[row_id]
    write_predictions(args.oof_output, order, oof_values)

    final_model_name, final_rate, final_tuning = choose_model_and_rate(
        x,
        y,
        language_array,
        folds=args.folds,
        seed=args.seed + 999,
    )
    final_model = build_model(final_model_name, args.seed + 9999)
    final_model.fit(x, y)

    apply_order, apply_primary = prediction_map(args.apply_primary)
    _, apply_fallback = prediction_map(args.apply_fallback)
    _, apply_lid = load_csv(args.apply_lid) if args.apply_lid else ([], {})
    apply_extra, apply_extra_columns = load_extra_features(
        args.extra_features_apply,
        expected_columns=extra_columns,
    )
    if apply_extra_columns != extra_columns:
        raise ValueError("Apply extra feature columns do not match validation")
    apply_ids = [
        row_id for row_id in apply_order if row_id in apply_primary and row_id in apply_fallback
    ]
    apply_x = np.asarray(
        [
            text_features(
                apply_primary[row_id],
                apply_fallback[row_id],
                language_for(row_id, apply_lid.get(row_id)),
                apply_lid.get(row_id),
            )
            + apply_extra.get(row_id, [0.0] * len(extra_columns))
            for row_id in apply_ids
        ],
        dtype=float,
    )
    apply_scores = final_model.predict(apply_x)
    apply_selected_local = selected_indices(apply_scores, final_rate)
    apply_selected_ids = {apply_ids[index] for index in apply_selected_local}
    output_values = dict(apply_primary)
    for row_id in apply_selected_ids:
        output_values[row_id] = apply_fallback[row_id]
    write_predictions(args.output, apply_order, output_values)

    selected_details = []
    for index, row_id in enumerate(apply_ids):
        if row_id in apply_selected_ids:
            selected_details.append(
                {
                    "ID": row_id,
                    "language": language_for(row_id, apply_lid.get(row_id)),
                    "score": float(apply_scores[index]),
                    "primary": apply_primary[row_id],
                    "fallback": apply_fallback[row_id],
                }
            )
    selected_details.sort(key=lambda row: row["score"], reverse=True)

    oof_gain = float(y[oof_selected].sum())
    by_language = {
        language: {
            "num_switches": int(
                sum(oof_selected[index] and languages[index] == language for index in range(len(ids)))
            ),
            "gain": float(
                sum(
                    y[index]
                    for index in range(len(ids))
                    if oof_selected[index] and languages[index] == language
                )
            ),
        }
        for language in LANGUAGES
    }
    report = {
        "num_validation": len(ids),
        "num_features": int(x.shape[1]),
        "extra_feature_columns": extra_columns,
        "outer_folds": fold_reports,
        "oof_num_switches": int(oof_selected.sum()),
        "oof_exact_combined_gain": oof_gain,
        "oof_positive_folds": sum(fold["gain"] >= 0 for fold in fold_reports),
        "oof_by_language": by_language,
        "final_model": final_model_name,
        "final_rate": final_rate,
        "final_tuning": final_tuning["best"],
        "apply_num_rows": len(apply_ids),
        "apply_num_switches": len(apply_selected_ids),
        "apply_switches": selected_details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
