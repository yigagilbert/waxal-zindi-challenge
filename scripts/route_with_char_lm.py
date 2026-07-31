#!/usr/bin/env python3
"""Route between two ASR systems with official-train-text character LMs.

Tune mode uses nested out-of-fold threshold selection on labeled validation:
  --references validation.csv

Apply mode loads the final thresholds from that report and uses only test-safe
features:
  --threshold-report tune_report.json --language-csv test_lid.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_prediction_csv, references_from_validation_csv  # noqa: E402
from waxal.scoring import edit_distance, score_records  # noqa: E402

EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
LANGUAGE_ALIASES = {"mas": "myx", "sog": "xog"}


def normalize_lm_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).lower().split())


class CharNGramLM:
    def __init__(self, order: int, alpha: float) -> None:
        self.order = order
        self.alpha = alpha
        self.ngrams: Counter[str] = Counter()
        self.contexts: Counter[str] = Counter()
        self.vocab: set[str] = {"\u0003"}

    def add(self, text: str) -> None:
        text = normalize_lm_text(text)
        self.vocab.update(text)
        padded = "\u0002" * (self.order - 1) + text + "\u0003"
        for pos in range(self.order - 1, len(padded)):
            context = padded[pos - self.order + 1 : pos]
            char = padded[pos]
            self.contexts[context] += 1
            self.ngrams[context + char] += 1

    def score(self, text: str) -> float:
        text = normalize_lm_text(text)
        padded = "\u0002" * (self.order - 1) + text + "\u0003"
        vocab_size = len(self.vocab) + 1
        log_prob = 0.0
        count = 0
        for pos in range(self.order - 1, len(padded)):
            context = padded[pos - self.order + 1 : pos]
            char = padded[pos]
            numerator = self.ngrams.get(context + char, 0) + self.alpha
            denominator = self.contexts.get(context, 0) + self.alpha * vocab_size
            log_prob += math.log(numerator / denominator)
            count += 1
        return log_prob / max(count, 1)


def load_predictions(path: Path) -> tuple[list[str], dict[str, str]]:
    rows = read_prediction_csv(path)
    return [row["ID"] for row in rows], {row["ID"]: row.get("Target", "") for row in rows}


def load_languages(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        value = (row.get("language") or row.get("Language") or "").strip()
        result[row["ID"]] = LANGUAGE_ALIASES.get(value, value)
    return result


def inferred_language(example_id: str) -> str:
    prefix = example_id.split("_", 1)[0]
    return LANGUAGE_ALIASES.get(prefix, prefix)


def max_ngram_count(text: str, order: int) -> int:
    tokens = [EDGE_PUNCTUATION.sub("", token.lower()) for token in text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < order:
        return 0
    counts = Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))
    return max(counts.values(), default=0)


def stable_fold(example_id: str, folds: int) -> int:
    digest = hashlib.sha1(example_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds


def choose_threshold(samples: list[dict], thresholds: list[float]) -> float:
    scored = []
    for threshold in thresholds:
        gain = sum(
            sample["delta"]
            for sample in samples
            if sample["loop"] or sample["margin"] > threshold
        )
        switches = sum(
            sample["loop"] or sample["margin"] > threshold
            for sample in samples
        )
        scored.append((gain, -switches, threshold))
    return max(scored)[2]


def write_predictions(path: Path, ordered_ids: list[str], predictions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Target"])
        writer.writeheader()
        for example_id in ordered_ids:
            writer.writerow({"ID": example_id, "Target": predictions[example_id]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset-dir", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--language-csv", type=Path, default=None)
    parser.add_argument("--threshold-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--lm-order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loop-ngram-order", type=int, default=4)
    parser.add_argument("--loop-min-count", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[-0.05, 0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 999.0],
    )
    args = parser.parse_args()
    if args.references is None and args.threshold_report is None:
        parser.error("Apply mode requires --threshold-report when --references is omitted")

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required") from exc

    dataset_path = args.train_dataset_dir / "hf_dataset"
    train = load_from_disk(dataset_path)["train"]
    models: dict[str, CharNGramLM] = {}
    for language, text in zip(train["language"], train["transcription"], strict=True):
        language = LANGUAGE_ALIASES.get(language, language)
        model = models.setdefault(language, CharNGramLM(args.lm_order, args.alpha))
        model.add(text)
    print(
        "trained character LMs:",
        {language: len(model.ngrams) for language, model in sorted(models.items())},
        flush=True,
    )

    ordered_ids, primary = load_predictions(args.primary)
    _, fallback = load_predictions(args.fallback)
    missing = [example_id for example_id in ordered_ids if example_id not in fallback]
    if missing:
        raise ValueError(f"Fallback is missing {len(missing)} IDs; first: {missing[:5]}")

    languages = load_languages(args.language_csv)
    references: list[dict[str, str]] = []
    ref_by_id: dict[str, dict[str, str]] = {}
    if args.references:
        references = references_from_validation_csv(args.references)
        ref_by_id = {row["ID"]: row for row in references}
        for row in references:
            language = row.get("language") or row.get("Language") or inferred_language(row["ID"])
            languages[row["ID"]] = LANGUAGE_ALIASES.get(language, language)

    samples = []
    if references:
        total_words = sum(len(row["Target"].split()) for row in references)
        total_chars = sum(len(" ".join(row["Target"].split())) for row in references)
    else:
        total_words = total_chars = 1

    for example_id in ordered_ids:
        language = languages.get(example_id) or inferred_language(example_id)
        if language not in models:
            raise ValueError(f"No character LM for language {language!r} (ID {example_id})")
        primary_text = primary[example_id]
        fallback_text = fallback[example_id]
        margin = models[language].score(fallback_text) - models[language].score(primary_text)
        loop = (
            max_ngram_count(primary_text, args.loop_ngram_order) >= args.loop_min_count
            and max_ngram_count(fallback_text, args.loop_ngram_order)
            < max_ngram_count(primary_text, args.loop_ngram_order)
        )
        sample = {
            "ID": example_id,
            "language": language,
            "margin": margin,
            "loop": loop,
            "delta": 0.0,
        }
        if references and example_id in ref_by_id:
            ref = " ".join(ref_by_id[example_id]["Target"].split())
            p = " ".join(primary_text.split())
            f = " ".join(fallback_text.split())
            primary_word_errors = edit_distance(ref.split(), p.split())
            fallback_word_errors = edit_distance(ref.split(), f.split())
            primary_char_errors = edit_distance(list(ref), list(p))
            fallback_char_errors = edit_distance(list(ref), list(f))
            sample["delta"] = (
                0.5 * (primary_word_errors - fallback_word_errors) / total_words
                + 0.5 * (primary_char_errors - fallback_char_errors) / total_chars
            )
        samples.append(sample)

    report: dict = {
        "lm_order": args.lm_order,
        "alpha": args.alpha,
        "loop_ngram_order": args.loop_ngram_order,
        "loop_min_count": args.loop_min_count,
        "num_rows": len(samples),
    }

    if references:
        threshold_choices: dict[str, list[float]] = defaultdict(list)
        oof_predictions: dict[str, str] = {}
        oof_switches: list[str] = []
        for fold in range(args.folds):
            for language in sorted(models):
                training = [
                    sample
                    for sample in samples
                    if sample["language"] == language and stable_fold(sample["ID"], args.folds) != fold
                ]
                held_out = [
                    sample
                    for sample in samples
                    if sample["language"] == language and stable_fold(sample["ID"], args.folds) == fold
                ]
                if not held_out:
                    continue
                threshold = choose_threshold(training, args.thresholds)
                threshold_choices[language].append(threshold)
                for sample in held_out:
                    switch = sample["loop"] or sample["margin"] > threshold
                    oof_predictions[sample["ID"]] = (
                        fallback[sample["ID"]] if switch else primary[sample["ID"]]
                    )
                    if switch:
                        oof_switches.append(sample["ID"])

        final_thresholds = {
            language: choose_threshold(
                [sample for sample in samples if sample["language"] == language],
                args.thresholds,
            )
            for language in sorted(models)
        }
        baseline_rows = [{"ID": row["ID"], "Target": primary[row["ID"]]} for row in references]
        oof_rows = [{"ID": row["ID"], "Target": oof_predictions[row["ID"]]} for row in references]
        baseline_metrics = score_records(references, baseline_rows, normalization="raw")
        oof_metrics = score_records(references, oof_rows, normalization="raw")
        baseline_error = baseline_metrics["overall_weighted"]["combined"]
        oof_error = oof_metrics["overall_weighted"]["combined"]
        report.update(
            {
                "mode": "tune",
                "folds": args.folds,
                "fold_thresholds": dict(threshold_choices),
                "final_thresholds": final_thresholds,
                "oof_num_switches": len(oof_switches),
                "oof_switches_by_language": dict(
                    Counter(languages[example_id] for example_id in oof_switches)
                ),
                "baseline": baseline_metrics,
                "oof": oof_metrics,
                "oof_score_gain": baseline_error - oof_error,
            }
        )
        write_predictions(args.output, [row["ID"] for row in references], oof_predictions)
    else:
        tuned = json.loads(args.threshold_report.read_text(encoding="utf-8"))
        final_thresholds = {key: float(value) for key, value in tuned["final_thresholds"].items()}
        applied: dict[str, str] = {}
        switches = []
        for sample in samples:
            threshold = final_thresholds[sample["language"]]
            switch = sample["loop"] or sample["margin"] > threshold
            applied[sample["ID"]] = fallback[sample["ID"]] if switch else primary[sample["ID"]]
            if switch:
                switches.append(
                    {
                        "ID": sample["ID"],
                        "language": sample["language"],
                        "margin": sample["margin"],
                        "loop": sample["loop"],
                    }
                )
        write_predictions(args.output, ordered_ids, applied)
        report.update(
            {
                "mode": "apply",
                "final_thresholds": final_thresholds,
                "num_switches": len(switches),
                "switches_by_language": dict(Counter(row["language"] for row in switches)),
                "switches": switches,
            }
        )

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"baseline", "oof", "switches"}}, indent=2))


if __name__ == "__main__":
    main()
