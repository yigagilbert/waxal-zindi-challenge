"""Local WER/CER scoring for WAXAL ASR experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .data import id_language
from .text_normalization import normalize_text


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Compute Levenshtein edit distance with O(min(n, m)) memory."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def wer_stats(references: Iterable[str], predictions: Iterable[str]) -> tuple[int, int]:
    """Return (word_errors, reference_word_count)."""
    errors = 0
    denom = 0
    for ref, hyp in zip(references, predictions, strict=True):
        ref_words = ref.split()
        hyp_words = hyp.split()
        errors += edit_distance(ref_words, hyp_words)
        denom += len(ref_words)
    return errors, denom


def cer_stats(references: Iterable[str], predictions: Iterable[str]) -> tuple[int, int]:
    """Return (char_errors, reference_char_count)."""
    errors = 0
    denom = 0
    for ref, hyp in zip(references, predictions, strict=True):
        errors += edit_distance(list(ref), list(hyp))
        denom += len(ref)
    return errors, denom


def safe_rate(errors: int, denom: int) -> float:
    """Compute an error rate with a stable empty-reference fallback."""
    if denom == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / denom


def compute_group_metrics(
    references: list[str],
    predictions: list[str],
    *,
    normalization: str = "raw",
) -> dict[str, float | int]:
    """Compute WER, CER, and combined score for one group."""
    refs = [normalize_text(x, normalization) for x in references]
    hyps = [normalize_text(x, normalization) for x in predictions]
    word_errors, word_count = wer_stats(refs, hyps)
    char_errors, char_count = cer_stats(refs, hyps)
    wer = safe_rate(word_errors, word_count)
    cer = safe_rate(char_errors, char_count)
    return {
        "num_examples": len(refs),
        "word_errors": word_errors,
        "reference_words": word_count,
        "char_errors": char_errors,
        "reference_chars": char_count,
        "wer": wer,
        "cer": cer,
        "combined": 0.5 * wer + 0.5 * cer,
    }


def score_records(
    references: list[dict[str, str]],
    predictions: list[dict[str, str]],
    *,
    normalization: str = "raw",
) -> dict:
    """Score ID,Target predictions against reference records.

    Reference records may include `language`; otherwise language is inferred
    from the ID prefix.
    """
    ref_by_id = {row["ID"]: row for row in references}
    pred_by_id = {row["ID"]: row for row in predictions}
    missing_predictions = sorted(set(ref_by_id) - set(pred_by_id))
    extra_predictions = sorted(set(pred_by_id) - set(ref_by_id))
    common_ids = [row["ID"] for row in references if row["ID"] in pred_by_id]

    refs = [ref_by_id[i]["Target"] for i in common_ids]
    hyps = [pred_by_id[i]["Target"] for i in common_ids]
    langs = [
        ref_by_id[i].get("language") or ref_by_id[i].get("Language") or id_language(i)
        for i in common_ids
    ]

    overall = compute_group_metrics(refs, hyps, normalization=normalization)
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"refs": [], "hyps": []})
    for lang, ref, hyp in zip(langs, refs, hyps, strict=True):
        grouped[lang]["refs"].append(ref)
        grouped[lang]["hyps"].append(hyp)

    by_language = {
        lang: compute_group_metrics(values["refs"], values["hyps"], normalization=normalization)
        for lang, values in sorted(grouped.items())
    }
    if by_language:
        macro = {
            "wer": sum(v["wer"] for v in by_language.values()) / len(by_language),
            "cer": sum(v["cer"] for v in by_language.values()) / len(by_language),
            "combined": sum(v["combined"] for v in by_language.values()) / len(by_language),
            "num_languages": len(by_language),
        }
    else:
        macro = {"wer": 0.0, "cer": 0.0, "combined": 0.0, "num_languages": 0}

    return {
        "normalization": normalization,
        "overall_weighted": overall,
        "macro_by_language": macro,
        "by_language": by_language,
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
    }

