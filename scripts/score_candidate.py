#!/usr/bin/env python3
"""Pre-submission scorer for the corrected Phase-2 validation bench.

Any change must first be produced on the labeled validation bench, scored here,
and only promoted to a test submission if it beats the baseline. The metric
semantics below are inferred from paired public submissions:

  * WER is CASE-INSENSITIVE   (two submissions differing only in casing returned
                               byte-identical WER 0.413864714)
  * CER is CASE-SENSITIVE     (the same pair moved CER 0.137837 -> 0.131082)
  * score = 1 - (WER + CER) / 2, languages weighted by their test counts

Calibration against real submissions (bench score -> public score):

  champion, no casing     0.8779 -> 0.724149
  champion + casing       0.8814 -> 0.727527      (delta -0.0035 predicted, -0.0034 actual)
  champion, alpha 1.1/1.0 0.8727 -> 0.689930      (delta -0.0087 predicted, -0.0376 actual)

The bench transfers well for text changes and has been directionally useful for
decode changes. It is *not* a valid promotion gate for acoustic changes: the
robust checkpoint gained +0.0019 here and lost -0.0013 publicly. Callers must
therefore declare the change class, and acoustic gains are held for independent
out-of-domain evidence rather than marked ready to submit.

Usage:
  python scripts/score_candidate.py --predictions outputs/day4_h100/bench_alpha_up.csv \
    --label "alpha 1.1/1.0"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz.distance import Levenshtein as L

BASELINE = 0.881433  # fallback only; exact value is recomputed when the file exists
BASELINE_PREDICTIONS = Path("outputs/day4_h100/bench_champion_val.csv")
TEST_WEIGHTS = {"lin": 448.0, "sna": 444.0}
_AFTER_SENTENCE_END = re.compile(r"([.!?]['\"\)\]]?\s+)([a-z])")


def restore_casing(text: str) -> str:
    if not text:
        return text
    out = text[:1].upper() + text[1:]
    return _AFTER_SENTENCE_END.sub(lambda m: m.group(1) + m.group(2).upper(), out)


def read_predictions(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows or set(rows[0]) != {"ID", "Target"}:
        raise SystemExit(f"{path}: expected non-empty CSV with exactly ID,Target columns")
    ids = [row["ID"] for row in rows]
    duplicate_ids = sorted({example_id for example_id in ids if ids.count(example_id) > 1})
    if duplicate_ids:
        raise SystemExit(f"{path}: duplicate IDs; first: {duplicate_ids[:5]}")
    return {row["ID"]: row["Target"] for row in rows}


def score_predictions(
    bench: dict[str, tuple[str, str]],
    pred: dict[str, str],
    transform,
) -> tuple[dict[str, dict[str, float | int]], float]:
    missing = [example_id for example_id in bench if example_id not in pred]
    if missing:
        raise SystemExit(
            f"candidate is missing {len(missing)} of {len(bench)} bench IDs; "
            f"first: {missing[:5]}"
        )

    per: dict[str, dict[str, float | int]] = {}
    for lang in sorted(TEST_WEIGHTS):
        ids = [example_id for example_id, (code, _) in bench.items() if code == lang]
        if not ids:
            raise SystemExit(f"bench contains no rows for required language {lang!r}")
        word_errors = reference_words = char_errors = reference_chars = 0
        for example_id in ids:
            ref = bench[example_id][1]
            hyp = transform(pred[example_id])
            word_errors += L.distance(ref.lower().split(), hyp.lower().split())
            reference_words += len(ref.split())
            char_errors += L.distance(ref, hyp)
            reference_chars += len(ref)
        per[lang] = {
            "n": len(ids),
            "wer": word_errors / max(reference_words, 1),
            "cer": char_errors / max(reference_chars, 1),
        }

    denominator = sum(TEST_WEIGHTS.values())
    combined = sum(
        (0.5 * float(per[lang]["wer"]) + 0.5 * float(per[lang]["cer"]))
        * TEST_WEIGHTS[lang]
        for lang in TEST_WEIGHTS
    ) / denominator
    return per, 1 - combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=Path("outputs/day4_h100/bench_ids.csv"))
    parser.add_argument("--label", default=None, help="Name for the decision log.")
    parser.add_argument(
        "--change-class",
        choices=["text", "decode", "acoustic", "unknown"],
        default="unknown",
        help="Controls promotion semantics; acoustic gains require an independent OOD gate.",
    )
    parser.add_argument("--baseline", type=float, default=None)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=BASELINE_PREDICTIONS,
        help="Current champion validation predictions; scored with casing restoration.",
    )
    parser.add_argument("--no-casing", action="store_true", help="Score as-is instead of applying casing restoration.")
    parser.add_argument("--log", type=Path, default=Path("outputs/day4_h100/CANDIDATES.jsonl"))
    args = parser.parse_args()

    bench = {
        r["ID"]: (r["language"], r["Target"])
        for r in csv.DictReader(args.bench.open(encoding="utf-8-sig"))
    }
    pred = read_predictions(args.predictions)
    transform = (lambda t: t) if args.no_casing else restore_casing
    per, score = score_predictions(bench, pred, transform)

    if args.baseline is not None:
        baseline = args.baseline
        baseline_source = "CLI value"
    elif args.baseline_predictions.exists():
        baseline_pred = read_predictions(args.baseline_predictions)
        _, baseline = score_predictions(bench, baseline_pred, restore_casing)
        baseline_source = str(args.baseline_predictions)
    else:
        baseline = BASELINE
        baseline_source = "fallback constant"

    delta = score - baseline
    if delta <= 0:
        verdict = "FAIL - do not submit"
        submit_eligible = False
    elif args.change_class in {"text", "decode"}:
        verdict = "PASS - eligible for submission"
        submit_eligible = True
    elif args.change_class == "acoustic":
        verdict = "HOLD - acoustic bench gains do not reliably transfer"
        submit_eligible = False
    else:
        verdict = "HOLD - declare --change-class before submission"
        submit_eligible = False

    print(f"candidate : {args.label or args.predictions.name}")
    for lang, m in per.items():
        print(f"  {lang}: n={m['n']:4d}  WER {m['wer']:.4f}  CER {m['cer']:.4f}")
    print(f"change class: {args.change_class}")
    print(f"bench score : {score:.6f}   baseline {baseline:.6f}   delta {delta:+.6f}")
    print(f"baseline src: {baseline_source}")
    print(f"VERDICT     : {verdict}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label or args.predictions.name,
            "predictions": str(args.predictions),
            "casing_applied": not args.no_casing,
            "change_class": args.change_class,
            "per_language": per,
            "bench_score": round(score, 6),
            "baseline": round(baseline, 6),
            "baseline_source": baseline_source,
            "delta": round(delta, 6),
            "verdict": verdict,
            "submit_eligible": submit_eligible,
        }) + "\n")
    print(f"logged to {args.log}")


if __name__ == "__main__":
    main()
