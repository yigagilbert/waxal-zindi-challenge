#!/usr/bin/env python3
"""Pre-submission scorer: predict a candidate's leaderboard score before spending a submission.

Any change must first be produced on the labeled validation bench, scored here,
and only promoted to a test submission if it beats the baseline. This reproduces
the leaderboard metric exactly as observed:

  * WER is CASE-INSENSITIVE   (two submissions differing only in casing returned
                               byte-identical WER 0.413864714)
  * CER is CASE-SENSITIVE     (the same pair moved CER 0.137837 -> 0.131082)
  * score = 1 - (WER + CER) / 2, languages weighted by their test counts

Calibration against real submissions (bench score -> public score):

  champion, no casing     0.8779 -> 0.724149
  champion + casing       0.8814 -> 0.727527      (delta -0.0035 predicted, -0.0034 actual)
  champion, alpha 1.1/1.0 0.8727 -> 0.689930      (delta -0.0087 predicted, -0.0376 actual)

Read: the bench is DIRECTIONALLY reliable in every case so far, exact for
formatting changes, and conservative in magnitude for decode changes (a bench
regression showed up ~4x larger on the leaderboard). Rule of thumb: never submit
a candidate that loses on the bench; treat bench gains as a lower bound.

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

BASELINE = 0.8814  # champion + casing on the 800-clip lin/sna bench
TEST_WEIGHTS = {"lin": 448.0, "sna": 444.0}
_AFTER_SENTENCE_END = re.compile(r"([.!?]['\"\)\]]?\s+)([a-z])")


def restore_casing(text: str) -> str:
    if not text:
        return text
    out = text[:1].upper() + text[1:]
    return _AFTER_SENTENCE_END.sub(lambda m: m.group(1) + m.group(2).upper(), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bench", type=Path, default=Path("outputs/day4_h100/bench_ids.csv"))
    parser.add_argument("--label", default=None, help="Name for the decision log.")
    parser.add_argument("--baseline", type=float, default=BASELINE)
    parser.add_argument("--no-casing", action="store_true", help="Score as-is instead of applying casing restoration.")
    parser.add_argument("--log", type=Path, default=Path("outputs/day4_h100/CANDIDATES.jsonl"))
    args = parser.parse_args()

    bench = {
        r["ID"]: (r["language"], r["Target"])
        for r in csv.DictReader(args.bench.open(encoding="utf-8-sig"))
    }
    pred = {r["ID"]: r["Target"] for r in csv.DictReader(args.predictions.open(encoding="utf-8-sig"))}
    transform = (lambda t: t) if args.no_casing else restore_casing

    per = {}
    for lang in sorted(TEST_WEIGHTS):
        ids = [k for k, (l, _) in bench.items() if l == lang and k in pred]
        if not ids:
            continue
        we = rw = ce = rc = 0
        for k in ids:
            ref = bench[k][1]
            hyp = transform(pred[k])
            we += L.distance(ref.lower().split(), hyp.lower().split()); rw += len(ref.split())
            ce += L.distance(ref, hyp); rc += len(ref)
        per[lang] = {"n": len(ids), "wer": we / max(rw, 1), "cer": ce / max(rc, 1)}

    missing = [k for k in bench if k not in pred]
    denom = sum(TEST_WEIGHTS[l] for l in per)
    combined = sum((0.5 * per[l]["wer"] + 0.5 * per[l]["cer"]) * TEST_WEIGHTS[l] for l in per) / denom
    score = 1 - combined
    delta = score - args.baseline
    verdict = "PASS - submit" if delta > 0 else "FAIL - do not submit"

    print(f"candidate : {args.label or args.predictions.name}")
    if missing:
        print(f"WARNING   : {len(missing)} bench clips missing from the candidate")
    for lang, m in per.items():
        print(f"  {lang}: n={m['n']:4d}  WER {m['wer']:.4f}  CER {m['cer']:.4f}")
    print(f"bench score : {score:.4f}   baseline {args.baseline:.4f}   delta {delta:+.4f}")
    print(f"VERDICT     : {verdict}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label or args.predictions.name,
            "predictions": str(args.predictions),
            "casing_applied": not args.no_casing,
            "per_language": per,
            "bench_score": round(score, 6),
            "baseline": args.baseline,
            "delta": round(delta, 6),
            "verdict": verdict,
        }) + "\n")
    print(f"logged to {args.log}")


if __name__ == "__main__":
    main()
