#!/usr/bin/env python3
"""Determine which text normalization the leaderboard metric applies.

Evidence from two Phase-2 submissions differing only by restored casing:

    submission          WER            CER
    lowercase        0.413864714    0.137836973
    cased            0.413864714    0.131081513

WER is byte-identical, CER changed -> the grader lowercases for WER but not for
CER. This script measures the same contrasts locally on labeled validation
predictions so the remaining question -- whether WER also strips punctuation --
can be answered without spending a submission.

Usage:
  python scripts/probe_metric_normalization.py \
    --predictions outputs/day4_h100/bench_champion_val.csv \
    --references data/processed/validation.csv --languages lin sna
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein as L


def strip_punct(text: str) -> str:
    out = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", out).strip()


VARIANTS = {
    "raw": lambda t: t,
    "lower": lambda t: t.lower(),
    "no_punct": strip_punct,
    "lower+no_punct": lambda t: strip_punct(t.lower()),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=["lin", "sna"])
    parser.add_argument("--weights", nargs="*", default=["lin=448", "sna=444"])
    args = parser.parse_args()

    weights = {}
    for spec in args.weights:
        lang, _, count = spec.partition("=")
        weights[lang] = float(count)

    refs = {
        r["ID"]: (r["language"], r["Target"])
        for r in csv.DictReader(args.references.open(encoding="utf-8-sig"))
        if r["language"] in args.languages
    }
    hyps = {r["ID"]: r["Target"] for r in csv.DictReader(args.predictions.open(encoding="utf-8-sig"))}
    common = [k for k in refs if k in hyps]
    print(f"scored clips: {len(common)}")

    print(f"{'variant':<18}" + "".join(f"{l+' WER':>11}{l+' CER':>11}" for l in args.languages) + f"{'TEST-W':>10}")
    for name, fn in VARIANTS.items():
        per = {}
        for lang in args.languages:
            ids = [k for k in common if refs[k][0] == lang]
            we = rw = ce = rc = 0
            for k in ids:
                ref, hyp = fn(refs[k][1]), fn(hyps[k])
                we += L.distance(ref.split(), hyp.split()); rw += len(ref.split())
                ce += L.distance(ref, hyp); rc += len(ref)
            per[lang] = (we / max(rw, 1), ce / max(rc, 1))
        tw = sum((0.5 * per[l][0] + 0.5 * per[l][1]) * weights.get(l, 1.0) for l in per)
        tw /= sum(weights.get(l, 1.0) for l in per)
        row = "".join(f"{per[l][0]:11.4f}{per[l][1]:11.4f}" for l in args.languages)
        print(f"{name:<18}{row}{tw:10.4f}")

    print("\nLeaderboard reference points (corrected Phase-2 test set):")
    print("  lowercase submission : WER 0.413865  CER 0.137837  score 0.724149")
    print("  cased submission     : WER 0.413865  CER 0.131082  score 0.727527")
    print("  => WER is case-insensitive; CER is case-sensitive.")


if __name__ == "__main__":
    main()
