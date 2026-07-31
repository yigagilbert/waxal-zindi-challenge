#!/usr/bin/env python3
"""Score prediction CSVs under the Phase-2 TEST language distribution.

The official validation split is xog 841 / myx 849 / nyn 831 / ach 519, but the
Phase-2 test routing is ach 500 / nyn 500 / myx 499 / xog 1. A 4-language macro
therefore gives xog 25% of the gate weight while it is 0.07% of the test, and
under-weights myx (the weakest language, a third of the test).

This scorer reports, per candidate: per-language error, the legacy 4-language
macro, and the test-weighted score = mean over the languages that actually occur
in the test routing, weighted by their test counts.

Usage:
  python scripts/score_test_weighted.py --references data/phase2_train/validation.csv \
    --routing outputs/analysis/phase2_lid_fused.csv \
    --candidates anchor=outputs/day2_h100/salt_val_true_forced_full_lp08.csv \
                 t0075=outputs/day2_h100/margin_sweep_validation_0.075.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.scoring import edit_distance  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein as _Lev
except ImportError:  # pragma: no cover
    _Lev = None


def errors(ref: str, hyp: str) -> tuple[int, int, int, int]:
    rw, hw = ref.split(), hyp.split()
    if _Lev is not None:
        we = _Lev.distance(rw, hw)
        ce = _Lev.distance(ref, hyp)
    else:
        we = edit_distance(rw, hw)
        ce = edit_distance(list(ref), list(hyp))
    return we, len(rw), ce, len(ref)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--routing", type=Path, default=None, help="Test ID,language table for weights.")
    parser.add_argument("--weights", nargs="*", default=None, help="lang=count overrides, e.g. ach=500 myx=499.")
    parser.add_argument("--candidates", nargs="+", required=True, help="name=path pairs.")
    args = parser.parse_args()

    refs = {r["ID"]: (r["language"], r["Target"]) for r in csv.DictReader(args.references.open(encoding="utf-8-sig"))}

    if args.weights:
        weights = {}
        for spec in args.weights:
            lang, _, count = spec.partition("=")
            weights[lang] = float(count)
    elif args.routing is not None:
        weights = {
            lang: float(count)
            for lang, count in Counter(
                r["language"] for r in csv.DictReader(args.routing.open(encoding="utf-8-sig"))
            ).items()
        }
    else:
        weights = {lang: 1.0 for lang, _ in refs.values()}
    total_weight = sum(weights.values())
    print("test weights:", {k: round(v / total_weight, 4) for k, v in sorted(weights.items())})

    langs = sorted({lang for lang, _ in refs.values()})
    header = f"{'candidate':<26}" + "".join(f"{l:>9}" for l in langs) + f"{'macro4':>10}{'TEST-W':>10}"
    print(header)
    for spec in args.candidates:
        name, _, path = spec.partition("=")
        hyps = {r["ID"]: r["Target"] for r in csv.DictReader(Path(path).open(encoding="utf-8-sig"))}
        agg: dict[str, list[int]] = {l: [0, 0, 0, 0] for l in langs}
        for example_id, (lang, ref) in refs.items():
            if example_id not in hyps:
                continue
            we, rw, ce, rc = errors(ref, hyps[example_id])
            a = agg[lang]
            a[0] += we; a[1] += rw; a[2] += ce; a[3] += rc
        per_lang = {}
        for lang, (we, rw, ce, rc) in agg.items():
            if rw and rc:
                per_lang[lang] = 0.5 * (we / rw) + 0.5 * (ce / rc)
        covered = [l for l in langs if l in per_lang]
        macro4 = sum(per_lang[l] for l in covered) / max(len(covered), 1)
        tw_num = sum(per_lang[l] * weights.get(l, 0.0) for l in covered)
        tw_den = sum(weights.get(l, 0.0) for l in covered)
        test_weighted = tw_num / tw_den if tw_den else float("nan")
        row = "".join(f"{per_lang.get(l, float('nan')):9.4f}" for l in langs)
        print(f"{name:<26}{row}{macro4:10.4f}{test_weighted:10.4f}")


if __name__ == "__main__":
    main()
