#!/usr/bin/env python3
"""Rank Phase-2 test rows for native-speaker review and emit a decision sheet.

Ranking uses decoder instability as the proxy for "this row is probably wrong":
  risk      = mean normalized distance from the submitted text to its own 8
              stochastic samples (high = the decoder is unsure)
  spread    = number of distinct samples (8 = every sample differs)
  af51_dist = distance to the independent af51 engine (high = engines disagree)
Expected loss is risk x words, because a shaky 40-word row costs far more than a
shaky 5-word row.

Emits one row per clip with the current text and three real alternatives, plus a
`choice` column for the reviewer: keep | alt1 | alt2 | alt3 | or paste corrected
text into `corrected_text`.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rapidfuzz.distance import Levenshtein as _Lev
except ImportError:  # pragma: no cover
    _Lev = None

EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def dist(a: str, b: str) -> float:
    aw, bw = a.split(), b.split()
    if _Lev is not None:
        w = _Lev.distance(aw, bw) / max(len(aw), len(bw), 1)
        c = _Lev.distance(a, b) / max(len(a), len(b), 1)
    else:
        w = c = 0.0
    return 0.5 * w + 0.5 * c


def max_repeat(text: str, order: int = 4) -> int:
    t = [EDGE.sub("", w.lower()) for w in text.split()]
    t = [w for w in t if w]
    grams = [tuple(t[i : i + order]) for i in range(len(t) - order + 1)]
    return max(Counter(grams).values(), default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True, help="n-best/sampled CSV: ID,rank,Target,...")
    parser.add_argument("--fallback", type=Path, required=True, help="Independent engine predictions (af51).")
    parser.add_argument("--routing", type=Path, required=True, help="ID,language table.")
    parser.add_argument("--durations", type=Path, default=None, help="Optional ID,duration CSV.")
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cur = {r["ID"]: r["Target"] for r in csv.DictReader(args.submission.open(encoding="utf-8-sig"))}
    fb = {r["ID"]: r["Target"] for r in csv.DictReader(args.fallback.open(encoding="utf-8-sig"))}
    route = {r["ID"]: r["language"] for r in csv.DictReader(args.routing.open(encoding="utf-8-sig"))}
    dur = {}
    if args.durations and args.durations.exists():
        dur = {r["ID"]: float(r["duration"]) for r in csv.DictReader(args.durations.open(encoding="utf-8-sig"))}

    samples: dict[str, list[str]] = defaultdict(list)
    for r in csv.DictReader(args.samples.open(encoding="utf-8-sig")):
        samples[r["ID"]].append(r["Target"])

    rows = []
    for example_id, text in cur.items():
        smp = samples.get(example_id, [])
        risk = sum(dist(text, s) for s in smp) / len(smp) if smp else 0.0
        spread = len(set(smp))
        # most central sample = lowest mean distance to the other samples
        alt1 = ""
        if smp:
            alt1 = min(smp, key=lambda s: sum(dist(s, o) for o in smp))
        alt3 = max(smp, key=len) if smp else ""
        fb_text = fb.get(example_id, "")
        words = len(text.split())
        d = dur.get(example_id, 0.0)
        density = len(text) / d if d else 0.0
        flags = []
        if risk >= 0.25: flags.append("unstable-decode")
        if spread >= 7: flags.append("all-samples-differ")
        if fb_text and dist(text, fb_text) >= 0.55: flags.append("engines-disagree")
        if max_repeat(text) >= 3: flags.append("repetition")
        if d and density < 4.5: flags.append(f"low-density {density:.1f}c/s")
        if alt3 and len(alt3) > 1.5 * max(len(text), 1): flags.append("sample-much-longer")
        rows.append(
            {
                "expected_loss": round(risk * words, 2),
                "ID": example_id,
                "language": route.get(example_id, "?"),
                "audio_file": f"{example_id}.wav",
                "duration_s": round(d, 1),
                "risk": round(risk, 3),
                "distinct_samples": spread,
                "flags": "; ".join(flags),
                "current_text": text,
                "alt1_central_sample": alt1 if alt1 != text else "",
                "alt2_af51": fb_text,
                "alt3_longest_sample": alt3 if alt3 not in (text, alt1) else "",
                "choice": "",
                "corrected_text": "",
            }
        )

    rows.sort(key=lambda r: -r["expected_loss"])
    top = rows[: args.top]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys()))
        w.writeheader()
        w.writerows(top)

    by_lang = Counter(r["language"] for r in top)
    print(f"wrote {len(top)} review rows -> {args.output}")
    print("language mix:", dict(by_lang.most_common()))
    print("expected_loss range:", top[0]["expected_loss"], "->", top[-1]["expected_loss"])
    print("flag counts:", Counter(f for r in top for f in r["flags"].split("; ") if f).most_common(8))


if __name__ == "__main__":
    main()
