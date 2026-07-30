#!/usr/bin/env python3
"""Pivot-based weighted ROVER: word-level voting across prediction CSVs.

The FIRST input is the pivot (your best system). Every other hypothesis is
aligned to the pivot word-by-word; a pivot word is replaced (or dropped) only
when a weighted majority of systems agrees on the alternative. Insertions are
never adopted — the combiner is deliberately conservative.

Votes are keyed on lowercased, punctuation-stripped word forms so that casing
and attached punctuation don't split votes; the emitted surface form comes from
the highest-weight system that voted for the winner (pivot preferred on ties).

Usage (bench gate first, then test):
  python scripts/rover_ensemble.py \
    --inputs outputs/predictions/salt_val_b50.8.csv:1.0 \
             outputs/predictions/salt_val_base.csv:0.9 \
             outputs/predictions/salt_val_adapter.csv:0.85 \
             outputs/predictions/s51_val_forced.csv:0.6 \
    --output outputs/predictions/salt_val_rover.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
from collections import defaultdict
from pathlib import Path

_STRIP = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def key(tok: str) -> str:
    return _STRIP.sub("", tok.lower())


def combine(hyps: list[tuple[list[str], float]]) -> list[str]:
    pivot_tokens, _ = hyps[0]
    n = len(pivot_tokens)
    votes: list[dict[str, float]] = [defaultdict(float) for _ in range(n)]
    surface: list[dict[str, tuple[float, str]]] = [dict() for _ in range(n)]
    pivot_keys = [key(t) for t in pivot_tokens]

    for rank, (tokens, w) in enumerate(hyps):
        tok_keys = [key(t) for t in tokens]
        sm = difflib.SequenceMatcher(None, pivot_keys, tok_keys)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace") and (tag == "equal" or (i2 - i1) == (j2 - j1)):
                for k in range(i2 - i1):
                    pos, tok = i1 + k, tokens[j1 + k]
                    kk = tok_keys[j1 + k]
                    if not kk:
                        continue
                    votes[pos][kk] += w
                    prev = surface[pos].get(kk)
                    # prefer pivot's surface form, else highest weight, stable on ties
                    eff = w + (100.0 if rank == 0 else 0.0)
                    if prev is None or eff > prev[0]:
                        surface[pos][kk] = (eff, tok)
            elif tag == "delete":
                for pos in range(i1, i2):
                    votes[pos][""] += w

    out: list[str] = []
    for pos, tok in enumerate(pivot_tokens):
        pk = pivot_keys[pos]
        if not votes[pos]:
            out.append(tok)
            continue
        best_key, best_w = max(votes[pos].items(), key=lambda kv: (kv[1], kv[0] == pk))
        pivot_w = votes[pos].get(pk, 0.0)
        if best_key != pk and best_w > pivot_w + 1e-9:
            if best_key:  # replacement won
                out.append(surface[pos][best_key][1])
            # else: weighted deletion won -> drop the word
        else:
            out.append(tok)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="path:weight (first = pivot).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    systems: list[tuple[dict[str, str], float, str]] = []
    for spec in args.inputs:
        path, _, w = spec.rpartition(":")
        rows = {r["ID"]: r["Target"] for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
        systems.append((rows, float(w), path))
        print(f"loaded {len(rows):5d} rows  w={w}  {path}")

    common = set(systems[0][0])
    for rows, _, path in systems[1:]:
        missing = common - set(rows)
        if missing:
            print(f"WARNING: {len(missing)} pivot IDs missing from {path}; pivot kept for those")

    changed = 0
    out_rows = []
    for example_id, pivot_text in systems[0][0].items():
        hyps = [(pivot_text.split(), systems[0][1])]
        for rows, w, _ in systems[1:]:
            if example_id in rows:
                hyps.append((rows[example_id].split(), w))
        merged = " ".join(combine(hyps)) if len(hyps) > 1 else pivot_text
        changed += merged != pivot_text
        out_rows.append({"ID": example_id, "Target": merged})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Target"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"rows changed vs pivot: {changed}/{len(out_rows)} -> {args.output}")


if __name__ == "__main__":
    main()
