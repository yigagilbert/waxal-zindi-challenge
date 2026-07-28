#!/usr/bin/env python3
"""Cluster Phase-2 clips by language using lexical markers in (good) ASR transcripts.

Text-based LID on af51's fluent transcripts: each Phase-2 language has distinctive
function words / orthographic signatures. Outputs per-ID language assignments (for
engine routing) plus a histogram and low-confidence list.

Clusters: ach = Acholi/Lango (Luo, Nilotic) · nyn = Runyankole-Rukiga ·
xog = Lusoga · myx = Lumasaba/Lugisu · unk = low-confidence.

Usage:
  python scripts/cluster_phase2_languages.py \
    --predictions outputs/predictions/phase2_af51_beam5_raw.csv \
    --output outputs/analysis/phase2_language_clusters.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

MARKERS = {
    "ach": {  # Acholi/Lango (Luo): non-Bantu function words
        "tye", "atye", "gitye", "kun", "dok", "woko", "dano", "latin", "obedo",
        "aneno", "adada", "madit", "matidi", "acel", "aryo", "ngat", "ento",
        "piny", "gin", "kome", "komgi", "kacel", "kiketo", "apura", "gang",
        "myel", "lum", "wiye", "cinge", "bongo", "kingi", "malac", "maleng", "pii",
    },
    "nyn": {  # Runyankole-Rukiga
        "aha", "kandi", "hariho", "harimu", "ariyo", "kwonka", "nari",
        "omushaija", "abashaija", "omukazi", "abakazi", "amaizi", "rubaju",
        "enyuma", "nindeeba", "omuri", "eine", "erikwera", "ndikubona", "obwato",
        "ekishushani", "aineho", "arimu", "bariyo", "nibakoresa", "ahandikireho",
    },
    "xog": {  # Lusoga
        "nenga", "gundi", "umulala", "ndala", "zene", "kwene", "budambi",
        "mwene", "zinzu", "zingubo", "zisuubo", "ngaji", "idani", "baseza",
        "matafali", "masanyalaze", "kisaala", "bisaala", "mukana", "liguje",
    },
    "myx": {  # Lumasaba/Lugisu — plus the kh/ts orthographic signature below
        "bakhasi", "abandu", "umundu", "kameetsi", "kametsi", "umusaani",
        "tsingubo", "tsikhu", "lugudo", "khundulo", "bibyambako", "likholelo",
        "angolobe", "khayuni", "imbata", "kamatoore", "ikofiira",
    },
}


def classify(text: str) -> tuple[str, dict[str, float], float]:
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return "unk", {}, 0.0
    wordset = Counter(words)
    scores: dict[str, float] = {}
    for lang, markers in MARKERS.items():
        scores[lang] = sum(count for w, count in wordset.items() if w in markers)
    # orthographic signature for Lumasaba: kh / ts(i) sequences are pervasive
    low = text.lower()
    scores["myx"] += 0.6 * low.count("kh") + 0.4 * low.count("tsi")
    # Lusoga: zi- noun-class prefix frequency
    scores["xog"] += 0.5 * sum(1 for w in words if w.startswith("zi") and len(w) > 3)
    total = sum(scores.values())
    if total == 0:
        return "unk", scores, 0.0
    best = max(scores, key=scores.get)
    confidence = scores[best] / total
    n_norm = scores[best] / max(len(words), 1)
    if scores[best] < 2 or (confidence < 0.45 and n_norm < 0.08):
        return "unk", scores, confidence
    return best, scores, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="ID,Target CSV of transcripts.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out_rows, hist = [], Counter()
    low_conf = []
    for r in rows:
        lang, scores, conf = classify(r["Target"])
        hist[lang] += 1
        out_rows.append({"ID": r["ID"], "language": lang, "confidence": round(conf, 3)})
        if lang == "unk":
            low_conf.append(r["ID"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "language", "confidence"])
        writer.writeheader()
        writer.writerows(out_rows)

    print("cluster histogram:", dict(hist.most_common()))
    print(f"unk (low-confidence): {len(low_conf)} — first: {low_conf[:10]}")
    print(f"Wrote {args.output}")
    report = {"histogram": dict(hist), "num_unk": len(low_conf), "output": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
