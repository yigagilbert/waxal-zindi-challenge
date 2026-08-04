#!/usr/bin/env python3
"""Filter pseudo-labels by teacher self-agreement and attach them as a dataset split.

Confidence signal: character disagreement between the champion's greedy and
beam+LM transcripts of the same clip (reference-free). Low disagreement means
the acoustic evidence dominates and the label is trustworthy; high disagreement
means the LM had to guess. Additional sanity: plausible character rate and a
minimum word count.

Kept rows are written as a `pseudo` split INSIDE the existing
`data/processed/hf_dataset` (schema-matched, FLAC bytes embedded), so training
can consume them via `data.extra_train_splits: [pseudo]` with no copying of the
labeled data.

Usage:
  python scripts/filter_pseudo_labels.py \
    --pseudo outputs/day4_h100/pseudo_labels_raw.csv \
    --dataset-dir data/processed --max-disagreement 0.10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-disagreement", type=float, default=0.10)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--min-chars-per-sec", type=float, default=4.0)
    parser.add_argument("--max-chars-per-sec", type=float, default=25.0)
    parser.add_argument("--report", type=Path, default=Path("outputs/day4_h100/pseudo_filter_report.json"))
    args = parser.parse_args()

    from datasets import Audio, Dataset, load_from_disk
    from rapidfuzz.distance import Levenshtein as L

    rows = list(csv.DictReader(args.pseudo.open(encoding="utf-8")))
    stats = {"total": len(rows), "kept": 0, "by_language": {}, "dropped": {"disagreement": 0, "rate": 0, "short": 0, "empty": 0}}
    deciles = [0] * 10
    kept = []
    for r in rows:
        g, b = r["greedy"].strip().lower(), r["beam"].strip().lower()
        if not b:
            stats["dropped"]["empty"] += 1
            continue
        disagreement = L.distance(g, b) / max(len(b), 1)
        deciles[min(int(disagreement * 10), 9)] += 1
        duration = float(r["duration"])
        rate = len(b) / max(duration, 0.1)
        if disagreement > args.max_disagreement:
            stats["dropped"]["disagreement"] += 1
        elif not (args.min_chars_per_sec <= rate <= args.max_chars_per_sec):
            stats["dropped"]["rate"] += 1
        elif len(b.split()) < args.min_words:
            stats["dropped"]["short"] += 1
        else:
            kept.append(r)
            lang = r["language"]
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
    stats["kept"] = len(kept)
    stats["disagreement_deciles"] = {f"{i/10:.1f}-{(i+1)/10:.1f}": n for i, n in enumerate(deciles)}
    print(json.dumps(stats, indent=2))

    dataset_dict = load_from_disk(args.dataset_dir / "hf_dataset")
    train_cols = dataset_dict["train"].column_names
    print("target schema:", train_cols)
    manifest = {r2["ID"]: r2["path"] for r2 in csv.DictReader(open("data/unlabeled_linsna/manifest.csv", encoding="utf-8"))}
    data: dict[str, list] = {c: [] for c in train_cols}
    for r in kept:
        for c in train_cols:
            if c == "ID":
                data[c].append(r["ID"])
            elif c == "audio":
                data[c].append(manifest[r["ID"]])
            elif c == "transcription":
                data[c].append(r["beam"])
            elif c == "language":
                data[c].append(r["language"])
            elif c == "duration":
                data[c].append(float(r["duration"]))
            elif c == "original_split":
                data[c].append("unlabeled_pseudo")
            else:
                data[c].append("")
    pseudo = Dataset.from_dict(data).cast_column("audio", Audio(sampling_rate=16_000))
    out_dir = args.dataset_dir / "hf_dataset" / "pseudo"
    pseudo.save_to_disk(str(out_dir))

    meta_path = args.dataset_dir / "hf_dataset" / "dataset_dict.json"
    meta = json.loads(meta_path.read_text())
    if "pseudo" not in meta["splits"]:
        meta["splits"].append("pseudo")
        meta_path.write_text(json.dumps(meta))
    print(f"pseudo split saved: {len(pseudo)} rows -> {out_dir}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
