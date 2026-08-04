#!/usr/bin/env python3
"""Prepare a small speaker-diverse AfriVoices acoustic OOD gate.

The converted ``evie-8/afrivoices`` corpus derives from the CC-BY-4.0
DigitalUmuganda/Afrivoice source and restores the missing Shona audio shards.
The champion used AfriVoices *text* in its KenLM, but not this audio. Therefore
use greedy metrics as the acoustic promotion gate; LM-routed metrics are only a
secondary diagnostic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


LANGUAGES = ("lin", "sna")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/afrivoices_ood"))
    parser.add_argument("--max-per-language", type=int, default=300)
    parser.add_argument("--max-per-speaker", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=35.0)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset

    parts = []
    language_reports = {}
    for language in LANGUAGES:
        stream = load_dataset(
            "evie-8/afrivoices", language, split="train", streaming=True
        ).cast_column("audio", Audio(sampling_rate=16_000))
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        speaker_counts: Counter[str] = Counter()
        rows = []
        scanned = 0
        for example in stream:
            scanned += 1
            text = str(example.get("text") or "").strip()
            speaker_id = str(example.get("speaker_id") or "").strip()
            duration = float(example.get("duration") or 0.0)
            if not text or not speaker_id:
                continue
            if not (args.min_duration <= duration <= args.max_duration):
                continue
            if speaker_counts[speaker_id] >= args.max_per_speaker:
                continue
            source_id = str(example.get("id") or f"row_{scanned}")
            rows.append(
                {
                    "ID": f"afrivoices_{language}_{source_id}",
                    "audio": example["audio"],
                    "language": language,
                    "transcription": text,
                    "speaker_id": speaker_id,
                    "duration": duration,
                    "original_split": "afrivoices_train_ood_gate",
                    "source_dataset": "evie-8/afrivoices",
                }
            )
            speaker_counts[speaker_id] += 1
            if len(rows) >= args.max_per_language:
                break
        if len(rows) < args.max_per_language:
            raise RuntimeError(
                f"{language}: collected only {len(rows)}/{args.max_per_language} rows "
                f"after scanning {scanned}; relax --max-per-speaker or duration limits"
            )
        parts.append(Dataset.from_list(rows))
        language_reports[language] = {
            "saved_rows": len(rows),
            "speakers": len(speaker_counts),
            "scanned_rows": scanned,
            "duration_hours": sum(row["duration"] for row in rows) / 3600.0,
            "max_rows_per_speaker": max(speaker_counts.values()),
        }

    dataset = DatasetDict({"validation": concatenate_datasets(parts)})
    dataset_path = args.output_dir / "hf_dataset"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(dataset_path)
    report = {
        "source": "evie-8/afrivoices",
        "upstream_source": "DigitalUmuganda/Afrivoice",
        "upstream_license": "CC-BY-4.0",
        "usage": "evaluation-only acoustic OOD gate; promote on greedy metrics",
        "text_leakage_note": "Champion KenLM used AfriVoices text; do not use LM-routed score as the primary acoustic gate.",
        "languages": language_reports,
        "dataset_path": str(dataset_path),
    }
    (args.output_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
