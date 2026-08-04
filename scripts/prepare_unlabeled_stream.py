#!/usr/bin/env python3
"""Stream WaxalNLP *unlabeled* lin/sna audio to FLAC without caching parquets.

The unlabeled splits are official challenge data (40 lin / 52 sna shards of new
speakers) and the only untapped in-domain resource for unseen-speaker
generalization — exactly the measured weakness (bench 0.882 on seen speakers vs
0.729 public on unseen). Audio is streamed (no ~40 GB parquet cache), resampled
to 16 kHz mono, and stored as 16-bit FLAC (~50% of WAV size).

Usage:
  python scripts/prepare_unlabeled_stream.py --languages lin sna \
    --per-language 12000 --output-dir data/unlabeled_linsna
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["lin", "sna"])
    parser.add_argument("--per-language", type=int, default=12000)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=35.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/unlabeled_linsna"))
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    seen: set[str] = set()
    if manifest_path.exists():  # resumable
        with manifest_path.open(encoding="utf-8") as f:
            seen = {row["ID"] for row in csv.DictReader(f)}
        print(f"resuming: {len(seen)} clips already on disk")

    mode = "a" if seen else "w"
    with manifest_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "path", "language", "duration"])
        if mode == "w":
            writer.writeheader()
        for lang in args.languages:
            kept = sum(1 for _ in ()) or len([s for s in seen if s.startswith(f"{lang}_")])
            if kept >= args.per_language:
                print(f"{lang}: already have {kept}, skipping")
                continue
            ds = load_dataset(
                "google/WaxalNLP",
                data_files=f"data/ASR/{lang}/{lang}-unlabeled-*.parquet",
                split="train",
                streaming=True,
            ).cast_column("audio", Audio(sampling_rate=16_000))
            skipped = 0
            for row in ds:
                if kept >= args.per_language:
                    break
                clip_id = str(row.get("id") or row.get("ID"))
                if clip_id in seen:
                    continue
                audio = row["audio"]
                array = np.asarray(audio["array"], dtype=np.float32)
                duration = len(array) / 16_000
                if not (args.min_duration <= duration <= args.max_duration):
                    skipped += 1
                    continue
                out = audio_dir / f"{clip_id}.flac"
                sf.write(out, array, 16_000, subtype="PCM_16", format="FLAC")
                writer.writerow({"ID": clip_id, "path": str(out), "language": lang,
                                 "duration": round(duration, 2)})
                seen.add(clip_id)
                kept += 1
                if kept % 500 == 0:
                    f.flush()
                    print(f"{lang}: {kept}/{args.per_language} (skipped {skipped})", flush=True)
            print(f"{lang}: DONE {kept} clips (skipped {skipped})", flush=True)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
