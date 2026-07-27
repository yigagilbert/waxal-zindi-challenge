#!/usr/bin/env python3
"""Ingest the Phase-2 test set (loose ID_XXXXX.wav files, no metadata) into the pipeline.

Phase 2 (dropped 2026-07-27) provides audio.zip (flat wav files named <ID>.wav, IDs like
ID_TPJAR with NO language prefix) plus Test_phase2.csv (the official ID list). This builds
data/processed_phase2/hf_dataset as a DatasetDict({"test": ...}) with ID + audio columns —
exactly what run_no_metadata_pipeline.py consumes (it guards on the absence of
transcription/language columns). Torchcodec-free (soundfile + librosa), datasets 3.x style.

Usage:
  python scripts/prepare_phase2_test.py \
    --audio-dir data/phase2/audio --test-csv data/phase2/Test_phase2.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_csv_dicts  # noqa: E402
from waxal.utils import ensure_dir, json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True, help="Dir containing the unzipped <ID>.wav files.")
    parser.add_argument("--test-csv", type=Path, default=None,
                        help="Test_phase2.csv with an ID column (official ID list + order). "
                             "If omitted, IDs are inferred from the wav filenames (order = sorted).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_phase2"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test cap.")
    return parser.parse_args()


def load_ids(test_csv: Path | None, audio_dir: Path) -> list[str]:
    if test_csv is not None:
        columns, rows, bad = read_csv_dicts(test_csv)
        id_col = "ID" if "ID" in columns else "id" if "id" in columns else None
        if id_col is None:
            raise SystemExit(f"{test_csv} has no ID column; got {columns}")
        if bad:
            raise SystemExit(f"{test_csv} has malformed rows: {bad[:3]}")
        return [row[id_col] for row in rows]
    print("WARNING: no --test-csv; inferring IDs from wav filenames (sorted). "
          "Use the official Test_phase2.csv for submission ordering.")
    return sorted(p.stem for p in audio_dir.glob("*.wav"))


def main() -> None:
    args = parse_args()
    import numpy as np
    import soundfile as sf

    try:
        import librosa
    except ImportError:
        librosa = None

    from datasets import Audio, Dataset, DatasetDict

    ids = load_ids(args.test_csv, args.audio_dir)
    if args.max_samples:
        ids = ids[: args.max_samples]
    print(f"{len(ids)} Phase-2 test IDs")

    rows, missing, durations = [], [], []
    for example_id in ids:
        path = args.audio_dir / f"{example_id}.wav"
        if not path.exists():
            missing.append(example_id)
            continue
        array, sr = sf.read(path, dtype="float32", always_2d=False)
        if getattr(array, "ndim", 1) > 1:
            array = array.mean(axis=1)
        if sr != args.sample_rate:
            if librosa is None:
                raise SystemExit(f"{path} is {sr}Hz; install librosa to resample to {args.sample_rate}")
            array = librosa.resample(np.asarray(array, dtype=np.float32), orig_sr=sr, target_sr=args.sample_rate)
            sr = args.sample_rate
        duration = round(len(array) / sr, 3) if sr else 0.0
        durations.append(duration)
        rows.append({
            "ID": example_id,
            "audio": {"array": np.asarray(array, dtype=np.float32), "sampling_rate": sr},
            "duration": duration,
        })
        if len(rows) % 200 == 0:
            print(f"  loaded {len(rows)}/{len(ids)}", flush=True)

    if missing:
        print(f"WARNING: {len(missing)} IDs have no wav file (first: {missing[:10]})")
    if not rows:
        raise SystemExit("No audio loaded; check --audio-dir")

    extra = sorted(set(p.stem for p in args.audio_dir.glob("*.wav")) - set(ids))
    if extra:
        print(f"NOTE: {len(extra)} wav files not in the ID list (ignored; first: {extra[:5]})")

    ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=args.sample_rate))
    ensure_dir(args.output_dir)
    out = args.output_dir / "hf_dataset"
    DatasetDict({"test": ds}).save_to_disk(out)

    summary = {
        "num_ids": len(ids),
        "num_loaded": len(rows),
        "num_missing_audio": len(missing),
        "missing_audio_ids": missing[:50],
        "num_extra_wavs": len(extra),
        "duration_seconds": {
            "total": round(float(np.sum(durations)), 1),
            "mean": round(float(np.mean(durations)), 2),
            "min": round(float(np.min(durations)), 2),
            "max": round(float(np.max(durations)), 2),
        },
        "output": str(out),
    }
    json_dump(summary, args.output_dir / "phase2_ingest_report.json")
    print(json.dumps(summary, indent=2))
    print(f"Saved Phase-2 test DatasetDict to {out} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
