#!/usr/bin/env python3
"""Standardize the Luganda cv-yogera-filtered parquet into the generalization-mix schema.

`yigagilbert/luganda-speech-cv-yogera-filtered` is a gated parquet dataset with two
subsets (lug_commonvoice/*.parquet, lug_makerereradio/*.parquet), each carrying an
audio column (HF Audio feature = inline bytes) and a `text` transcript column. This
emits rows with the SAME 11-column schema as standardize_afrivoice_audio.py /
prepare_generalization_mix, so all three concatenate cleanly for clean_audio_v3.

Torchcodec-free: casts audio to Audio(decode=False) and decodes the raw bytes with
soundfile (+ librosa resample), so it is robust to the box's datasets/torchcodec churn
(same approach that worked for Afrivoice). Resumable: each subset is saved to
shards/<config> and skipped on re-run; the final combine step reuses whatever is saved.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import ensure_dir, json_dump  # noqa: E402

# Same column schema as prepare_generalization_mix / standardize_afrivoice_audio.
COLUMNS = [
    "ID", "audio", "transcription", "language", "original_split", "duration",
    "source_dataset", "source_split", "source_language", "source_license", "source_risk",
]

# Subsets of the gated repo. license/risk mirror docs/RULES_AND_DATA_USE.md wording.
SUBSETS = [
    {"config": "lug_commonvoice", "data_files": "lug_commonvoice/*.parquet",
     "license": "CC0-1.0 (Common Voice)", "risk": "external_cc0_commonvoice_read_speech"},
    {"config": "lug_makerereradio", "data_files": "lug_makerereradio/*.parquet",
     "license": "CC-BY-SA-4.0 (Makerere Radio)", "risk": "external_ccbysa_makerere_radio_speech"},
]

TEXT_COLS = ("transcription", "text", "sentence", "Target", "transcript", "normalized_text", "raw_transcription")
AUDIO_COLS = ("audio", "audio_filepath", "path", "wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="yigagilbert/luganda-speech-cv-yogera-filtered")
    parser.add_argument("--language", default="lug")
    parser.add_argument("--subsets", nargs="*", default=[s["config"] for s in SUBSETS],
                        help="Which subset configs to process (default: both).")
    parser.add_argument("--output-dir", type=Path, default=Path("data/luganda_standardized"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--max-rows-per-subset", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _pick(colnames, candidates):
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in colnames:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _decode(value, target_sr, librosa):
    """Return (float32 mono array @ target_sr, sr) or (None, None). Torchcodec-free."""
    import numpy as np
    import soundfile as sf

    array = sr = None
    if isinstance(value, dict):
        if value.get("array") is not None:
            array = np.asarray(value["array"], dtype=np.float32)
            sr = int(value.get("sampling_rate") or target_sr)
        elif value.get("bytes"):
            array, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
        elif value.get("path"):
            array, sr = sf.read(value["path"], dtype="float32", always_2d=False)
    elif isinstance(value, (bytes, bytearray)):
        array, sr = sf.read(io.BytesIO(value), dtype="float32", always_2d=False)
    elif isinstance(value, str):
        array, sr = sf.read(value, dtype="float32", always_2d=False)
    if array is None:
        return None, None
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    array = np.asarray(array, dtype=np.float32)
    if sr != target_sr:
        if librosa is None:
            return None, None
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.asarray(array, dtype=np.float32), sr


def process_subset(repo: str, subset: dict, args: argparse.Namespace):
    from datasets import Audio, load_dataset

    ds = load_dataset(repo, data_files=subset["data_files"], split="train")
    if args.max_rows_per_subset and len(ds) > args.max_rows_per_subset:
        ds = ds.select(range(args.max_rows_per_subset))
    audio_col = _pick(ds.column_names, AUDIO_COLS)
    text_col = _pick(ds.column_names, TEXT_COLS)
    if audio_col is None or text_col is None:
        raise SystemExit(f"  {subset['config']}: could not find audio/text columns in {ds.column_names}")
    print(f"  {subset['config']}: {len(ds)} rows | audio='{audio_col}' text='{text_col}'")
    # Get raw bytes without triggering torchcodec decoding.
    try:
        ds = ds.cast_column(audio_col, Audio(decode=False))
    except Exception:
        pass  # already a path/bytes column

    config = subset["config"]

    def _map(batch, idxs):
        try:
            import librosa
        except ImportError:
            librosa = None
        out = {c: [] for c in COLUMNS}
        for value, text_raw, idx in zip(batch[audio_col], batch[text_col], idxs, strict=True):
            text = normalize_text(str(text_raw or ""), args.normalization)
            if not text:
                continue
            array, sr = _decode(value, args.sample_rate, librosa)
            if array is None:
                continue
            out["ID"].append(f"luganda_{config}_{idx}")
            out["audio"].append({"array": array, "sampling_rate": sr})
            out["transcription"].append(text)
            out["language"].append(args.language)
            out["original_split"].append("external_train")
            out["duration"].append(round(len(array) / sr, 3) if sr else -1.0)
            out["source_dataset"].append(args.repo)
            out["source_split"].append(config)
            out["source_language"].append(args.language)
            out["source_license"].append(subset["license"])
            out["source_risk"].append(subset["risk"])
        return out

    mapped = ds.map(
        _map,
        batched=True,
        batch_size=32,
        with_indices=True,
        num_proc=args.num_proc if len(ds) > 512 else None,
        remove_columns=ds.column_names,
        desc=f"Standardizing {config}",
    )
    mapped = mapped.cast_column("audio", Audio(sampling_rate=args.sample_rate))
    return mapped


def main() -> None:
    args = parse_args()
    from datasets import concatenate_datasets, load_from_disk

    ensure_dir(args.output_dir)
    shards_dir = ensure_dir(args.output_dir / "shards")
    stats: dict = {}
    wanted = {s["config"]: s for s in SUBSETS}
    for config in args.subsets:
        subset = wanted.get(config) or {"config": config, "data_files": f"{config}/*.parquet",
                                         "license": "unknown", "risk": "external_unknown"}
        print(f"== {config} ({args.language}) ==")
        subset_out = shards_dir / config
        if subset_out.exists():
            n = len(load_from_disk(str(subset_out)))
            print(f"  skip (already done): {config} ({n} rows)")
            stats[config] = {"rows": n, "status": "reused"}
            continue
        mapped = process_subset(args.repo, subset, args)
        mapped.save_to_disk(subset_out)
        stats[config] = {"rows": len(mapped), "status": "processed"}
        print(f"  [{config}] saved {len(mapped)} rows")

    subset_dirs = [d for d in sorted(shards_dir.iterdir()) if (d / "dataset_info.json").exists()]
    if not subset_dirs:
        raise SystemExit("No subsets saved yet; check gated access + login, then re-run.")
    parts = [load_from_disk(str(d)) for d in subset_dirs]
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    out = args.output_dir / "hf_dataset"
    dataset.save_to_disk(out)
    summary = {
        "repo": args.repo, "subsets": args.subsets, "sample_rate": args.sample_rate,
        "rows": len(dataset), "subset_datasets_combined": len(subset_dirs), "by_subset": stats,
        "by_language": dict(sorted(Counter(dataset["language"]).items())), "output": str(out),
    }
    json_dump(summary, args.report or (args.output_dir / "luganda_standardize_report.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved standardized Luganda to {out} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
