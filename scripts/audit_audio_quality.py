#!/usr/bin/env python3
"""Audit prepared WAXAL audio quality and text/audio rate statistics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.audio import audio_array_from_example, to_mono  # noqa: E402
from waxal.data import TARGET_LANGUAGES  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


FIELDNAMES = [
    "ID",
    "language",
    "split",
    "duration_seconds",
    "transcript_chars",
    "transcript_words",
    "chars_per_second",
    "words_per_second",
    "audio_rms",
    "peak_amplitude",
    "clipping_ratio",
    "approximate_silence_ratio",
    "is_too_short",
    "is_too_long",
    "is_low_energy",
    "is_clipped",
    "is_text_rate_outlier",
    "quality_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--languages", nargs="*", default=list(TARGET_LANGUAGES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--min-duration", type=float, default=0.30)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--low-rms-threshold", type=float, default=0.003)
    parser.add_argument("--low-peak-threshold", type=float, default=0.02)
    parser.add_argument("--clipping-threshold", type=float, default=0.999)
    parser.add_argument("--clipping-ratio-threshold", type=float, default=0.01)
    parser.add_argument("--min-chars-per-second", type=float, default=1.0)
    parser.add_argument("--max-chars-per-second", type=float, default=35.0)
    parser.add_argument("--min-words-per-second", type=float, default=0.20)
    parser.add_argument("--max-words-per-second", type=float, default=7.0)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def safe_float(value: float | int | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def transcript_from_example(example: dict[str, Any]) -> str:
    return str(example.get("transcription") or example.get("Target") or "")


def frame_silence_ratio(array: Any, sample_rate: int, global_rms: float) -> float:
    """Approximate silence by frame-level RMS, avoiding a hard VAD dependency."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for audio quality auditing.") from exc

    arr = to_mono(array)
    if arr.size == 0:
        return 1.0

    frame = max(int(0.025 * sample_rate), 1)
    hop = max(int(0.010 * sample_rate), 1)
    if arr.size < frame:
        frame_rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
        threshold = max(1e-4, 0.10 * global_rms)
        return 1.0 if frame_rms < threshold else 0.0

    values = []
    for start in range(0, arr.size - frame + 1, hop):
        segment = arr[start : start + frame].astype(np.float64)
        values.append(float(np.sqrt(np.mean(segment * segment))))
    if not values:
        return 1.0
    threshold = max(1e-4, 0.10 * global_rms)
    return float(sum(v < threshold for v in values) / len(values))


def compute_row(example: dict[str, Any], split_name: str, args: argparse.Namespace) -> dict[str, str]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for audio quality auditing.") from exc

    audio = example.get("audio")
    example_id = str(example.get("ID") or example.get("id") or "")
    language = str(example.get("language") or example_id.split("_", 1)[0])
    transcript = normalize_text(transcript_from_example(example), "raw")

    flags: list[str] = []
    try:
        array, sample_rate = audio_array_from_example(audio)
        arr = to_mono(array)
        duration = float(arr.size / sample_rate) if sample_rate else 0.0
        if arr.size:
            peak = float(np.max(np.abs(arr)))
            rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
            clipping_ratio = float(np.mean(np.abs(arr) >= args.clipping_threshold))
            silence_ratio = frame_silence_ratio(arr, sample_rate, rms)
        else:
            peak = 0.0
            rms = 0.0
            clipping_ratio = 0.0
            silence_ratio = 1.0
            flags.append("empty_audio")
    except Exception as exc:
        duration = 0.0
        peak = 0.0
        rms = 0.0
        clipping_ratio = 0.0
        silence_ratio = 1.0
        flags.append(f"audio_error:{type(exc).__name__}")

    transcript_chars = len(transcript)
    transcript_words = len(transcript.split())
    chars_per_second = transcript_chars / duration if duration > 0 and transcript_chars else 0.0
    words_per_second = transcript_words / duration if duration > 0 and transcript_words else 0.0

    is_too_short = duration < args.min_duration
    is_too_long = duration > args.max_duration
    is_low_energy = rms < args.low_rms_threshold or peak < args.low_peak_threshold
    is_clipped = clipping_ratio > args.clipping_ratio_threshold
    has_text = transcript_chars > 0
    is_text_rate_outlier = bool(
        has_text
        and duration > 0
        and (
            chars_per_second < args.min_chars_per_second
            or chars_per_second > args.max_chars_per_second
            or words_per_second < args.min_words_per_second
            or words_per_second > args.max_words_per_second
        )
    )

    if is_too_short:
        flags.append("too_short")
    if is_too_long:
        flags.append("too_long")
    if is_low_energy:
        flags.append("low_energy")
    if is_clipped:
        flags.append("clipped")
    if is_text_rate_outlier:
        flags.append("text_rate_outlier")
    if not has_text and split_name != "test":
        flags.append("empty_transcript")

    return {
        "ID": example_id,
        "language": language,
        "split": split_name,
        "duration_seconds": safe_float(duration),
        "transcript_chars": str(transcript_chars),
        "transcript_words": str(transcript_words),
        "chars_per_second": safe_float(chars_per_second),
        "words_per_second": safe_float(words_per_second),
        "audio_rms": safe_float(rms, digits=8),
        "peak_amplitude": safe_float(peak, digits=8),
        "clipping_ratio": safe_float(clipping_ratio, digits=8),
        "approximate_silence_ratio": safe_float(silence_ratio),
        "is_too_short": bool_text(is_too_short),
        "is_too_long": bool_text(is_too_long),
        "is_low_energy": bool_text(is_low_energy),
        "is_clipped": bool_text(is_clipped),
        "is_text_rate_outlier": bool_text(is_text_rate_outlier),
        "quality_flags": ";".join(flags),
    }


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None, "mean": None}
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for audio quality auditing.") from exc

    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(mean(values)),
    }


def make_summary(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['split']}:{row['language']}"].append(row)

    summary: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "languages": args.languages,
        "thresholds": {
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "low_rms_threshold": args.low_rms_threshold,
            "low_peak_threshold": args.low_peak_threshold,
            "clipping_ratio_threshold": args.clipping_ratio_threshold,
            "min_chars_per_second": args.min_chars_per_second,
            "max_chars_per_second": args.max_chars_per_second,
            "min_words_per_second": args.min_words_per_second,
            "max_words_per_second": args.max_words_per_second,
        },
        "total_examples": len(rows),
        "by_split_language": {},
    }

    for key, group_rows in sorted(groups.items()):
        flag_counter: Counter[str] = Counter()
        for row in group_rows:
            for flag in row["quality_flags"].split(";"):
                if flag:
                    flag_counter[flag] += 1
        summary["by_split_language"][key] = {
            "num_examples": len(group_rows),
            "duration_seconds": quantiles([float(r["duration_seconds"]) for r in group_rows if r["duration_seconds"]]),
            "chars_per_second": quantiles([float(r["chars_per_second"]) for r in group_rows if r["chars_per_second"]]),
            "words_per_second": quantiles([float(r["words_per_second"]) for r in group_rows if r["words_per_second"]]),
            "audio_rms": quantiles([float(r["audio_rms"]) for r in group_rows if r["audio_rms"]]),
            "silence_ratio": quantiles([float(r["approximate_silence_ratio"]) for r in group_rows if r["approximate_silence_ratio"]]),
            "flag_counts": dict(sorted(flag_counter.items())),
        }
    return summary


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install with `uv sync`.") from exc

    dataset_path = args.dataset_dir / "hf_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared audio dataset not found: {dataset_path}. "
            "Run scripts/prepare_dataset.py without --metadata-only first."
        )

    dataset_dict = load_from_disk(dataset_path)
    split_names = list(dataset_dict.keys()) if args.split == "all" else [args.split]
    missing = [split for split in split_names if split not in dataset_dict]
    if missing:
        raise ValueError(f"Missing split(s) in {dataset_path}: {missing}. Available: {list(dataset_dict.keys())}")

    language_set = set(args.languages or TARGET_LANGUAGES)
    rows: list[dict[str, str]] = []
    for split_name in split_names:
        ds = dataset_dict[split_name]
        if language_set:
            ds = ds.filter(lambda row, wanted=language_set: row["language"] in wanted)
        print(f"Auditing {len(ds)} examples from split={split_name} languages={sorted(language_set)}")
        for idx, example in enumerate(ds, start=1):
            rows.append(compute_row(example, split_name, args))
            if args.progress_every and idx % args.progress_every == 0:
                print(f"  audited {idx}/{len(ds)} from {split_name}", flush=True)

    output = args.output or Path("outputs/quality") / f"audio_quality_{args.split}.csv"
    summary_output = args.summary_output or output.with_suffix(".summary.json")
    write_rows(output, rows)
    summary = make_summary(rows, args)
    json_dump(summary, summary_output)

    print(f"Wrote audio quality rows: {output}")
    print(f"Wrote audio quality summary: {summary_output}")
    print(f"Total examples: {summary['total_examples']}")
    for key, value in summary["by_split_language"].items():
        print(f"{key}: {value['num_examples']} examples, flags={value['flag_counts']}")


if __name__ == "__main__":
    main()
