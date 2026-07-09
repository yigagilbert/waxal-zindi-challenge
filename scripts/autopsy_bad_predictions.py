#!/usr/bin/env python3
"""Audio autopsy for dot-only / very-short predictions.

For every flagged prediction, measures duration, RMS energy, silence ratio,
and clipping from the source audio, attaches fallback-model predictions, and
classifies each clip as likely-silent vs likely-real-speech so we know whether
further modeling effort can help.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from waxal.data import id_language, read_prediction_csv, write_csv_rows  # noqa: E402
from waxal.utils import json_dump  # noqa: E402

from run_xlsr_inference import load_split  # noqa: E402

PUNCT_ONLY_RE = re.compile(r"^[\s\.,!?;:'\"()\[\]{}\-_/\\|]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--fallback",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Fallback prediction CSV to attach. May be repeated.",
    )
    parser.add_argument("--short-chars", type=int, default=12)
    parser.add_argument("--short-words", type=int, default=2)
    parser.add_argument("--silence-rms-threshold", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/bad_prediction_autopsy.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/analysis/bad_prediction_autopsy.summary.json"))
    return parser.parse_args()


def is_flagged(text: str, *, short_chars: int, short_words: int) -> list[str]:
    stripped = (text or "").strip()
    reasons = []
    if not stripped:
        reasons.append("empty")
    elif set(stripped) <= {"."}:
        reasons.append("dot_only")
    elif PUNCT_ONLY_RE.fullmatch(stripped):
        reasons.append("punctuation_only")
    if 0 < len(stripped) < short_chars or 0 < len(stripped.split()) <= short_words:
        reasons.append("very_short")
    return reasons


def audio_stats(array, sampling_rate: int, *, silence_rms_threshold: float) -> dict:
    import numpy as np

    samples = np.asarray(array, dtype=np.float32).flatten()
    if samples.size == 0:
        return {"duration": 0.0, "rms": 0.0, "silence_ratio": 1.0, "clipping_ratio": 0.0, "peak": 0.0}
    frame = max(1, int(sampling_rate * 0.03))
    usable = samples[: (samples.size // frame) * frame]
    frames = usable.reshape(-1, frame) if usable.size else samples.reshape(1, -1)
    frame_rms = np.sqrt((frames**2).mean(axis=1))
    return {
        "duration": float(samples.size / sampling_rate),
        "rms": float(np.sqrt((samples**2).mean())),
        "silence_ratio": float((frame_rms < silence_rms_threshold).mean()),
        "clipping_ratio": float((np.abs(samples) > 0.99).mean()),
        "peak": float(np.abs(samples).max()),
    }


def main() -> None:
    args = parse_args()
    rows = read_prediction_csv(args.predictions)
    fallbacks = {}
    for value in args.fallback:
        name, _, path = value.partition("=")
        if not path:
            raise ValueError(f"--fallback must be NAME=PATH, got {value!r}")
        fallbacks[name] = {r["ID"]: r.get("Target", "") for r in read_prediction_csv(Path(path))}

    flagged = {}
    for row in rows:
        reasons = is_flagged(row.get("Target", ""), short_chars=args.short_chars, short_words=args.short_words)
        if reasons:
            flagged[row["ID"]] = {"prediction": row.get("Target", ""), "reasons": reasons}
    print(f"Flagged {len(flagged)}/{len(rows)} predictions for autopsy.")
    if not flagged:
        json_dump({"flagged": 0}, args.summary_output)
        return

    ds = load_split(args.dataset_dir, args.split)
    wanted = set(flagged)
    ds = ds.filter(lambda row_id, ids=wanted: row_id in ids, input_columns=["ID"])

    out_rows = []
    for example in ds:
        example_id = example["ID"]
        info = flagged[example_id]
        audio = example["audio"]
        stats = audio_stats(audio["array"], int(audio.get("sampling_rate", 16_000)), silence_rms_threshold=args.silence_rms_threshold)
        likely_silent = stats["rms"] < 0.005 or stats["silence_ratio"] > 0.9 or stats["duration"] < 0.4

        fallback_cells = {}
        fallback_sane = False
        for name, pred_map in fallbacks.items():
            candidate = pred_map.get(example_id, "")
            fallback_cells[f"fallback_{name}"] = candidate
            if candidate and not is_flagged(candidate, short_chars=args.short_chars, short_words=args.short_words):
                fallback_sane = True

        out_rows.append(
            {
                "ID": example_id,
                "language": example.get("language") or id_language(example_id),
                "reasons": "|".join(info["reasons"]),
                "prediction": info["prediction"][:200],
                "duration": round(stats["duration"], 2),
                "rms": round(stats["rms"], 5),
                "silence_ratio": round(stats["silence_ratio"], 3),
                "clipping_ratio": round(stats["clipping_ratio"], 4),
                "peak": round(stats["peak"], 3),
                "likely_silent": str(likely_silent),
                "fallback_has_sane_transcript": str(fallback_sane),
                **{k: v[:200] for k, v in fallback_cells.items()},
            }
        )
        print(f"  {example_id}: dur={stats['duration']:.1f}s rms={stats['rms']:.4f} silent={likely_silent} fallback_sane={fallback_sane}", flush=True)

    fieldnames = list(out_rows[0].keys()) if out_rows else ["ID"]
    write_csv_rows(args.output, out_rows, fieldnames)

    by_language = Counter(r["language"] for r in out_rows)
    silent = [r for r in out_rows if r["likely_silent"] == "True"]
    rescuable = [r for r in out_rows if r["likely_silent"] == "False" and r["fallback_has_sane_transcript"] == "True"]
    speech_no_fallback = [r for r in out_rows if r["likely_silent"] == "False" and r["fallback_has_sane_transcript"] == "False"]
    summary = {
        "flagged": len(flagged),
        "found_in_split": len(out_rows),
        "by_language": dict(sorted(by_language.items())),
        "likely_silent": len(silent),
        "real_speech_with_sane_fallback": len(rescuable),
        "real_speech_no_sane_fallback": len(speech_no_fallback),
        "rescuable_ids": [r["ID"] for r in rescuable],
        "speech_no_fallback_ids": [r["ID"] for r in speech_no_fallback][:50],
        "recommendation": (
            "mostly silent/degenerate audio; stop spending effort"
            if out_rows and len(silent) / len(out_rows) > 0.7
            else "meaningful fraction is real speech; add a stronger fallback tier (e.g. noirlab Whisper Lingala)"
        ),
    }
    json_dump(summary, args.summary_output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
