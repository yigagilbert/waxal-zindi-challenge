#!/usr/bin/env python3
"""Standardize Afrivoice (Shona/Lingala) into the generalization-mix schema.

Afrivoice is a WebDataset: audio lives in <Lang>/audio_shards/audio_N.tar.xz (each
sample keyed like 'audio_XXXX'), and transcripts live in the JSON-Lines manifests
<Lang>/manifest_N.json ('transcription' + 'audio_filepath'). This joins them on the
audio key and emits rows with audio@16kHz + transcription + source tags, ready to
concatenate into the clean-audio superset and run through clean_and_trim_audio_dataset.

Decodes audio with soundfile + librosa (no torchcodec / no datasets Audio decoding),
so it is robust to the box's Python-env churn. Downloads the tar.xz shards, extracts
the wav members with tarfile, and joins them to the manifest transcripts on the key.

Run a capped smoke first (--max-per-language 50) to confirm the manifest<->wav join
before the full multi-GB download.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import ensure_dir, json_dump  # noqa: E402

AFRIVOICE_LANG = {"Shona": "sna", "Lingala": "lin"}

# Same column schema as prepare_generalization_mix so this concatenates cleanly.
COLUMNS = [
    "ID", "audio", "transcription", "language", "original_split", "duration",
    "source_dataset", "source_split", "source_language", "source_license", "source_risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="DigitalUmuganda/Afrivoice")
    parser.add_argument("--folders", nargs="*", default=["Shona", "Lingala"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/afrivoice_standardized"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--max-per-language", type=int, default=None, help="Cap per language (use a small value for a join smoke test).")
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load_manifest_map(repo: str, folder: str) -> tuple[dict[str, str], int]:
    """Return {audio-key -> transcription} from the JSON-Lines manifests (text only)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    manifests = [f for f in files if f.startswith(f"{folder}/") and f.endswith(".json")]
    mapping: dict[str, str] = {}
    for manifest in manifests:
        path = hf_hub_download(repo, manifest, repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                transcription = record.get("transcription")
                audio_filepath = record.get("audio_filepath") or ""
                if transcription and str(transcription).strip() and audio_filepath:
                    key = Path(str(audio_filepath)).stem  # e.g. 'audio_5WVMLWM20NSR'
                    mapping[key] = str(transcription)
    return mapping, len(manifests)


AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a")


def standardized_rows(repo: str, folder: str, language: str, mapping: dict[str, str], args: argparse.Namespace):
    """Yield standardized rows joining shard audio (tarfile+soundfile) with transcripts."""
    import io
    import tarfile

    import numpy as np
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    try:
        import librosa
    except ImportError:
        librosa = None

    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith(f"{folder}/audio_shards/") and f.endswith(".tar.xz"))
    matched = 0
    unmatched = 0
    for shard in shards:
        path = hf_hub_download(repo, shard, repo_type="dataset")
        with tarfile.open(path, "r:xz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(AUDIO_EXTS):
                    continue
                key = Path(member.name).stem  # 'audio_XXXX'
                transcription = mapping.get(key) or mapping.get(member.name)
                if not transcription:
                    unmatched += 1
                    if unmatched <= 5:
                        print(f"  [{folder}] unmatched audio member: {member.name!r}")
                    continue
                text = normalize_text(transcription, args.normalization)
                if not text:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                try:
                    array, sr = sf.read(io.BytesIO(handle.read()), dtype="float32", always_2d=False)
                except Exception as exc:
                    print(f"  [{folder}] decode failed for {member.name}: {type(exc).__name__}: {exc}")
                    continue
                if getattr(array, "ndim", 1) > 1:
                    array = array.mean(axis=1)  # to mono
                if sr != args.sample_rate:
                    if librosa is None:
                        print("  librosa required to resample; skipping"); continue
                    array = librosa.resample(np.asarray(array, dtype=np.float32), orig_sr=sr, target_sr=args.sample_rate)
                    sr = args.sample_rate
                array = np.asarray(array, dtype=np.float32)
                yield {
                    "ID": f"afrivoice_{language}_{key}",
                    "audio": {"array": array, "sampling_rate": sr},
                    "transcription": text,
                    "language": language,
                    "original_split": "external_train",
                    "duration": round(len(array) / sr, 3) if sr else -1.0,
                    "source_dataset": "DigitalUmuganda/Afrivoice",
                    "source_split": folder,
                    "source_language": language,
                    "source_license": "CC-BY-4.0",
                    "source_risk": "external_ccby_image_description_speech",
                }
                matched += 1
                if args.max_per_language and matched >= args.max_per_language:
                    break
        if args.max_per_language and matched >= args.max_per_language:
            break
    print(f"  [{folder}] matched {matched}, unmatched {unmatched} (of {len(mapping)} transcripts)")
    standardized_rows.last_stats = {"matched": matched, "unmatched": unmatched, "transcripts": len(mapping)}


def main() -> None:
    args = parse_args()
    from datasets import Audio, Dataset, concatenate_datasets

    ensure_dir(args.output_dir)
    parts = []
    stats = {}
    for folder in args.folders:
        language = AFRIVOICE_LANG.get(folder, folder.lower()[:3])
        print(f"== {folder} ({language}) ==")
        mapping, num_manifests = load_manifest_map(args.repo, folder)
        print(f"  {len(mapping)} transcripts from {num_manifests} manifests")
        if not mapping:
            print(f"  no transcripts for {folder}; skipping")
            continue
        part = Dataset.from_generator(
            standardized_rows,
            gen_kwargs={"repo": args.repo, "folder": folder, "language": language, "mapping": mapping, "args": args},
        )
        part = part.cast_column("audio", Audio(sampling_rate=args.sample_rate))
        parts.append(part)
        stats[folder] = getattr(standardized_rows, "last_stats", {})

    if not parts:
        raise SystemExit("No Afrivoice data standardized; check gating/access and folder names.")
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    out = args.output_dir / "hf_dataset"
    dataset.save_to_disk(out)
    summary = {
        "repo": args.repo,
        "folders": args.folders,
        "sample_rate": args.sample_rate,
        "rows": len(dataset),
        "by_folder": stats,
        "by_language": dict(sorted(Counter(dataset["language"]).items())),
        "output": str(out),
    }
    json_dump(summary, args.report or (args.output_dir / "afrivoice_standardize_report.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved standardized Afrivoice to {out}")


if __name__ == "__main__":
    main()
