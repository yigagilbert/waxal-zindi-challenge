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
    parser.add_argument("--max-shards-per-language", type=int, default=None, help="Cap shards per language (use a small value, e.g. 1, for a smoke test).")
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--no-delete-shards", dest="delete_shards", action="store_false", help="Keep downloaded tar.xz shards (default: delete after processing to save disk).")
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


def list_shards(repo: str, folder: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"{folder}/audio_shards/") and f.endswith(".tar.xz"))


def process_one_shard(repo: str, shard: str, folder: str, language: str, mapping: dict[str, str], args: argparse.Namespace):
    """Download + extract one shard, return standardized rows. None => download failed (resumable)."""
    import io
    import os
    import tarfile

    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    try:
        import librosa
    except ImportError:
        librosa = None

    try:
        path = hf_hub_download(repo, shard, repo_type="dataset")
    except Exception as exc:
        print(f"  DOWNLOAD FAILED {shard}: {type(exc).__name__}: {exc} -> skipping (re-run to resume)")
        return None, 0

    rows = []
    unmatched = 0
    try:
        with tarfile.open(path, "r:xz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(AUDIO_EXTS):
                    continue
                key = Path(member.name).stem
                transcription = mapping.get(key) or mapping.get(member.name)
                if not transcription:
                    unmatched += 1
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
                    print(f"  decode failed {member.name}: {type(exc).__name__}: {exc}")
                    continue
                if getattr(array, "ndim", 1) > 1:
                    array = array.mean(axis=1)
                if sr != args.sample_rate:
                    if librosa is None:
                        continue
                    array = librosa.resample(np.asarray(array, dtype=np.float32), orig_sr=sr, target_sr=args.sample_rate)
                    sr = args.sample_rate
                array = np.asarray(array, dtype=np.float32)
                rows.append({
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
                })
    except Exception as exc:
        print(f"  EXTRACT FAILED {shard}: {type(exc).__name__}: {exc} -> skipping (re-run to resume)")
        return None, unmatched

    if args.delete_shards:
        try:
            os.remove(os.path.realpath(path))  # free disk; already extracted what we need
        except OSError:
            pass
    return rows, unmatched


def main() -> None:
    args = parse_args()
    from datasets import Audio, Dataset, concatenate_datasets, load_from_disk

    ensure_dir(args.output_dir)
    shards_dir = ensure_dir(args.output_dir / "shards")
    stats: dict = {}
    for folder in args.folders:
        language = AFRIVOICE_LANG.get(folder, folder.lower()[:3])
        print(f"== {folder} ({language}) ==")
        mapping, num_manifests = load_manifest_map(args.repo, folder)
        print(f"  {len(mapping)} transcripts from {num_manifests} manifests")
        if not mapping:
            print(f"  no transcripts for {folder}; skipping")
            continue
        shards = list_shards(args.repo, folder)
        if args.max_shards_per_language:
            shards = shards[: args.max_shards_per_language]
        matched = failed = unmatched_total = 0
        for i, shard in enumerate(shards, 1):
            shard_out = shards_dir / f"{folder}__{Path(shard).name}".replace(".tar.xz", "")
            if shard_out.exists():
                print(f"  [{i}/{len(shards)}] skip (already done): {shard}")
                continue
            print(f"  [{i}/{len(shards)}] {shard}")
            rows, unmatched = process_one_shard(args.repo, shard, folder, language, mapping, args)
            unmatched_total += unmatched
            if rows is None:
                failed += 1
                continue  # transient failure; re-run to resume this shard
            if rows:
                part = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=args.sample_rate))
                part.save_to_disk(shard_out)
                matched += len(rows)
        stats[folder] = {"matched_this_run": matched, "shards_failed": failed, "unmatched_this_run": unmatched_total,
                         "transcripts": len(mapping), "total_shards": len(shards)}
        print(f"  [{folder}] this run: matched {matched}, failed_shards {failed} (re-run to resume any failures)")

    # Combine all saved per-shard datasets (from this and any previous runs)
    shard_dirs = [d for d in sorted(shards_dir.iterdir()) if (d / "dataset_info.json").exists()]
    if not shard_dirs:
        raise SystemExit("No shards saved yet; check access/gating, then re-run.")
    parts = [load_from_disk(str(d)) for d in shard_dirs]
    dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    out = args.output_dir / "hf_dataset"
    dataset.save_to_disk(out)
    summary = {
        "repo": args.repo, "folders": args.folders, "sample_rate": args.sample_rate,
        "rows": len(dataset), "shard_datasets_combined": len(shard_dirs), "by_folder": stats,
        "by_language": dict(sorted(Counter(dataset["language"]).items())), "output": str(out),
    }
    json_dump(summary, args.report or (args.output_dir / "afrivoice_standardize_report.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved standardized Afrivoice to {out} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
