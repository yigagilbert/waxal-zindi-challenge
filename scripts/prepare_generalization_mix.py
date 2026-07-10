#!/usr/bin/env python3
"""Prepare a mixed ASR dataset for robust single-stage WAXAL training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.audio import duration_seconds  # noqa: E402
from waxal.data import read_csv_dicts, write_csv_rows  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import clean_name, json_dump  # noqa: E402


LANG_TO_FLEURS = {
    "lug": "lg_ug",
    "lin": "ln_cd",
    "sna": "sn_zw",
}

SOURCE_FIELDS = [
    "ID",
    "Target",
    "language",
    "original_split",
    "source_dataset",
    "source_split",
    "source_language",
    "source_license",
    "source_risk",
    "duration",
]

TEXT_COLUMNS = (
    "Target",
    "transcription",
    "text",
    "sentence",
    "raw_transcription",
    "normalized_text",
    "transcript",
)
ID_COLUMNS = ("ID", "id", "path", "audio_id", "file")
DURATION_COLUMNS = ("duration", "duration_seconds", "audio_duration", "num_samples")

DEFAULT_COLUMN_VALUES = {
    "ID": "",
    "audio": None,
    "transcription": "",
    "language": "",
    "original_split": "",
    "duration": -1.0,
    "source_dataset": "",
    "source_split": "",
    "source_language": "",
    "source_license": "",
    "source_risk": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waxal-dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--waxal-train-manifest", type=Path, default=Path("data/quality/clean_train_alvin_lingala_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_generalization_mix"))
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--languages", nargs="*", default=["lin", "lug", "sna"], choices=["lin", "lug", "sna"])
    parser.add_argument(
        "--fleurs-languages",
        nargs="*",
        default=None,
        choices=["lin", "lug", "sna"],
        help="Languages to augment with FLEURS. Defaults to --languages.",
    )
    parser.add_argument("--include-fleurs", action="store_true")
    parser.add_argument("--include-salt", action="store_true")
    parser.add_argument("--include-lingala-100hrs", action="store_true")
    parser.add_argument(
        "--allow-unverified-lingala-100hrs",
        action="store_true",
        help="Required together with --include-lingala-100hrs because dataset/source licenses need manual sign-off.",
    )
    parser.add_argument("--external-splits", nargs="*", default=["train", "validation", "test"])
    parser.add_argument("--fleurs-max-per-language", type=int, default=None)
    parser.add_argument("--salt-max", type=int, default=None)
    parser.add_argument("--salt-configs", nargs="*", default=["multispeaker-lug", "studio-lug"])
    parser.add_argument("--lingala-100hrs-max", type=int, default=None)
    parser.add_argument("--skip-duration", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    columns, rows, bad = read_csv_dicts(path)
    if bad:
        raise ValueError(f"{path} has malformed rows with extra fields: {bad[:3]}")
    if not columns:
        raise ValueError(f"{path} has no header")
    return rows


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}; got {sorted(columns)}")


def choose_text(row: dict[str, Any]) -> str:
    for key in ("Target", "transcription", "text", "sentence", "raw_transcription", "normalized_text", "transcript"):
        value = row.get(key)
        if value not in {None, ""}:
            return normalize_text(str(value), "language_safe")
    return ""


def choose_row_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("ID", "id", "path", "audio_id", "file"):
        value = row.get(key)
        if value not in {None, ""}:
            return clean_name(str(value))
    audio = row.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return clean_name(str(audio["path"]))
    return fallback


def row_duration(row: dict[str, Any], *, skip_duration: bool) -> float | None:
    if skip_duration:
        return None
    audio = row.get("audio")
    if isinstance(audio, dict) and audio.get("array") is not None and audio.get("sampling_rate"):
        try:
            return duration_seconds(audio["array"], int(audio["sampling_rate"]))
        except Exception:
            return None
    for key in ("duration", "duration_seconds", "audio_duration"):
        value = row.get(key)
        if value not in {None, ""}:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            return out if math.isfinite(out) else None
    return None


def safe_select(ds, max_examples: int | None, seed: int):
    if max_examples is None or len(ds) <= max_examples:
        return ds
    return ds.shuffle(seed=seed).select(range(max_examples))


def _batch_value(columns: dict[str, Any], key: str, index: int) -> Any:
    values = columns.get(key)
    if values is None:
        return None
    return values[index]


def _choose_text_from_batch(columns: dict[str, Any], index: int) -> str:
    for key in TEXT_COLUMNS:
        value = _batch_value(columns, key, index)
        if value is not None and str(value).strip():
            return normalize_text(str(value), "language_safe")
    return ""


def _choose_id_from_batch(columns: dict[str, Any], index: int, fallback: str) -> str:
    for key in ID_COLUMNS:
        value = _batch_value(columns, key, index)
        if value is not None and str(value).strip():
            return clean_name(str(value))
    return fallback


def _duration_from_batch(
    columns: dict[str, Any],
    index: int,
    *,
    sample_rate: int,
    skip_duration: bool,
) -> float:
    if skip_duration:
        return -1.0
    for key in ("duration", "duration_seconds", "audio_duration"):
        value = _batch_value(columns, key, index)
        if value is None or value == "":
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration):
            return duration
    num_samples = _batch_value(columns, "num_samples", index)
    if num_samples not in {None, ""}:
        try:
            duration = float(num_samples) / float(sample_rate)
        except (TypeError, ValueError, ZeroDivisionError):
            return -1.0
        if math.isfinite(duration):
            return duration
    return -1.0


def standardize_dataset(
    ds,
    *,
    language: str,
    source_dataset: str,
    source_split: str,
    source_language: str,
    source_license: str,
    source_risk: str,
    id_prefix: str,
    sample_rate: int,
    skip_duration: bool,
):
    """Attach WAXAL-compatible metadata without decoding or materializing audio."""
    if "audio" not in ds.column_names:
        raise ValueError(f"{source_dataset} split={source_split} has no audio column: {ds.column_names}")

    input_columns = [
        column
        for column in (*TEXT_COLUMNS, *ID_COLUMNS, *DURATION_COLUMNS)
        if column in ds.column_names and column != "audio"
    ]
    if not any(column in ds.column_names for column in TEXT_COLUMNS):
        raise ValueError(f"{source_dataset} split={source_split} has no recognized transcript column: {ds.column_names}")

    def attach_metadata(*values):
        *column_values, indices = values
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        columns = dict(zip(input_columns, column_values, strict=True))
        ids = []
        transcriptions = []
        durations = []
        for item_index, dataset_index in enumerate(indices):
            source_id = _choose_id_from_batch(columns, item_index, str(dataset_index))
            ids.append(f"{id_prefix}_{source_split}_{source_id}")
            transcriptions.append(_choose_text_from_batch(columns, item_index))
            durations.append(
                _duration_from_batch(
                    columns,
                    item_index,
                    sample_rate=sample_rate,
                    skip_duration=skip_duration,
                )
            )
        size = len(indices)
        return {
            "ID": ids,
            "transcription": transcriptions,
            "language": [language] * size,
            "original_split": ["external_train"] * size,
            "duration": durations,
            "source_dataset": [source_dataset] * size,
            "source_split": [source_split] * size,
            "source_language": [source_language] * size,
            "source_license": [source_license] * size,
            "source_risk": [source_risk] * size,
        }

    standardized = ds.map(
        attach_metadata,
        batched=True,
        with_indices=True,
        input_columns=input_columns,
        desc=f"Standardizing {source_dataset} {source_language}/{source_split}",
    )
    return standardized.filter(
        lambda transcription: bool(str(transcription).strip()),
        input_columns=["transcription"],
        desc=f"Filtering empty transcripts from {source_dataset} {source_language}/{source_split}",
    )


def prepare_waxal_train(args: argparse.Namespace):
    from datasets import load_from_disk

    dataset_dict = load_from_disk(args.waxal_dataset_dir / "hf_dataset")
    train_ds = dataset_dict["train"]
    manifest_rows = load_rows(args.waxal_train_manifest)
    require_columns(args.waxal_train_manifest, manifest_rows, {"ID", "Target", "language"})
    manifest_by_id = {row["ID"]: row for row in manifest_rows if row["language"] in set(args.languages)}
    wanted = set(manifest_by_id)
    train_ds = train_ds.filter(lambda row_id: row_id in wanted, input_columns=["ID"])

    def attach(ids, languages, original_splits):
        return {
            "transcription": [normalize_text(manifest_by_id[row_id]["Target"], "language_safe") for row_id in ids],
            "source_dataset": ["waxal_official_clean"] * len(ids),
            "source_split": [split or "train" for split in original_splits],
            "source_language": list(languages),
            "source_license": ["competition_provided"] * len(ids),
            "source_risk": ["official_challenge_train"] * len(ids),
        }

    return train_ds.map(attach, batched=True, input_columns=["ID", "language", "original_split"])


def prepare_waxal_validation(args: argparse.Namespace):
    from datasets import load_from_disk

    dataset_dict = load_from_disk(args.waxal_dataset_dir / "hf_dataset")
    validation = dataset_dict["validation"]
    wanted = set(args.languages)
    validation = validation.filter(lambda language: language in wanted, input_columns=["language"])

    def attach(languages):
        return {
            "source_dataset": ["waxal_official_validation"] * len(languages),
            "source_split": ["validation"] * len(languages),
            "source_language": list(languages),
            "source_license": ["competition_provided"] * len(languages),
            "source_risk": ["official_challenge_validation_holdout"] * len(languages),
        }

    return validation.map(attach, batched=True, input_columns=["language"])


def prepare_waxal_test(args: argparse.Namespace):
    from datasets import load_from_disk

    dataset_dict = load_from_disk(args.waxal_dataset_dir / "hf_dataset")
    if "test" not in dataset_dict:
        return None
    test = dataset_dict["test"]
    wanted = set(args.languages)
    test = test.filter(lambda language: language in wanted, input_columns=["language"])

    def attach(languages):
        return {
            "source_dataset": ["waxal_official_test_audio"] * len(languages),
            "source_split": ["test"] * len(languages),
            "source_language": list(languages),
            "source_license": ["competition_provided"] * len(languages),
            "source_risk": ["unlabeled_challenge_test_audio"] * len(languages),
        }

    return test.map(attach, batched=True, input_columns=["language"])


def load_external_dataset(name: str, config: str | None, split: str, sample_rate: int):
    from datasets import Audio, load_dataset

    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    return ds.cast_column("audio", Audio(sampling_rate=sample_rate))


def prepare_fleurs(args: argparse.Namespace) -> list[Any]:
    from datasets import concatenate_datasets

    datasets = []
    fleurs_languages = args.fleurs_languages or args.languages
    for language in fleurs_languages:
        config = LANG_TO_FLEURS.get(language)
        if not config:
            continue
        per_language_parts = []
        for split in args.external_splits:
            try:
                ds = load_external_dataset("google/fleurs", config, split, args.sample_rate)
            except Exception as exc:
                print(f"WARNING: failed to load FLEURS {config} split={split}: {type(exc).__name__}: {exc}")
                continue
            per_language_parts.append(
                standardize_dataset(
                    ds,
                    language=language,
                    source_dataset="google/fleurs",
                    source_split=split,
                    source_language=config,
                    source_license="CC-BY-4.0",
                    source_risk="public_external_read_speech",
                    id_prefix=f"fleurs_{language}",
                    sample_rate=args.sample_rate,
                    skip_duration=args.skip_duration,
                )
            )
        if not per_language_parts:
            continue
        per_language_ds = concatenate_datasets(per_language_parts)
        per_language_ds = safe_select(per_language_ds, args.fleurs_max_per_language, args.seed)
        datasets.append(per_language_ds)
    return datasets


def prepare_salt(args: argparse.Namespace) -> list[Any]:
    from datasets import concatenate_datasets

    if "lug" not in args.languages:
        return []
    datasets = []
    source_language_values = {"lug", "lg", "luganda", "Luganda", "ganda", "Ganda"}
    for config in args.salt_configs:
        for split in args.external_splits:
            try:
                ds = load_external_dataset("Sunbird/salt", config, split, args.sample_rate)
            except Exception as exc:
                if split != "train":
                    continue
                print(
                    f"WARNING: failed to load Sunbird/salt config={config} split={split}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if "language" in ds.column_names:
                ds = ds.filter(lambda language: language in source_language_values, input_columns=["language"])
            elif "lang" in ds.column_names:
                ds = ds.filter(lambda language: language in source_language_values, input_columns=["lang"])
            datasets.append(
                standardize_dataset(
                    ds,
                    language="lug",
                    source_dataset="Sunbird/salt",
                    source_split=split,
                    source_language=config,
                    source_license="CC-BY-SA-4.0",
                    source_risk="public_external_luganda_speech",
                    id_prefix=f"salt_{clean_name(config)}",
                    sample_rate=args.sample_rate,
                    skip_duration=args.skip_duration,
                )
            )
    if not datasets:
        return []
    salt = concatenate_datasets(datasets)
    return [safe_select(salt, args.salt_max, args.seed)]


# Verified per-source on 2026-07-09 (see docs/LINGALA_100HRS_PROPOSED_README.md):
# Afrivoice CC-BY-4.0, LRSC CC BY 4.0 (DOI 10.17632/28x8tc9n9k.1), FLEURS CC-BY-4.0.
# 'lingala_tts' rows have unknown provenance/license and are always excluded.
ALLOWED_LINGALA_100HRS_SOURCES = {"Afrivoice", "LRSC", "fleurs"}


def waxal_lingala_texts(waxal_dataset_dir: Path) -> set[str]:
    """Normalized Lingala transcripts from WAXAL train AND validation metadata."""
    texts: set[str] = set()
    for name in ("train.csv", "validation.csv"):
        path = waxal_dataset_dir / name
        if not path.exists():
            print(f"WARNING: {path} missing; text-overlap dedup will be incomplete.")
            continue
        _, rows, _ = read_csv_dicts(path)
        for row in rows:
            if (row.get("language") or "") != "lin":
                continue
            normalized = normalize_text(row.get("Target") or row.get("transcription") or "", "language_safe")
            if normalized:
                texts.add(normalized)
    return texts


def prepare_lingala_100hrs(args: argparse.Namespace) -> list[Any]:
    from datasets import concatenate_datasets

    if "lin" not in args.languages:
        return []
    if not args.allow_unverified_lingala_100hrs:
        raise ValueError(
            "--include-lingala-100hrs requires --allow-unverified-lingala-100hrs as an explicit opt-in. "
            "Per-source licenses were verified CC-BY-4.0 on 2026-07-09; unknown 'lingala_tts' rows and "
            "WAXAL train/validation text overlaps are excluded automatically below."
        )
    overlap_texts = waxal_lingala_texts(args.waxal_dataset_dir)
    print(f"Lingala_100hrs dedup set: {len(overlap_texts)} WAXAL lin train+validation transcripts")
    datasets = []
    for split in args.external_splits:
        try:
            ds = load_external_dataset("KasuleTrevor/Lingala_100hrs", None, split, args.sample_rate)
        except Exception as exc:
            print(f"WARNING: failed to load Lingala_100hrs split={split}: {type(exc).__name__}: {exc}")
            continue
        rows_loaded = len(ds)
        if "source" in ds.column_names:
            ds = ds.filter(
                lambda source: source in ALLOWED_LINGALA_100HRS_SOURCES,
                input_columns=["source"],
                desc=f"Lingala_100hrs {split}: dropping non-verified sources",
            )
        rows_after_source = len(ds)
        ds = ds.filter(
            lambda text, overlap=overlap_texts: normalize_text(str(text or ""), "language_safe") not in overlap,
            input_columns=["text"],
            desc=f"Lingala_100hrs {split}: dropping WAXAL text overlaps",
        )
        print(
            f"Lingala_100hrs {split}: {rows_loaded} loaded -> {rows_after_source} after source filter "
            f"-> {len(ds)} after WAXAL-overlap dedup"
        )
        datasets.append(
            standardize_dataset(
                ds,
                language="lin",
                source_dataset="KasuleTrevor/Lingala_100hrs",
                source_split=split,
                source_language="lin",
                source_license="CC-BY-4.0 (per-source Afrivoice/LRSC/FLEURS, verified 2026-07-09)",
                source_risk="external_ccby_deduped_vs_waxal_lin_train_and_validation",
                id_prefix="lingala100",
                sample_rate=args.sample_rate,
                skip_duration=args.skip_duration,
            )
        )
    if not datasets:
        return []
    lingala = concatenate_datasets(datasets)
    return [safe_select(lingala, args.lingala_100hrs_max, args.seed)]


def align_columns(ds, columns: list[str]):
    remove = [column for column in ds.column_names if column not in columns]
    if remove:
        ds = ds.remove_columns(remove)
    for column in columns:
        if column not in ds.column_names:
            ds = ds.add_column(column, [DEFAULT_COLUMN_VALUES[column]] * len(ds))
    return ds.select_columns(columns)


def write_metadata_csv(path: Path, ds) -> None:
    rows = []
    metadata_ds = ds.remove_columns(["audio"]) if "audio" in ds.column_names else ds
    for row in metadata_ds:
        rows.append(
            {
                "ID": row["ID"],
                "Target": row.get("transcription", ""),
                "language": row.get("language", ""),
                "original_split": row.get("original_split", ""),
                "source_dataset": row.get("source_dataset", ""),
                "source_split": row.get("source_split", ""),
                "source_language": row.get("source_language", ""),
                "source_license": row.get("source_license", ""),
                "source_risk": row.get("source_risk", ""),
                "duration": "" if row.get("duration") is None else row.get("duration"),
            }
        )
    write_csv_rows(path, rows, SOURCE_FIELDS)


def main() -> None:
    args = parse_args()
    if args.include_lingala_100hrs and not args.allow_unverified_lingala_100hrs:
        raise ValueError(
            "Refusing to include KasuleTrevor/Lingala_100hrs without "
            "--allow-unverified-lingala-100hrs. Audit/license sign-off is required."
        )

    try:
        from datasets import DatasetDict, concatenate_datasets
    except ImportError as exc:
        raise RuntimeError("Install datasets[audio] before preparing the generalization mix.") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    waxal_train = prepare_waxal_train(args)
    validation = prepare_waxal_validation(args)
    test = prepare_waxal_test(args)

    external_parts = []
    source_reports = []
    if args.include_fleurs:
        fleurs_parts = prepare_fleurs(args)
        source_reports.append({"source": "google/fleurs", "rows": sum(len(part) for part in fleurs_parts), "license": "CC-BY-4.0"})
        external_parts.extend(fleurs_parts)
    if args.include_salt:
        salt_parts = prepare_salt(args)
        source_reports.append({"source": "Sunbird/salt", "rows": sum(len(part) for part in salt_parts), "license": "CC-BY-SA-4.0"})
        external_parts.extend(salt_parts)
    if args.include_lingala_100hrs:
        lingala_parts = prepare_lingala_100hrs(args)
        source_reports.append(
            {
                "source": "KasuleTrevor/Lingala_100hrs",
                "rows": sum(len(part) for part in lingala_parts),
                "license": "CC-BY-4.0 per source (Afrivoice/LRSC/FLEURS, verified 2026-07-09)",
                "risk": "lingala_tts rows and WAXAL lin train/validation text overlaps excluded at ingestion",
            }
        )
        external_parts.extend(lingala_parts)

    columns = [
        "ID",
        "audio",
        "transcription",
        "language",
        "original_split",
        "duration",
        "source_dataset",
        "source_split",
        "source_language",
        "source_license",
        "source_risk",
    ]
    train_parts = [align_columns(waxal_train, columns)]
    train_parts.extend(align_columns(part, columns) for part in external_parts)
    train = concatenate_datasets(train_parts).shuffle(seed=args.seed)
    validation = align_columns(validation, columns)

    dataset_dict = DatasetDict({"train": train, "validation": validation})
    if test is not None:
        dataset_dict["test"] = align_columns(test, columns)

    out_path = args.output_dir / "hf_dataset"
    tmp_path = args.output_dir / "hf_dataset.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    dataset_dict.save_to_disk(tmp_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.replace(out_path)

    write_metadata_csv(args.output_dir / "train.csv", train)
    write_metadata_csv(args.output_dir / "validation.csv", validation)
    if "test" in dataset_dict:
        write_metadata_csv(args.output_dir / "test.csv", dataset_dict["test"])

    by_source_language: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for source, language in zip(train["source_dataset"], train["language"], strict=True):
        by_source_language[source][language] += 1
    by_language = Counter(train["language"])
    by_source = Counter(train["source_dataset"])
    summary = {
        "output_dir": str(args.output_dir),
        "dataset_path": str(out_path),
        "waxal_dataset_dir": str(args.waxal_dataset_dir),
        "waxal_train_manifest": str(args.waxal_train_manifest),
        "languages": args.languages,
        "fleurs_languages": args.fleurs_languages or args.languages,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "external_splits": args.external_splits,
        "include_fleurs": args.include_fleurs,
        "include_salt": args.include_salt,
        "include_lingala_100hrs": args.include_lingala_100hrs,
        "allow_unverified_lingala_100hrs": args.allow_unverified_lingala_100hrs,
        "salt_configs": args.salt_configs,
        "source_reports": source_reports,
        "split_rows": {split: len(dataset_dict[split]) for split in dataset_dict},
        "train_by_language": dict(sorted(by_language.items())),
        "train_by_source": dict(sorted(by_source.items())),
        "train_by_source_language": {
            source: dict(sorted(values.items()))
            for source, values in sorted(by_source_language.items())
        },
        "rule_notes": {
            "waxal": "official challenge train/validation/test audio only; no public test labels used",
            "fleurs": "public CC-BY-4.0 external data",
            "salt": "public CC-BY-SA-4.0 external Luganda data",
            "lingala_100hrs": "explicit opt-in; per-source CC-BY-4.0 verified 2026-07-09; lingala_tts rows and WAXAL lin train/validation text overlaps excluded at ingestion",
        },
    }
    json_dump(summary, args.output_dir / "generalization_mix_report.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
