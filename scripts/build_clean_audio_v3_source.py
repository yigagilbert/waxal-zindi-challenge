#!/usr/bin/env python3
"""Assemble the clean_audio_v3 SOURCE DatasetDict for clean_and_trim_audio_dataset.

Concatenates the in-domain WAXAL generalization-mix TRAIN with the standardized
external datasets (Afrivoice sna/lin, Luganda cv-yogera lug) into a single train
split, and keeps the WAXAL generalization-mix VALIDATION unchanged (in-domain only)
so validation WER/CER stays comparable across every experiment.

All inputs already share the 11-column schema (see standardize_afrivoice_audio.py /
standardize_luganda_filtered.py / prepare_generalization_mix.py), so the merge is a
column-align + concatenate. `source_dataset` already distinguishes in-domain ("waxal")
from external rows, which is the in-domain tag the cleaner's per-source report uses.

Output: data/processed_clean_audio_v3/hf_dataset (a DatasetDict), ready for:
  python scripts/clean_and_trim_audio_dataset.py \
    --dataset-dir data/processed_clean_audio_v3 \
    --preprocessing-version clean_audio_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.utils import ensure_dir, json_dump  # noqa: E402

COLUMNS = [
    "ID", "audio", "transcription", "language", "original_split", "duration",
    "source_dataset", "source_split", "source_language", "source_license", "source_risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waxal-dir", type=Path, default=Path("data/processed_generalization_mix"),
                        help="Dir holding hf_dataset (DatasetDict with train/validation) for the in-domain mix.")
    parser.add_argument("--external", type=Path, action="append", default=None,
                        help="Standardized external dataset dir (holds hf_dataset, a single Dataset). Repeat.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_clean_audio_v3"))
    parser.add_argument("--max-per-external", type=int, default=None, help="Cap rows per external source (balance).")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _load_any(path: Path):
    """Load a saved dataset dir whether it's the dir itself or holds hf_dataset/."""
    from datasets import load_from_disk

    for candidate in (path / "hf_dataset", path):
        if (candidate / "dataset_dict.json").exists() or (candidate / "dataset_info.json").exists():
            return load_from_disk(str(candidate))
    return load_from_disk(str(path))


# Defaults for a base WAXAL DatasetDict that predates the source_* schema
# (e.g. pure data/processed from prepare_dataset, before generalization_mix's align).
WAXAL_SOURCE_DEFAULTS = {
    "source_dataset": "waxal",
    "source_split": "",  # filled from original_split per row below
    "source_language": "",  # filled from language per row below
    "source_license": "CC-BY-4.0",
    "source_risk": "in_domain_waxal",
}


def _ensure_schema(ds, *, is_waxal_base: bool):
    """Add any missing schema columns. WAXAL base rows get in-domain source_* defaults."""
    cols = set(ds.column_names)
    n = len(ds)
    if "source_split" not in cols and "original_split" in cols:
        ds = ds.add_column("source_split", list(ds["original_split"]))
        cols.add("source_split")
    if "source_language" not in cols and "language" in cols:
        ds = ds.add_column("source_language", list(ds["language"]))
        cols.add("source_language")
    for col in COLUMNS:
        if col in cols:
            continue
        default = WAXAL_SOURCE_DEFAULTS.get(col, "" if col != "duration" else -1.0) if is_waxal_base else (
            "" if col != "duration" else -1.0)
        ds = ds.add_column(col, [default] * n)
        cols.add(col)
    return ds


def _align(ds, template_features):
    """Reorder/cast a single Dataset to the template train features so concatenate works."""
    missing = [c for c in COLUMNS if c not in ds.column_names]
    if missing:
        raise SystemExit(f"external dataset missing columns {missing}; got {ds.column_names}")
    extra = [c for c in ds.column_names if c not in COLUMNS]
    if extra:
        ds = ds.remove_columns(extra)
    ds = ds.select_columns(COLUMNS)
    if template_features is not None:
        ds = ds.cast(template_features)
    return ds


def main() -> None:
    args = parse_args()
    from datasets import DatasetDict, concatenate_datasets

    mix = _load_any(args.waxal_dir)
    if not hasattr(mix, "keys") or "train" not in mix:
        raise SystemExit(f"{args.waxal_dir} did not load a DatasetDict with a 'train' split")
    train = _ensure_schema(mix["train"], is_waxal_base=True).select_columns(COLUMNS)
    template = train.features
    validation = (
        _ensure_schema(mix["validation"], is_waxal_base=True).select_columns(COLUMNS)
        if "validation" in mix else None
    )
    print(f"in-domain WAXAL train: {len(train)} rows | validation: {len(validation) if validation is not None else 0}")

    parts = [train]
    contributions: dict[str, int] = {"waxal(in-domain train)": len(train)}
    for ext_path in args.external or []:
        ds = _load_any(ext_path)
        if hasattr(ds, "keys"):  # a DatasetDict — concatenate its splits
            ds = concatenate_datasets([ds[k] for k in ds.keys()])
        ds = _align(ds, template)
        if args.max_per_external and len(ds) > args.max_per_external:
            ds = ds.shuffle(seed=42).select(range(args.max_per_external))
        parts.append(ds)
        label = str(ext_path)
        contributions[label] = len(ds)
        print(f"external {label}: +{len(ds)} rows")

    merged_train = concatenate_datasets(parts)
    splits = {"train": merged_train}
    if validation is not None:
        splits["validation"] = validation
    out_dd = DatasetDict(splits)

    ensure_dir(args.output_dir)
    out = args.output_dir / "hf_dataset"
    out_dd.save_to_disk(out)

    summary = {
        "output": str(out),
        "train_rows": len(merged_train),
        "validation_rows": len(validation) if validation is not None else 0,
        "contributions": contributions,
        "train_by_language": dict(sorted(Counter(merged_train["language"]).items())),
        "train_by_source": dict(sorted(Counter(merged_train["source_dataset"]).items())),
    }
    json_dump(summary, args.report or (args.output_dir / "clean_audio_v3_merge_report.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved clean_audio_v3 source DatasetDict to {out} ({len(merged_train)} train rows)")


if __name__ == "__main__":
    main()
