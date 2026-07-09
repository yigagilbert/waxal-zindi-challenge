#!/usr/bin/env python3
"""Push the cleaned-audio DatasetDict plus reports/docs to a PRIVATE HF repo.

Refuses to run with --public unless --i-know-this-is-public is also passed.
Audio bytes are embedded by push_to_hub, so future runs can `load_dataset`
the repo directly with no re-cleaning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/final_combined_clean_audio_dataset"))
    parser.add_argument("--repo-id", required=True, help="e.g. <HF_USERNAME>/waxal-combined-clean-audio-asr-private")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--i-know-this-is-public", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true", help="Only upload docs/reports (e.g. card update).")
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        metavar="LOCAL[:REPO_PATH]",
        help="Additional file to upload. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.public and not args.i_know_this_is_public:
        raise SystemExit("Refusing --public without --i-know-this-is-public. Competition data stays private by default.")
    private = not args.public

    from datasets import load_from_disk
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"Repo ready: {args.repo_id} (private={private})")

    if not args.skip_dataset:
        dataset_dict = load_from_disk(args.dataset / "hf_dataset")
        for split, ds in dataset_dict.items():
            print(f"  split {split}: {len(ds)} rows")
        dataset_dict.push_to_hub(args.repo_id, private=private)
        print("DatasetDict pushed (audio embedded).")

    default_uploads = [
        (Path("docs/CLEAN_DATASET_CARD.md"), "README.md"),
        (args.dataset / "train_metadata.csv", "metadata/train_metadata.csv"),
        (args.dataset / "medium_train_metadata.csv", "metadata/medium_train_metadata.csv"),
        (args.dataset / "validation_metadata.csv", "metadata/validation_metadata.csv"),
        (args.dataset / "excluded_metadata.csv", "metadata/excluded_metadata.csv"),
        (Path("outputs/data_quality/cleaning_impact_report.json"), "reports/cleaning_impact_report.json"),
        (Path("outputs/data_quality/full_audio_text_audit_summary.json"), "reports/full_audio_text_audit_summary.json"),
        (Path("scripts/clean_and_trim_audio_dataset.py"), "scripts/clean_and_trim_audio_dataset.py"),
        (Path("scripts/audit_audio_text_consistency.py"), "scripts/audit_audio_text_consistency.py"),
        (Path("docs/CLEAN_AUDIO_DATASET_REPRODUCIBILITY.md"), "docs/CLEAN_AUDIO_DATASET_REPRODUCIBILITY.md"),
        (Path("docs/DATA_QUALITY_AUDIT.md"), "docs/DATA_QUALITY_AUDIT.md"),
    ]
    for value in args.extra_file:
        local, _, repo_path = value.partition(":")
        default_uploads.append((Path(local), repo_path or Path(local).name))

    for local, repo_path in default_uploads:
        if not local.exists():
            print(f"  skipping missing {local}")
            continue
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=str(repo_path),
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"  uploaded {local} -> {repo_path}")

    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
