#!/usr/bin/env python3
"""Score an ID/Target prediction CSV against a prepared Hugging Face split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_prediction_csv  # noqa: E402
from waxal.scoring import score_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from datasets import load_from_disk

    dataset = load_from_disk(args.dataset_dir / "hf_dataset")[args.split]
    required = {"ID", "transcription"}
    missing_columns = sorted(required - set(dataset.column_names))
    if missing_columns:
        raise ValueError(f"dataset split is missing columns: {missing_columns}")

    references = [
        {
            "ID": example_id,
            "Target": transcription,
            "language": language if "language" in dataset.column_names else "",
        }
        for example_id, transcription, language in zip(
            dataset["ID"],
            dataset["transcription"],
            dataset["language"] if "language" in dataset.column_names else [""] * len(dataset),
            strict=True,
        )
    ]
    predictions = read_prediction_csv(args.predictions)
    report = score_records(references, predictions, normalization=args.normalization)
    if report["missing_predictions"] or report["extra_predictions"]:
        raise ValueError(
            f"prediction coverage mismatch: {len(report['missing_predictions'])} missing, "
            f"{len(report['extra_predictions'])} extra"
        )

    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
