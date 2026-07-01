"""Submission helpers for WAXAL Zindi predictions."""

from __future__ import annotations

from pathlib import Path

from .data import load_sample_submission, read_prediction_csv, write_csv_rows


def align_predictions_to_sample(
    predictions: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
    *,
    fill_missing: str = "",
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Reorder predictions to match SampleSubmission.csv exactly."""
    pred_by_id = {row["ID"]: row.get("Target", "") for row in predictions}
    sample_ids = [row["ID"] for row in sample_rows]
    missing = [example_id for example_id in sample_ids if example_id not in pred_by_id]
    extra = sorted(set(pred_by_id) - set(sample_ids))
    aligned = [
        {"ID": example_id, "Target": pred_by_id.get(example_id, fill_missing)}
        for example_id in sample_ids
    ]
    return aligned, missing, extra


def make_submission_file(
    *,
    predictions_path: str | Path,
    raw_dir: str | Path,
    output_path: str | Path,
    fill_missing: str = "",
) -> dict:
    """Create a Zindi submission from a prediction CSV."""
    predictions = read_prediction_csv(predictions_path)
    sample_rows = load_sample_submission(raw_dir)
    aligned, missing, extra = align_predictions_to_sample(
        predictions, sample_rows, fill_missing=fill_missing
    )
    write_csv_rows(output_path, aligned, ["ID", "Target"])
    return {
        "output_path": str(output_path),
        "num_rows": len(aligned),
        "missing_predictions": missing,
        "extra_predictions": extra,
    }

