#!/usr/bin/env python3
"""Create a Zindi submission file from test predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import default_raw_dir  # noqa: E402
from waxal.submission import make_submission_file  # noqa: E402
from waxal.utils import clean_name, utc_timestamp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fill-missing", default="")
    parser.add_argument("--empty-target", default=".")
    parser.add_argument("--no-sanitize", action="store_true")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = (
            Path("outputs/submissions")
            / f"submission_{clean_name(args.model_name)}_{utc_timestamp()}.csv"
        )
    result = make_submission_file(
        predictions_path=args.predictions,
        raw_dir=args.raw_dir,
        output_path=output,
        fill_missing=args.fill_missing,
        sanitize=not args.no_sanitize,
        empty_fallback=args.empty_target,
    )
    print(f"Wrote {result['num_rows']} rows to {result['output_path']}")
    if result["missing_predictions"]:
        print(f"WARNING: missing predictions filled: {len(result['missing_predictions'])}")
        print(result["missing_predictions"][:20])
    if result["extra_predictions"]:
        print(f"WARNING: extra prediction IDs ignored: {len(result['extra_predictions'])}")
        print(result["extra_predictions"][:20])
    if result["empty_targets_after_sanitize"]:
        print(
            "WARNING: empty targets remain after sanitization: "
            f"{len(result['empty_targets_after_sanitize'])}"
        )
        print(result["empty_targets_after_sanitize"][:20])


if __name__ == "__main__":
    main()
