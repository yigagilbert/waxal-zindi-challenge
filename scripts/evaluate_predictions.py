#!/usr/bin/env python3
"""Evaluate ID,Target predictions against validation references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_prediction_csv, references_from_validation_csv  # noqa: E402
from waxal.scoring import score_records  # noqa: E402
from waxal.text_normalization import POLICIES  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--normalization", choices=[*POLICIES, "all"], default="starter_lower")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    predictions = read_prediction_csv(args.predictions)
    references = references_from_validation_csv(args.references)
    policies = POLICIES if args.normalization == "all" else (args.normalization,)
    results = {
        policy: score_records(references, predictions, normalization=policy)
        for policy in policies
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output:
        json_dump(results, args.output)


if __name__ == "__main__":
    main()

