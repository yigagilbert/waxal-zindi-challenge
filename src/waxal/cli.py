"""Console-script wrappers around the project scripts."""

from __future__ import annotations

import runpy
from pathlib import Path


def _run_script(script_name: str) -> None:
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "scripts" / script_name), run_name="__main__")


def audit_data() -> None:
    _run_script("audit_data.py")


def prepare_dataset() -> None:
    _run_script("prepare_dataset.py")


def check_gpu_env() -> None:
    _run_script("check_gpu_env.py")


def evaluate_predictions() -> None:
    _run_script("evaluate_predictions.py")


def make_submission() -> None:
    _run_script("make_submission.py")

