"""Small local-first utilities used by scripts."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_timestamp() -> str:
    """Return a filesystem-friendly UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_dump(data: Any, path: str | Path) -> None:
    """Write JSON with stable formatting."""
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def clean_name(value: str) -> str:
    """Make a short filesystem-safe name from a model or run label."""
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def git_commit() -> str | None:
    """Return the current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def repo_root_from_script(script_file: str | Path) -> Path:
    """Resolve the repository root from a script path."""
    return Path(script_file).resolve().parents[1]


def save_experiment_log(
    output_dir: str | Path,
    *,
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> Path:
    """Save a local JSON experiment record."""
    out_dir = ensure_dir(output_dir)
    payload = {
        "run_name": run_name,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "config": config,
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "environment": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }
    out_path = out_dir / f"{clean_name(run_name)}_{payload['timestamp_utc']}.json"
    json_dump(payload, out_path)
    return out_path

