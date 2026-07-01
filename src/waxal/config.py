"""Configuration helpers for WAXAL ASR experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for config files. Install requirements.txt first."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def merge_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-recursive merge of config and non-None overrides."""
    merged = dict(config)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged

