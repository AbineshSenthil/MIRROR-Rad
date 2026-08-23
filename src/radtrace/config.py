"""Configuration helpers shared by command-line tools and the dashboard."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load the JSON configuration and resolve project-relative paths."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config = deepcopy(config)
    if overrides:
        config = _merge(config, overrides)
    config["project_root"] = str(PROJECT_ROOT)
    for key, raw_path in config.get("paths", {}).items():
        candidate = Path(raw_path)
        config["paths"][key] = str(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)
    return config


def ensure_output_directories(config: Mapping[str, Any]) -> None:
    """Create only generated-output directories; never modify the source XML directory."""
    for key in ("processed", "splits", "artifacts", "reports"):
        value = config.get("paths", {}).get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)

