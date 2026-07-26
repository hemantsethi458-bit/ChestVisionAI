"""Safe file I/O helpers."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from utils.paths import ensure_dir


def read_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write a dictionary to a JSON file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)


def read_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write a dictionary to a YAML file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
