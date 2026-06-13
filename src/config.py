from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io_utils import project_root, resolve_project_path


CONFIG_DIR = project_root() / "config"


def load_yaml_config(
    filename: str | Path,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = resolve_project_path(filename)
    if not path.exists() and not str(filename).startswith("config/"):
        path = CONFIG_DIR / str(filename)

    if not path.exists():
        return default or {}

    try:
        import yaml
    except ImportError:
        return default or {}

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded

