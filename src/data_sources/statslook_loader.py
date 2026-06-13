from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io_utils import project_root, read_csv, resolve_project_path


DEFAULT_STATSLOOK_PATH = (
    project_root() / "data" / "raw" / "statslook" / "Statslook_Scouting_Master_Final.csv"
)
ROOT_STATSLOOK_FALLBACK = project_root() / "Statslook_Scouting_Master_Final.csv"

REQUIRED_STATSLOOK_COLUMNS = [
    "playerName",
    "displayName",
    "dateOfBirth",
    "Detailed_Position",
    "Position",
    "teamName",
    "Min",
]


def find_statslook_source(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = resolve_project_path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Statslook file not found: {candidate}")
        return candidate

    if DEFAULT_STATSLOOK_PATH.exists():
        return DEFAULT_STATSLOOK_PATH
    if ROOT_STATSLOOK_FALLBACK.exists():
        return ROOT_STATSLOOK_FALLBACK
    raise FileNotFoundError(
        "Statslook source not found. Expected data/raw/statslook/"
        "Statslook_Scouting_Master_Final.csv"
    )


def load_statslook(path: str | Path | None = None) -> pd.DataFrame:
    source = find_statslook_source(path)
    df = read_csv(source)
    missing = [col for col in REQUIRED_STATSLOOK_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Statslook file is missing required columns: {missing}")
    return df

