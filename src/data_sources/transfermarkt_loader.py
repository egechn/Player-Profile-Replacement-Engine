from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io_utils import project_root, read_csv, resolve_project_path


DEFAULT_TRANSFERMARKT_DIR = project_root() / "data" / "raw" / "transfermarkt"

REQUIRED_TRANSFERMARKT_FILES = [
    "players.csv",
    "clubs.csv",
    "competitions.csv",
    "player_valuations.csv",
]

OPTIONAL_TRANSFERMARKT_FILES = [
    "appearances.csv",
    "transfers.csv",
]


def missing_transfermarkt_files(
    transfermarkt_dir: str | Path | None = None,
    required_files: list[str] | None = None,
) -> list[Path]:
    base = resolve_project_path(transfermarkt_dir or DEFAULT_TRANSFERMARKT_DIR)
    required = required_files or REQUIRED_TRANSFERMARKT_FILES
    return [base / filename for filename in required if not (base / filename).exists()]


def load_transfermarkt_csvs(
    transfermarkt_dir: str | Path | None = None,
    include_optional: bool = False,
) -> dict[str, pd.DataFrame]:
    base = resolve_project_path(transfermarkt_dir or DEFAULT_TRANSFERMARKT_DIR)
    missing = missing_transfermarkt_files(base)
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Transfermarkt enrichment files are missing. Add the Kaggle "
            f"player-scores CSVs to {base}:\n{formatted}"
        )

    filenames = list(REQUIRED_TRANSFERMARKT_FILES)
    if include_optional:
        filenames.extend(
            filename for filename in OPTIONAL_TRANSFERMARKT_FILES if (base / filename).exists()
        )

    return {Path(filename).stem: read_csv(base / filename) for filename in filenames}

