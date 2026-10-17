from __future__ import annotations

import re
from pathlib import Path

from src.config import load_yaml_config
from src.io_utils import project_root


STATSLOOK_RAW_DIR = project_root() / "data" / "raw" / "statslook"
PROCESSED_DIR = project_root() / "data" / "processed"
LEGACY_STATSLOOK_FILE = STATSLOOK_RAW_DIR / "Statslook_Scouting_Master_Final.csv"
LEGACY_MASTER_FILE = PROCESSED_DIR / "players_master.parquet"
SEASON_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def load_season_config() -> dict:
    return load_yaml_config("config/seasons.yaml", {"legacy_season": "2025-26", "default_season": ""})


def legacy_season() -> str:
    return str(load_season_config().get("legacy_season") or "2025-26")


def _season_folders(base: Path) -> list[str]:
    if not base.exists():
        return []
    return [
        child.name
        for child in base.iterdir()
        if child.is_dir() and SEASON_PATTERN.match(child.name)
    ]


def raw_seasons() -> list[str]:
    seasons = {
        season
        for season in _season_folders(STATSLOOK_RAW_DIR)
        if any((STATSLOOK_RAW_DIR / season).glob("*.csv"))
    }
    if LEGACY_STATSLOOK_FILE.exists():
        seasons.add(legacy_season())
    return sorted(seasons)


def available_seasons() -> list[str]:
    """Seasons that have a processed master table or raw data to build one."""
    seasons = {
        season
        for season in _season_folders(PROCESSED_DIR)
        if (PROCESSED_DIR / season / "players_master.parquet").exists()
    }
    if LEGACY_MASTER_FILE.exists():
        seasons.add(legacy_season())
    return sorted(seasons | set(raw_seasons()))


def default_season() -> str | None:
    configured = str(load_season_config().get("default_season") or "")
    seasons = available_seasons()
    if configured and configured in seasons:
        return configured
    return seasons[-1] if seasons else None


def statslook_source_for(season: str) -> Path:
    folder = STATSLOOK_RAW_DIR / season
    csvs = sorted(folder.glob("*.csv")) if folder.exists() else []
    if csvs:
        if len(csvs) > 1:
            raise ValueError(f"Expected one StatsLook CSV in {folder}, found {len(csvs)}.")
        return csvs[0]
    if season == legacy_season() and LEGACY_STATSLOOK_FILE.exists():
        return LEGACY_STATSLOOK_FILE
    raise FileNotFoundError(f"No StatsLook CSV for season {season} in {folder}.")


def processed_dir_for(season: str) -> Path:
    return PROCESSED_DIR / season


def master_path_for(season: str) -> Path:
    path = processed_dir_for(season) / "players_master.parquet"
    if not path.exists() and season == legacy_season() and LEGACY_MASTER_FILE.exists():
        return LEGACY_MASTER_FILE
    return path


def load_season_team_leagues(season: str | None) -> dict[str, str]:
    if not season:
        return {}
    config = load_yaml_config(f"config/season_team_leagues/{season}.yaml", {})
    return {str(team): str(league) for team, league in (config or {}).items() if league}


def default_min_minutes(season: str, minutes) -> int:
    """Default minutes floor for a season: config value, else 20% of the
    90th-percentile player's minutes, rounded to 50."""
    configured = (load_season_config().get("min_minutes") or {}).get(season)
    if configured not in (None, "", "auto"):
        return int(configured)
    import pandas as pd

    values = pd.to_numeric(pd.Series(minutes), errors="coerce").dropna()
    if values.empty:
        return 0
    return int(round(values.quantile(0.9) * 0.2 / 50) * 50)
