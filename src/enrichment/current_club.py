from __future__ import annotations

import pandas as pd

from src.cleaning.normalize_names import normalize_person_name
from src.seasons import raw_seasons, statslook_source_for


CURRENT_CLUB_COLUMNS = ["current_team_id", "current_team"]


def _birth_key(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")


def latest_season_clubs() -> pd.DataFrame:
    """Club of every player in the latest season's StatsLook pull: their current club.
    (The fetcher already swaps national-team entries for the player's club.)"""
    seasons = raw_seasons()
    if not seasons:
        return pd.DataFrame(columns=["playerName", "displayName", "dateOfBirth", *CURRENT_CLUB_COLUMNS])
    clubs = pd.read_csv(statslook_source_for(seasons[-1]), low_memory=False)
    clubs = clubs.rename(columns={"teamId": "current_team_id", "teamName": "current_team"})
    keep = [column for column in ["playerId", "playerName", "displayName", "dateOfBirth"] if column in clubs.columns]
    return clubs[keep + CURRENT_CLUB_COLUMNS].dropna(subset=["current_team"])


def add_current_club(df: pd.DataFrame, clubs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add each player's current club: their club in the latest season.

    Rows are matched on StatsLook playerId when both files have it, else on
    name + date of birth (the legacy 2025-26 file has no ids). Players not in
    the latest season keep the team of their own season.
    """
    result = df.drop(columns=CURRENT_CLUB_COLUMNS, errors="ignore")
    clubs = latest_season_clubs() if clubs is None else clubs
    clubs["birth"] = _birth_key(clubs["dateOfBirth"])
    current = pd.DataFrame(index=result.index, columns=CURRENT_CLUB_COLUMNS, dtype=object)

    if "playerId" in result.columns and "playerId" in clubs.columns:
        by_id = clubs.drop_duplicates("playerId").set_index("playerId")[CURRENT_CLUB_COLUMNS]
        current.update(by_id.reindex(pd.to_numeric(result["playerId"], errors="coerce")).set_axis(result.index))

    birth = _birth_key(result["dateOfBirth"])
    for name_column in ["playerName", "displayName"]:
        keys = clubs.assign(key=clubs[name_column].map(normalize_person_name) + "|" + clubs["birth"])
        keys = keys.drop_duplicates("key", keep=False).set_index("key")[CURRENT_CLUB_COLUMNS]
        row_keys = result[name_column].map(normalize_person_name) + "|" + birth
        missing = current["current_team"].isna()
        current.loc[missing] = keys.reindex(row_keys[missing]).set_axis(row_keys[missing].index).values

    result["current_team_id"] = current["current_team_id"].fillna(result.get("teamId"))
    result["current_team"] = current["current_team"].fillna(result.get("teamName"))
    return result
