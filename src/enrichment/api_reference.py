"""Fill gaps in a season export from StatsLook's API.

The legacy 2025-26 export has no StatsLook ids, empty or national-team clubs for
some players and a few goalkeepers listed as outfield players. A reference
pulled from the API (`python -m src.data_sources.statslook_fetch --reference
--season 2025-26`) fills only those gaps; stats and the list of players are
left as they are.
"""
from __future__ import annotations

import json

import pandas as pd

from src.cleaning.normalize_names import normalize_person_name
from src.config import load_yaml_config
from src.seasons import STATSLOOK_RAW_DIR, base_season

REFERENCE_DIR = STATSLOOK_RAW_DIR / "reference"


def reference_path(season: str):
    return REFERENCE_DIR / f"statslook_{base_season(season).replace('-', '_')}_api_reference.csv"


def _countries() -> set[str]:
    names: set[str] = set()
    for metadata in STATSLOOK_RAW_DIR.glob("*/page_metadata.json"):
        names |= {country["name"] for country in json.loads(metadata.read_text())["countries"]}
    return names


def _keys(df: pd.DataFrame, name_column: str) -> pd.Series:
    birth = pd.to_datetime(df["dateOfBirth"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return df[name_column].map(normalize_person_name) + "|" + birth


def fill_from_api_reference(df: pd.DataFrame, season: str | None) -> pd.DataFrame:
    path = reference_path(season) if season else None
    if path is None or not path.exists():
        return df
    reference = pd.read_csv(path, low_memory=False)
    result = df.copy()

    # Match on name + birth date (full name first, then short name); ambiguous keys are skipped.
    matched = pd.DataFrame(index=result.index, columns=reference.columns, dtype=object)
    for name_column in ["playerName", "displayName"]:
        keys = reference.assign(_key=_keys(reference, name_column)).drop_duplicates("_key", keep=False)
        row_keys = _keys(result, name_column)
        missing = matched["playerId"].isna()
        found = keys.set_index("_key").reindex(row_keys[missing])
        matched.loc[missing] = found.set_axis(row_keys[missing].index).values

    if "playerId" not in result.columns:
        result["playerId"] = pd.NA
    result["playerId"] = result["playerId"].fillna(matched["playerId"])
    result["statslook_league_id"] = pd.to_numeric(matched["league_id"], errors="coerce")

    # Club: only where the export has none or a national team.
    countries = _countries()
    no_club = result["teamName"].isna() | result["teamName"].isin(countries)
    league_team = matched["league_team"].where(~matched["league_team"].isin(countries))
    api_team = matched["teamName"].where(~matched["teamName"].isin(countries))
    club = league_team.fillna(api_team)
    club_id = matched["league_team_id"].where(league_team.notna(), matched["teamId"].where(api_team.notna()))
    fill = no_club & club.notna()
    result.loc[fill, "teamName"] = club[fill]
    if "teamId" in result.columns:
        result.loc[fill, "teamId"] = pd.to_numeric(club_id[fill], errors="coerce")

    # Goalkeepers the export gives an outfield detailed position (e.g. Position
    # "Goalkeeper" with Detailed_Position "LB"): the detailed one would win.
    keeper = (matched["Position"].eq("Goalkeeper") | result["Position"].eq("Goalkeeper")) & result[
        "Detailed_Position"
    ].ne("GK")
    result.loc[keeper, "Position"] = "Goalkeeper"
    result.loc[keeper, "Detailed_Position"] = "GK"

    print(
        f"{season}: API reference matched {matched['playerId'].notna().sum():,} of {len(result):,} players; "
        f"club filled for {fill.sum():,}, {keeper.sum():,} goalkeepers corrected."
    )
    return result


def fill_leagues_from_statslook(df: pd.DataFrame) -> pd.DataFrame:
    """Players still without a league get the domestic league StatsLook lists them in."""
    if "statslook_league_id" not in df.columns:
        return df
    names = {int(k): v for k, v in (load_yaml_config("config/statslook_leagues.yaml", {}) or {}).items()}
    result = df.copy()
    league = pd.to_numeric(result["statslook_league_id"], errors="coerce").map(names)
    missing = result["league_display_name"].isna() & league.notna()
    result.loc[missing, "league_display_name"] = league[missing]
    if "league_source" in result.columns:
        result.loc[missing, "league_source"] = "statslook_league"
    return result
