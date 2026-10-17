from __future__ import annotations

import re

import pandas as pd

from src.cleaning.normalize_teams import normalize_team_name
from src.enrichment.entity_matching import team_similarity_score
from src.features.reliability import sample_indicator


# Fuzzy Transfermarkt matches can be the wrong player; their names get a ⚠.
UNCERTAIN_MATCH_PREFIX = "fuzzy_"
MATCHING_AREAS_PATTERN = re.compile(r"Matches target strengths in (.*?)\.")
# Below this, the StatsLook club and the Transfermarkt club are different clubs.
SAME_CLUB_MIN_SCORE = 80
UNKNOWN_CONTRACT = "?"


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def _player_names(df: pd.DataFrame) -> pd.Series:
    names = df["displayName"] if "displayName" in df.columns else df["playerName"]
    names = names.fillna(df["playerName"]).astype(str)
    if "match_method" in df.columns:
        uncertain = df["match_method"].astype(str).str.startswith(UNCERTAIN_MATCH_PREFIX)
        names = names.where(~uncertain, names + " ⚠")
    return names


def _matching_areas(df: pd.DataFrame) -> pd.Series:
    if "similarity_reason" not in df.columns:
        return pd.Series("", index=df.index)
    return df["similarity_reason"].astype(str).str.extract(MATCHING_AREAS_PATTERN)[0].fillna("")


def _current_teams(df: pd.DataFrame) -> pd.Series:
    """Today's club, whatever season the stats are from."""
    if "current_team" in df.columns:
        return df["current_team"].fillna(df.get("teamName"))
    return df.get("teamName", pd.Series(pd.NA, index=df.index))


def _contract_years(df: pd.DataFrame) -> pd.Series:
    """Contract end year from Transfermarkt, or "?" when it can't be trusted:
    the contract has already ended, or the player has since moved (the current
    club from StatsLook is not the club Transfermarkt has the contract with)."""
    if "contract_until" not in df.columns:
        return pd.Series("", index=df.index)
    contract = pd.to_datetime(df["contract_until"], errors="coerce")
    # Text, so the year isn't shown with a thousands separator (2,029).
    years = contract.dt.year.astype("Int64").astype("string").fillna("")
    unknown = contract.notna() & (contract < pd.Timestamp.today().normalize())
    if "current_club_name" in df.columns:
        moved = [
            pd.notna(team) and pd.notna(club)
            and team_similarity_score(normalize_team_name(team), normalize_team_name(club)) < SAME_CLUB_MIN_SCORE
            for team, club in zip(_current_teams(df), df["current_club_name"])
        ]
        unknown |= contract.notna() & pd.Series(moved, index=df.index)
    return years.where(~unknown, UNKNOWN_CONTRACT)


def focus_areas(df: pd.DataFrame) -> str:
    """The target's profile areas the search was based on (the same for every row)."""
    if "similarity_basis" not in df.columns or df.empty:
        return ""
    return str(df["similarity_basis"].iloc[0])


def format_player_table(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    result["Player"] = _player_names(df)
    result["Current club"] = _current_teams(df)
    # League of the season the stats are from: the league strength adjustment uses it.
    result["League (stats)"] = df.get("league_display_name")
    result["League strength"] = _numeric(df, "league_strength_coefficient").round(2)
    result["Age"] = _numeric(df, "age").round(0).astype("Int64")
    result["Pos"] = df.get("Detailed_Position")
    result["Minutes"] = _numeric(df, "Min").round(0).astype("Int64")
    result["Sample"] = sample_indicator(_numeric(df, "Min"))
    result["Value (€m)"] = (_numeric(df, "market_value_eur") / 1_000_000).round(1)
    result["Contract"] = _contract_years(df)
    similarity = "league_adjusted_similarity_score"
    if similarity not in df.columns:
        similarity = "similarity_score"
    result["Similarity %"] = (_numeric(df, similarity).clip(upper=1) * 100).round(0).astype("Int64")
    if "replacement_score" in df.columns:
        result["Replacement score"] = _numeric(df, "replacement_score").round(1)
    result["Matching areas"] = _matching_areas(df)
    return result.reset_index(drop=True)
