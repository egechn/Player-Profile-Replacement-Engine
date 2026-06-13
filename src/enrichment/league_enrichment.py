from __future__ import annotations

import pandas as pd

from src.config import load_yaml_config


def attach_league_info(
    players: pd.DataFrame,
    clubs: pd.DataFrame,
    competitions: pd.DataFrame,
) -> pd.DataFrame:
    result = players.copy()

    clubs_subset = clubs[
        [
            "club_id",
            "club_name_from_clubs",
            "club_domestic_competition_id",
            "club_total_market_value",
            "club_squad_size",
        ]
    ].copy()
    result = result.merge(
        clubs_subset,
        left_on="current_club_id",
        right_on="club_id",
        how="left",
    )

    result["current_club_name"] = result["current_club_name"].combine_first(
        result["club_name_from_clubs"]
    )
    result["domestic_competition_id"] = result["domestic_competition_id"].combine_first(
        result["club_domestic_competition_id"]
    )
    result["league_id"] = result["domestic_competition_id"]

    competitions_subset = competitions[
        ["league_id", "league_name", "league_country", "competition_type"]
    ].copy()
    result = result.merge(competitions_subset, on="league_id", how="left")
    result = apply_league_overrides(result)
    return result.drop(columns=["club_id"], errors="ignore")


def apply_league_overrides(players: pd.DataFrame) -> pd.DataFrame:
    result = players.copy()
    config = load_yaml_config(
        "config/league_overrides.yaml",
        {"competition_display_names": {}, "club_league_overrides": {}},
    )

    competition_display_names = {
        str(key): value for key, value in config.get("competition_display_names", {}).items()
    }
    club_overrides = {
        str(key): value for key, value in config.get("club_league_overrides", {}).items()
    }

    result["league_display_name"] = result["league_id"].map(competition_display_names)
    result["league_display_name"] = result["league_display_name"].combine_first(
        result["league_name"]
    )

    if club_overrides:
        club_ids = pd.to_numeric(result["current_club_id"], errors="coerce").astype("Int64")
        for idx, club_id in club_ids.items():
            override = club_overrides.get(str(club_id)) if pd.notna(club_id) else None
            if not override:
                continue
            for column in [
                "league_id",
                "league_name",
                "league_country",
                "domestic_competition_id",
            ]:
                if column in override:
                    result.at[idx, column] = override[column]
            result.at[idx, "league_display_name"] = override.get(
                "league_display_name", override.get("league_name")
            )

    return result
