from __future__ import annotations

import pandas as pd

from src.cleaning.normalize_names import normalize_person_name
from src.cleaning.normalize_teams import normalize_team_name
from src.data_sources.transfermarkt_loader import load_transfermarkt_csvs
from src.enrichment.league_enrichment import attach_league_info
from src.enrichment.market_value import resolve_market_values


def _series(df: pd.DataFrame, column: str, default=None) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def clean_transfermarkt_players(players: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=players.index)
    result["tm_player_id"] = pd.to_numeric(_series(players, "player_id"), errors="coerce")
    result["tm_player_name"] = _series(players, "name").fillna("")
    result["date_of_birth"] = pd.to_datetime(
        _series(players, "date_of_birth"), errors="coerce"
    )
    result["country_of_birth"] = _series(players, "country_of_birth")
    result["nationality_name"] = _series(players, "country_of_citizenship").combine_first(
        _series(players, "nationality")
    )
    result["nationality_name"] = result["nationality_name"].combine_first(
        result["country_of_birth"]
    )
    result["current_club_id"] = pd.to_numeric(
        _series(players, "current_club_id"), errors="coerce"
    )
    result["current_club_name"] = _series(players, "current_club_name")
    result["domestic_competition_id"] = _series(
        players, "current_club_domestic_competition_id"
    )
    result["market_value_eur"] = pd.to_numeric(
        _series(players, "market_value_in_eur"), errors="coerce"
    )
    result["highest_market_value_eur"] = pd.to_numeric(
        _series(players, "highest_market_value_in_eur"), errors="coerce"
    )
    result["contract_until"] = pd.to_datetime(
        _series(players, "contract_expiration_date"), errors="coerce"
    )
    result["transfermarkt_url"] = _series(players, "url")
    result["tm_player_name_norm"] = result["tm_player_name"].map(normalize_person_name)
    result["tm_current_club_name_norm"] = result["current_club_name"].map(
        normalize_team_name
    )
    return result.dropna(subset=["tm_player_id"]).drop_duplicates("tm_player_id")


def clean_transfermarkt_clubs(clubs: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=clubs.index)
    result["club_id"] = pd.to_numeric(_series(clubs, "club_id"), errors="coerce")
    result["club_name_from_clubs"] = _series(clubs, "name")
    result["club_domestic_competition_id"] = _series(clubs, "domestic_competition_id")
    result["club_total_market_value"] = pd.to_numeric(
        _series(clubs, "total_market_value"), errors="coerce"
    )
    result["club_squad_size"] = pd.to_numeric(_series(clubs, "squad_size"), errors="coerce")
    result["club_name_norm"] = result["club_name_from_clubs"].map(normalize_team_name)
    return result.dropna(subset=["club_id"]).drop_duplicates("club_id")


def clean_transfermarkt_competitions(competitions: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=competitions.index)
    result["league_id"] = _series(competitions, "competition_id")
    result["league_name"] = _series(competitions, "name")
    result["league_country"] = _series(competitions, "country_name")
    result["competition_type"] = _series(competitions, "type")
    return result.drop_duplicates("league_id")


def build_transfermarkt_enrichment_from_frames(
    players: pd.DataFrame,
    clubs: pd.DataFrame,
    competitions: pd.DataFrame,
    player_valuations: pd.DataFrame,
) -> pd.DataFrame:
    clean_players = clean_transfermarkt_players(players)
    clean_clubs = clean_transfermarkt_clubs(clubs)
    clean_competitions = clean_transfermarkt_competitions(competitions)
    enriched = resolve_market_values(clean_players, player_valuations)
    enriched = attach_league_info(enriched, clean_clubs, clean_competitions)
    return enriched


def build_transfermarkt_enrichment(transfermarkt_dir=None) -> pd.DataFrame:
    raw = load_transfermarkt_csvs(transfermarkt_dir)
    return build_transfermarkt_enrichment_from_frames(
        players=raw["players"],
        clubs=raw["clubs"],
        competitions=raw["competitions"],
        player_valuations=raw["player_valuations"],
    )

