from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning.clean_statslook import clean_statslook
from src.cleaning.clean_transfermarkt import build_transfermarkt_enrichment
from src.config import load_yaml_config
from src.data_sources.statslook_loader import find_statslook_source
from src.data_sources.transfermarkt_loader import (
    DEFAULT_TRANSFERMARKT_DIR,
    missing_transfermarkt_files,
)
from src.enrichment.current_club import add_current_club
from src.enrichment.entity_matching import match_players
from src.enrichment.team_league_mapping import apply_user_team_league_mapping
from src.features.league_adjustment import apply_league_strength_columns
from src.io_utils import write_table


ENRICHMENT_COLUMNS = [
    "tm_player_id",
    "tm_player_name",
    "nationality_name",
    "market_value_eur",
    "highest_market_value_eur",
    "market_value_date",
    "current_club_id",
    "current_club_name",
    "league_id",
    "league_name",
    "league_display_name",
    "league_country",
    "domestic_competition_id",
    "source_league_name",
    "league_mapping_team",
    "league_mapping_confidence",
    "league_source",
    "contract_until",
    "transfermarkt_url",
    "match_confidence",
    "match_method",
    "needs_review",
]


def _empty_enrichment_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ENRICHMENT_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["match_confidence"] = pd.to_numeric(
        result["match_confidence"], errors="coerce"
    ).fillna(0.0)
    result["match_method"] = result["match_method"].fillna("unmatched")
    result["needs_review"] = [
        True if pd.isna(value) else bool(value) for value in result["needs_review"]
    ]
    return result


def build_players_master(
    statslook_path: str | Path | None = None,
    transfermarkt_dir: str | Path | None = None,
    processed_dir: str | Path = "data/processed",
    allow_missing_transfermarkt: bool = True,
    season: str | None = None,
) -> pd.DataFrame:
    statslook_source = find_statslook_source(statslook_path)
    processed_dir = Path(processed_dir)
    statslook_clean = add_current_club(clean_statslook(source_path=statslook_source))
    if season:
        statslook_clean.insert(0, "season", season)
    write_table(statslook_clean, processed_dir / "statslook_clean.parquet")

    tm_dir = transfermarkt_dir or DEFAULT_TRANSFERMARKT_DIR
    missing = missing_transfermarkt_files(tm_dir)
    if missing:
        if not allow_missing_transfermarkt:
            formatted = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing Transfermarkt files:\n{formatted}")

        master = apply_user_team_league_mapping(_empty_enrichment_columns(statslook_clean))
        master = apply_league_strength_columns(master)
        write_table(master, processed_dir / "players_master.parquet")
        unmatched = master[["playerName", "displayName", "dateOfBirth", "teamName"]].copy()
        unmatched["reason"] = "transfermarkt_files_missing"
        write_table(unmatched, processed_dir / "unmatched_players.csv")
        return master

    transfermarkt_enrichment = build_transfermarkt_enrichment(tm_dir)
    write_table(transfermarkt_enrichment, processed_dir / "transfermarkt_enrichment.parquet")

    matching_config = load_yaml_config(
        "config/matching_config.yaml",
        default={
            "minimum_fuzzy_score": 90,
            "same_team_bonus": 5,
            "same_dob_bonus": 8,
            "review_confidence_threshold": 90,
        },
    )
    matches = match_players(statslook_clean, transfermarkt_enrichment, **matching_config)
    master = statslook_clean.merge(matches, on="statslook_row_id", how="left")
    master = master.merge(
        transfermarkt_enrichment.drop(
            columns=["tm_player_name_norm", "tm_current_club_name_norm"],
            errors="ignore",
        ),
        on="tm_player_id",
        how="left",
    )
    master = _empty_enrichment_columns(master)
    master = apply_user_team_league_mapping(master)
    master = apply_league_strength_columns(master)

    unmatched = master[master["match_method"].eq("unmatched")].copy()
    write_table(
        unmatched[["playerName", "displayName", "dateOfBirth", "teamName", "Position"]],
        processed_dir / "unmatched_players.csv",
    )
    write_table(master, processed_dir / "players_master.parquet")
    return master
