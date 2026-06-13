from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from src.cleaning.normalize_teams import normalize_team_name
from src.config import load_yaml_config


TEAM_MATCH_THRESHOLD = 90.0


def _team_similarity(left: str, right: str) -> float:
    left = left or ""
    right = right or ""
    if not left or not right:
        return 0.0
    if left == right:
        return 100.0

    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens and right_tokens:
        if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
            return 100.0
        overlap = len(left_tokens & right_tokens) / max(min(len(left_tokens), len(right_tokens)), 1)
        if overlap >= 0.75:
            return 90.0

    try:
        from rapidfuzz import fuzz

        return float(fuzz.token_sort_ratio(left, right))
    except ImportError:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, left, right).ratio() * 100


@lru_cache(maxsize=1)
def _load_team_league_mapping() -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, dict[str, Any]]]]:
    config = load_yaml_config("config/team_league_mapping.yaml", default={"teams": {}})
    teams = config.get("teams", {})
    exact: dict[str, dict[str, Any]] = {}
    candidates: list[tuple[str, str, dict[str, Any]]] = []

    for team_name, info in teams.items():
        normalized = normalize_team_name(team_name)
        enriched_info = {**info, "league_mapping_team": team_name}
        exact[normalized] = enriched_info
        candidates.append((team_name, normalized, enriched_info))

    return exact, candidates


@lru_cache(maxsize=4096)
def _resolve_normalized_team(normalized: str) -> dict[str, Any] | None:
    exact, candidates = _load_team_league_mapping()
    if normalized in exact:
        return {**exact[normalized], "league_mapping_confidence": 100.0}

    best_info = None
    best_score = 0.0
    for _, candidate_normalized, info in candidates:
        score = _team_similarity(normalized, candidate_normalized)
        if score > best_score:
            best_score = score
            best_info = info

    if best_info and best_score >= TEAM_MATCH_THRESHOLD:
        return {**best_info, "league_mapping_confidence": round(best_score, 2)}
    return None


def resolve_team_league(team_name: Any) -> dict[str, Any] | None:
    normalized = normalize_team_name(team_name)
    if not normalized:
        return None
    return _resolve_normalized_team(normalized)


def apply_user_team_league_mapping(
    df: pd.DataFrame,
    team_col: str = "teamName",
) -> pd.DataFrame:
    result = df.copy()
    raw_transfermarkt_columns = {
        "league_id": "tm_league_id",
        "league_name": "tm_league_name",
        "league_display_name": "tm_league_display_name",
        "league_country": "tm_league_country",
        "domestic_competition_id": "tm_domestic_competition_id",
    }
    for source_col, raw_col in raw_transfermarkt_columns.items():
        if source_col in result.columns and raw_col not in result.columns:
            result[raw_col] = result[source_col]

    added_columns = [
        "league_id",
        "league_name",
        "league_display_name",
        "league_country",
        "domestic_competition_id",
        "source_league_name",
        "league_mapping_team",
        "league_mapping_confidence",
        "league_source",
    ]
    for column in added_columns:
        if column not in result.columns:
            result[column] = pd.NA

    authoritative_columns = [
        "league_id",
        "league_name",
        "league_display_name",
        "league_country",
        "domestic_competition_id",
        "source_league_name",
        "league_mapping_team",
        "league_mapping_confidence",
    ]
    for column in authoritative_columns:
        result[column] = pd.NA
    result["league_source"] = "unmapped"

    for idx, row in result.iterrows():
        team_value = row.get(team_col)
        info = resolve_team_league(team_value)
        source = "user_team_mapping"
        if info is None:
            continue

        for column in [
            "league_id",
            "league_name",
            "league_display_name",
            "league_country",
            "domestic_competition_id",
            "source_league_name",
            "league_mapping_team",
            "league_mapping_confidence",
        ]:
            result.at[idx, column] = info.get(column)
        result.at[idx, "league_source"] = source

    return result
