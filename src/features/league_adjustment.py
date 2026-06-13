from __future__ import annotations

import pandas as pd

from src.cleaning.normalize_names import normalize_text
from src.config import load_yaml_config


def load_league_strength_config() -> dict:
    return load_yaml_config(
        "config/league_strength.yaml",
        {
            "default_strength": 0.75,
            "league_similarity_weight": 0.10,
            "leagues": {},
            "league_metadata": {},
        },
    )


def load_league_strengths() -> tuple[float, dict[str, float]]:
    config = load_league_strength_config()
    leagues = {
        normalize_text(key): float(value)
        for key, value in config.get("leagues", {}).items()
        if value is not None
    }
    return float(config.get("default_strength", 0.75)), leagues


def load_league_strength_metadata() -> dict[str, dict]:
    config = load_league_strength_config()
    return {
        normalize_text(key): value or {}
        for key, value in config.get("league_metadata", {}).items()
    }


def league_similarity_weight() -> float:
    config = load_league_strength_config()
    return float(config.get("league_similarity_weight", 0.10))


def _select_league_column(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    candidates = [preferred, "league_display_name", "league_name", "source_league_name"]
    for column in candidates:
        if column and column in df.columns:
            return column
    return None


def apply_league_strength_columns(
    df: pd.DataFrame,
    league_col: str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    selected_col = _select_league_column(result, league_col)
    default_strength, strengths = load_league_strengths()
    metadata = load_league_strength_metadata()

    if selected_col is None:
        league_keys = pd.Series("", index=result.index)
    else:
        league_keys = result[selected_col].map(normalize_text)

    result["league_strength_coefficient"] = league_keys.map(strengths).fillna(default_strength)
    result["league_strength_source"] = league_keys.map(
        lambda value: "opta_power_rankings" if value in strengths else "default"
    )
    result["opta_league_name"] = league_keys.map(
        lambda value: metadata.get(value, {}).get("opta_league_name", pd.NA)
    )
    result["opta_league_country"] = league_keys.map(
        lambda value: metadata.get(value, {}).get("opta_country", pd.NA)
    )
    result["opta_league_rank"] = league_keys.map(
        lambda value: metadata.get(value, {}).get("opta_global_rank", pd.NA)
    )
    result["opta_league_average_rating"] = league_keys.map(
        lambda value: metadata.get(value, {}).get("opta_average_rating", pd.NA)
    )
    result["opta_league_top10_rating"] = league_keys.map(
        lambda value: metadata.get(value, {}).get("opta_top10_rating", pd.NA)
    )
    result["opta_league_top5_rating"] = league_keys.map(
        lambda value: metadata.get(value, {}).get("opta_top5_rating", pd.NA)
    )
    return result


def load_legacy_league_strengths() -> tuple[float, dict[str, float]]:
    config = load_yaml_config(
        "config/league_strength.yaml",
        {"default_strength": 0.75, "leagues": {}},
    )
    leagues = {normalize_text(key): float(value) for key, value in config.get("leagues", {}).items()}
    return float(config.get("default_strength", 0.75)), leagues


def apply_league_strength_adjustment(
    df: pd.DataFrame,
    score_columns: list[str],
    league_col: str | None = None,
) -> pd.DataFrame:
    result = apply_league_strength_columns(df, league_col=league_col)
    multiplier = pd.to_numeric(
        result["league_strength_coefficient"], errors="coerce"
    ).fillna(load_league_strength_config().get("default_strength", 0.75))

    for column in score_columns:
        if column in result.columns:
            result[f"{column}_league_adjusted"] = (
                pd.to_numeric(result[column], errors="coerce").fillna(0) * multiplier
            ).clip(0, 100)
    return result
