from __future__ import annotations

import pandas as pd

from src.cleaning.normalize_names import normalize_text
from src.config import load_yaml_config


# Per-90 outputs scaled by league strength before percentiles. Volume of
# involvement (passes attempted, touches, dribble attempts), rates and physical
# traits are left as they are: they say how a player plays, not how good the
# opposition was.
LEAGUE_TRANSLATED_METRICS = [
    "Gol_per90",
    "xG_per90",
    "xGoT_per90",
    "Shots_Total_per90",
    "Shots_on_Target_per90",
    "Asist_per90",
    "Key_Passes_per90",
    "Chances_Created_per90",
    "Big_Chances_Created_per90",
    "Accurate_Crosses_per90",
    "Successful_Dribbles_per90",
    "Passes_Final_Third_per90",
    "Through_Balls_Won_per90",
    "Accurate_Passes_per90",
    "Long_Balls_Won_per90",
    "Fouls_Drawn_per90",
    "Aerials_Won_per90",
    "Tackles_Won_per90_padj",
    "Interception_per90_padj",
    "Ball_Recovery_per90_padj",
    "Blocked_Shots_per90_padj",
]


def load_league_strength_config() -> dict:
    return load_yaml_config(
        "config/league_strength.yaml",
        {
            "default_strength": 0.75,
            "league_similarity_weight": 0.0,
            "metric_translation_exponent": 2.0,
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
    return float(config.get("league_similarity_weight", 0.0))


def load_inferred_team_leagues() -> dict[str, str]:
    config = load_yaml_config("config/team_league_mapping_inferred.yaml", {"teams": {}})
    return {
        team: info.get("league_display_name")
        for team, info in (config.get("teams") or {}).items()
        if info and info.get("league_display_name")
    }


def fill_inferred_leagues(df: pd.DataFrame, team_col: str = "teamName") -> pd.DataFrame:
    """Fill missing leagues from config/team_league_mapping_inferred.yaml and
    apply config/season_team_leagues/<season>.yaml overrides."""
    if team_col not in df.columns:
        return df
    result = df.copy()
    inferred = result[team_col].map(load_inferred_team_leagues())
    if "league_display_name" not in result.columns:
        result["league_display_name"] = pd.NA
    missing = result["league_display_name"].isna() & inferred.notna()
    result.loc[missing, "league_display_name"] = inferred[missing]
    if "league_source" in result.columns:
        result.loc[missing, "league_source"] = "inferred_team_mapping"

    # Per-season overrides (promotions, relegations) win over every other mapping.
    if "season" in result.columns:
        from src.seasons import load_season_team_leagues

        for season in result["season"].dropna().unique():
            overrides = load_season_team_leagues(str(season))
            if not overrides:
                continue
            in_season = result["season"].eq(season)
            override = result[team_col].map(overrides)
            hit = in_season & override.notna()
            result.loc[hit, "league_display_name"] = override[hit]
            if "league_source" in result.columns:
                result.loc[hit, "league_source"] = "season_override"
    return result


def apply_league_translation(df: pd.DataFrame) -> pd.DataFrame:
    """Scale per-90 outputs by league strength (coefficient ** exponent)."""
    result = df if "league_strength_coefficient" in df.columns else apply_league_strength_columns(df)
    result = result.copy()
    exponent = float(load_league_strength_config().get("metric_translation_exponent", 2.0))
    factor = pd.to_numeric(result["league_strength_coefficient"], errors="coerce").fillna(
        float(load_league_strength_config().get("default_strength", 0.75))
    ) ** exponent
    for column in LEAGUE_TRANSLATED_METRICS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0) * factor
    return result


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
    result = fill_inferred_leagues(df)
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
