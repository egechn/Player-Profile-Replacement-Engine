from __future__ import annotations

import pandas as pd


DISPLAY_COLUMNS = [
    "playerName",
    "teamName",
    "league_display_name",
    "league_source",
    "opta_league_rank",
    "opta_league_average_rating",
    "league_strength_coefficient",
    "league_strength_gap",
    "league_adjustment_factor",
    "age",
    "Position",
    "Detailed_Position",
    "Min",
    "market_value_eur",
    "contract_until",
    "similarity_score",
    "league_adjusted_similarity_score",
    "replacement_score",
    "match_confidence",
    "match_method",
    "needs_review",
    "recommendation_reason",
]


def format_player_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DISPLAY_COLUMNS if column in df.columns]
    result = df[columns].copy()
    for column in [
        "similarity_score",
        "league_adjusted_similarity_score",
        "league_strength_coefficient",
        "league_strength_gap",
        "league_adjustment_factor",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(3)
    for column in [
        "replacement_score",
        "match_confidence",
        "opta_league_average_rating",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(2)
    if "opta_league_rank" in result.columns:
        result["opta_league_rank"] = pd.to_numeric(
            result["opta_league_rank"], errors="coerce"
        ).astype("Int64")
    if "market_value_eur" in result.columns:
        result["market_value_eur"] = pd.to_numeric(
            result["market_value_eur"], errors="coerce"
        ).round(0)
    return result
