from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_yaml_config
from src.features.defensive_context import add_defensive_context
from src.features.league_adjustment import apply_league_translation
from src.features.percentiles import add_percentiles
from src.features.per90 import add_per90
from src.features.reliability import shrink_small_samples


# CIES-style performance areas, with one combined air area because StatsLook has
# no duel locations to split air defence from air attack. Main scores are
# percentiles against all outfield players, so they are comparable across
# positions; `<area>_role_score` holds the same score computed within the
# player's position group.
DEFAULT_PROFILE_WEIGHTS = {
    "ground_defence_score": {
        "Tackles_Won_per90_padj": 0.20,
        "Interception_per90_padj": 0.20,
        "Ball_Recovery_per90_padj": 0.15,
        "Blocked_Shots_per90_padj": 0.05,
        "duel_success_rate": 0.15,
        "tackle_success_rate": 0.10,
        "challenge_loss_rate_inverse": 0.15,
    },
    "air_score": {
        "Aerials_Won_per90": 0.45,
        "aerial_win_rate": 0.35,
        "height": 0.20,
    },
    "build_up_score": {
        "Passes_per90": 0.20,
        "Accurate_Passes_per90": 0.20,
        "pass_accuracy": 0.20,
        "Long_Balls_Won_per90": 0.20,
        "Possession_Lost_per90_inverse": 0.20,
    },
    "orchestration_score": {
        "Passes_Final_Third_per90": 0.40,
        "Through_Balls_Won_per90": 0.25,
        "Touches_per90": 0.20,
        "Key_Passes_per90": 0.15,
    },
    "take_on_score": {
        "Successful_Dribbles_per90": 0.40,
        "Dribble_Attempts_per90": 0.25,
        "dribble_success_rate": 0.15,
        "Fouls_Drawn_per90": 0.20,
    },
    "chance_creation_score": {
        "Key_Passes_per90": 0.20,
        "Chances_Created_per90": 0.20,
        "Big_Chances_Created_per90": 0.20,
        "Asist_per90": 0.15,
        "Accurate_Crosses_per90": 0.15,
        "Total_Crosses_per90": 0.10,
    },
    "finishing_score": {
        "Gol_per90": 0.25,
        "xG_per90": 0.25,
        "xGoT_per90": 0.20,
        "Shots_on_Target_per90": 0.15,
        "Shots_Total_per90": 0.15,
    },
}

PROFILE_SCORE_COLUMNS = list(DEFAULT_PROFILE_WEIGHTS.keys())

PROFILE_GROUPS = {
    "defence": ["ground_defence_score"],
    "air": ["air_score"],
    "possession": [
        "build_up_score",
        "orchestration_score",
        "take_on_score",
        "chance_creation_score",
    ],
    "attack": ["finishing_score"],
}

ROLE_SUFFIX = "_role"


def role_column(profile_column: str) -> str:
    return profile_column.replace("_score", f"{ROLE_SUFFIX}_score")


def profile_group(profile_column: str) -> str | None:
    for group, columns in PROFILE_GROUPS.items():
        if profile_column in columns:
            return group
    return None


def load_profile_weights() -> dict[str, dict[str, float]]:
    loaded = load_yaml_config("config/profile_weights.yaml", DEFAULT_PROFILE_WEIGHTS)
    return loaded or DEFAULT_PROFILE_WEIGHTS


def _base_metric(metric_name: str) -> tuple[str, bool]:
    if metric_name.endswith("_inverse"):
        return metric_name[: -len("_inverse")], True
    return metric_name, False


def _required_metrics(weights: dict[str, dict[str, float]]) -> list[str]:
    metrics: set[str] = set()
    for profile_weights in weights.values():
        for metric_name in profile_weights:
            metric, _ = _base_metric(metric_name)
            metrics.add(metric)
    return sorted(metrics)


def _weighted_scores(
    result: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    suffix: str,
    column_name,
) -> pd.DataFrame:
    for profile_name, metric_weights in weights.items():
        weighted_total = pd.Series(0.0, index=result.index)
        weight_sum = 0.0
        for metric_name, weight in metric_weights.items():
            base_metric, inverse = _base_metric(metric_name)
            percentile_col = f"{base_metric}{suffix}"
            if percentile_col not in result.columns:
                continue
            values = pd.to_numeric(result[percentile_col], errors="coerce").fillna(50)
            if inverse:
                values = 100 - values
            weighted_total = weighted_total + values * float(weight)
            weight_sum += float(weight)

        result[column_name(profile_name)] = (
            (weighted_total / weight_sum).round(2) if weight_sum else 0.0
        )
    return result


def compute_profile_scores(
    df: pd.DataFrame,
    profile_weights: dict[str, dict[str, float]] | None = None,
    percentile_group_col: str | None = None,
    role_group_col: str = "position_group",
) -> pd.DataFrame:
    weights = profile_weights or load_profile_weights()
    result = add_defensive_context(add_per90(df))
    required_metrics = _required_metrics(weights)

    # Percentiles are taken on league-translated outputs; the raw per-90
    # values stay in `result` for display.
    translated = apply_league_translation(result)
    # Few minutes, noisy numbers: pull them toward the position's typical value first.
    translated = shrink_small_samples(translated, required_metrics)
    if "league_strength_coefficient" not in result.columns:
        result = result.join(translated[["league_strength_coefficient"]])

    # Main scores: percentiles against all outfield players (or percentile_group_col if given).
    pool = translated
    if percentile_group_col is None and role_group_col in translated.columns:
        pool = translated[translated[role_group_col].ne("goalkeeper")]
    ranked = add_percentiles(pool, required_metrics, group_col=percentile_group_col)
    pct_cols = [f"{metric}_pct" for metric in required_metrics if f"{metric}_pct" in ranked.columns]
    result = result.drop(columns=[c for c in pct_cols if c in result.columns]).join(ranked[pct_cols])
    result = _weighted_scores(result, weights, "_pct", lambda name: name)

    # Role scores: the same areas ranked within the position group.
    role_ranked = add_percentiles(
        translated, required_metrics, group_col=role_group_col, suffix="_role_pct"
    )
    role_cols = [f"{metric}_role_pct" for metric in required_metrics if f"{metric}_role_pct" in role_ranked.columns]
    result = result.drop(columns=[c for c in role_cols if c in result.columns]).join(role_ranked[role_cols])
    result = _weighted_scores(result, weights, "_role_pct", role_column)

    return result


def profile_vector_columns(df: pd.DataFrame | None = None) -> list[str]:
    if df is None:
        return PROFILE_SCORE_COLUMNS
    return [column for column in PROFILE_SCORE_COLUMNS if column in df.columns]
