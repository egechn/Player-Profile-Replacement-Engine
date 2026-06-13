from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import load_yaml_config
from src.features.percentiles import add_percentiles
from src.features.per90 import add_per90


DEFAULT_PROFILE_WEIGHTS = {
    "ball_progressor_score": {
        "Successful_Dribbles_per90": 0.30,
        "Dribble_Attempts_per90": 0.20,
        "Passes_Final_Third_per90": 0.20,
        "Touches_per90": 0.15,
        "Through_Balls_Won_per90": 0.15,
    },
    "creator_score": {
        "Asist_per90": 0.15,
        "Key_Passes_per90": 0.25,
        "Chances_Created_per90": 0.25,
        "Big_Chances_Created_per90": 0.20,
        "Through_Balls_Won_per90": 0.15,
    },
    "finisher_score": {
        "Gol_per90": 0.25,
        "xG_per90": 0.25,
        "xGoT_per90": 0.20,
        "Shots_Total_per90": 0.15,
        "Shots_on_Target_per90": 0.15,
    },
    "presser_score": {
        "Tackles_per90": 0.25,
        "Tackles_Won_per90": 0.20,
        "Interception_per90": 0.20,
        "Ball_Recovery_per90": 0.25,
        "Fouls_per90": 0.10,
    },
    "ball_winner_score": {
        "Tackles_Won_per90": 0.25,
        "Interception_per90": 0.25,
        "Ball_Recovery_per90": 0.25,
        "Duels_Won_per90": 0.25,
    },
    "aerial_score": {
        "Aerials_Won_per90": 0.45,
        "Aerials_per90": 0.25,
        "aerial_win_rate": 0.20,
        "height": 0.10,
    },
    "possession_security_score": {
        "pass_accuracy": 0.30,
        "Accurate_Passes_per90": 0.20,
        "Possession_Lost_per90_inverse": 0.20,
        "Turn_Over_per90_inverse": 0.15,
        "Dispossed_per90_inverse": 0.15,
    },
    "wide_creator_score": {
        "Total_Crosses_per90": 0.20,
        "Accurate_Crosses_per90": 0.25,
        "Successful_Dribbles_per90": 0.20,
        "Key_Passes_per90": 0.20,
        "Chances_Created_per90": 0.15,
    },
    "box_threat_score": {
        "Gol_per90": 0.20,
        "xG_per90": 0.25,
        "xGoT_per90": 0.15,
        "Shots_Total_per90": 0.15,
        "Shots_on_Target_per90": 0.15,
        "Big_Chances_Created_per90": 0.10,
    },
    "deep_playmaker_score": {
        "Passes_per90": 0.20,
        "Accurate_Passes_per90": 0.20,
        "Passes_Final_Third_per90": 0.25,
        "Long_Balls_Won_per90": 0.20,
        "Through_Balls_Won_per90": 0.15,
    },
}

PROFILE_SCORE_COLUMNS = list(DEFAULT_PROFILE_WEIGHTS.keys())


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


def compute_profile_scores(
    df: pd.DataFrame,
    profile_weights: dict[str, dict[str, float]] | None = None,
    percentile_group_col: str = "position_group",
) -> pd.DataFrame:
    weights = profile_weights or load_profile_weights()
    result = add_per90(df)
    required_metrics = _required_metrics(weights)
    result = add_percentiles(result, required_metrics, group_col=percentile_group_col)

    for profile_name, metric_weights in weights.items():
        weighted_total = pd.Series(0.0, index=result.index)
        weight_sum = 0.0
        for metric_name, weight in metric_weights.items():
            base_metric, inverse = _base_metric(metric_name)
            percentile_col = f"{base_metric}_pct"
            if percentile_col not in result.columns:
                continue
            values = pd.to_numeric(result[percentile_col], errors="coerce").fillna(50)
            if inverse:
                values = 100 - values
            weighted_total = weighted_total + values * float(weight)
            weight_sum += float(weight)

        result[profile_name] = (weighted_total / weight_sum).round(2) if weight_sum else 0.0

    return result


def profile_vector_columns(df: pd.DataFrame | None = None) -> list[str]:
    if df is None:
        return PROFILE_SCORE_COLUMNS
    return [column for column in PROFILE_SCORE_COLUMNS if column in df.columns]

