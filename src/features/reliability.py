"""Small-sample correction for per-90 metrics and rates.

A player with 30 minutes and one goal scores 3 goals per 90. Before ranking,
each metric is pulled toward the typical value for the player's position:

    adjusted = (value * minutes + position_typical * K) / (minutes + K)

With K = 450, a player on 450 minutes sits halfway between their own value and
the position's; at 2,700 minutes the player's own value carries 86%. Only the
values used for percentiles are adjusted; raw per-90 values stay for display.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_yaml_config


DEFAULT_RELIABILITY = {
    # Minutes of "position average" mixed into every player's numbers.
    "shrinkage_minutes": 450,
    # Players with at least this many minutes define a position's typical value.
    "prior_min_minutes": 900,
    # Sample indicator bands shown in the tables: minutes for ●●● and ●●.
    "sample_bands": [900, 450],
}

# Physical attributes, not performance: no sample-size issue.
UNADJUSTED_METRICS = {"height"}


def load_reliability_config() -> dict:
    loaded = load_yaml_config("config/reliability.yaml", {}) or {}
    return {**DEFAULT_RELIABILITY, **loaded}


def _position_typical(
    values: pd.Series, minutes: pd.Series, groups: pd.Series | None, prior_min_minutes: float
) -> pd.Series:
    established = minutes >= prior_min_minutes
    overall = values[established].median() if established.any() else values.median()
    if groups is None:
        return pd.Series(overall, index=values.index)
    by_group = values[established].groupby(groups[established]).median()
    return groups.map(by_group).fillna(overall)


def shrink_small_samples(
    df: pd.DataFrame,
    metric_columns: list[str],
    minutes_col: str = "Min",
    group_col: str = "position_group",
    config: dict | None = None,
) -> pd.DataFrame:
    config = config or load_reliability_config()
    k = float(config["shrinkage_minutes"])
    if k <= 0 or minutes_col not in df.columns:
        return df

    result = df.copy()
    minutes = pd.to_numeric(result[minutes_col], errors="coerce").fillna(0).clip(lower=0)
    groups = result[group_col] if group_col in result.columns else None
    own_weight = minutes / (minutes + k)
    for column in metric_columns:
        if column in UNADJUSTED_METRICS or column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        typical = _position_typical(values, minutes, groups, float(config["prior_min_minutes"]))
        result[column] = (values.fillna(typical) * own_weight + typical * (1 - own_weight))
    return result


def sample_indicator(minutes: pd.Series, config: dict | None = None) -> pd.Series:
    """●●● / ●● / ● by minutes played."""
    high, medium = (config or load_reliability_config())["sample_bands"]
    minutes = pd.to_numeric(minutes, errors="coerce").fillna(0)
    return pd.Series("●", index=minutes.index).mask(minutes >= medium, "●●").mask(minutes >= high, "●●●")
