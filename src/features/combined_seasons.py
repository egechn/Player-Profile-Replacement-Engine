"""One profile per player across several seasons.

Each season is scored on its own (percentiles within that season, league and
small-sample adjustments included). The combined profile is the minutes-weighted
average of those season scores, so 300 early-season minutes count for little
next to 3,000 minutes last season. Club, age, value and contract come from the
player's latest season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning.normalize_names import normalize_person_name
from src.features.profile_scores import profile_vector_columns, role_column

SEPARATOR = " + "
RATE_COLUMNS = [
    "pass_accuracy",
    "duel_success_rate",
    "aerial_win_rate",
    "dribble_success_rate",
    "tackle_success_rate",
    "shot_accuracy",
]


def combined_label(seasons: list[str]) -> str:
    return SEPARATOR.join(sorted(seasons))


def combined_parts(label: str) -> list[str]:
    return label.split(SEPARATOR) if SEPARATOR in label else [label]


def _player_keys(df: pd.DataFrame) -> pd.Series:
    birth = pd.to_datetime(df["dateOfBirth"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return df["playerName"].map(normalize_person_name) + "|" + birth


def combine_season_profiles(frames: list[pd.DataFrame], label: str) -> pd.DataFrame:
    """frames: scored season tables, oldest first."""
    stacked = pd.concat(
        [frame.assign(_season_order=order) for order, frame in enumerate(frames)], ignore_index=True
    )
    minutes = pd.to_numeric(stacked["Min"], errors="coerce").fillna(0)
    # Players with no minutes at all still get a (plain) average.
    stacked = stacked.assign(_key=_player_keys(stacked), _weight=minutes.where(minutes > 0, 1e-6))

    score_cols = profile_vector_columns(stacked)
    averaged = score_cols + [role_column(c) for c in score_cols]
    averaged += [c for c in stacked.columns if c.endswith("_per90")] + RATE_COLUMNS
    averaged = [c for c in dict.fromkeys(averaged) if c in stacked.columns]

    values = stacked[averaged].apply(pd.to_numeric, errors="coerce")
    weighted = values.mul(stacked["_weight"], axis=0).where(values.notna())
    weights = values.notna().mul(stacked["_weight"], axis=0)
    grouped_weighted = weighted.groupby(stacked["_key"]).sum(min_count=1)
    grouped_weights = weights.groupby(stacked["_key"]).sum().replace(0, np.nan)
    means = grouped_weighted / grouped_weights

    latest = stacked.sort_values("_season_order").groupby("_key").tail(1).set_index("_key").copy()
    latest[averaged] = means.reindex(latest.index)
    latest["Min"] = minutes.groupby(stacked["_key"]).sum().reindex(latest.index)
    latest["seasons_included"] = (
        stacked.groupby("_key")["season"].agg(lambda s: SEPARATOR.join(sorted(s.astype(str).unique())))
    ).reindex(latest.index)
    latest["season"] = label
    result = latest.drop(columns=["_season_order", "_weight"]).reset_index(drop=True)
    result["statslook_row_id"] = np.arange(len(result))
    return result
