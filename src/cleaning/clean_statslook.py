from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning.normalize_names import normalize_person_name
from src.cleaning.normalize_positions import normalize_position_group
from src.cleaning.normalize_teams import normalize_team_name
from src.data_sources.statslook_loader import load_statslook
from src.features.age import add_age
from src.features.per90 import DEFAULT_PER90_COLUMNS, add_per90
from src.io_utils import write_table


STATSLOOK_NUMERIC_COLUMNS = sorted(
    set(
        DEFAULT_PER90_COLUMNS
        + [
            "appearances",
            "height",
            "Min",
            "Yellow_Card",
            "Goals_Conceded",
            "Dribbled_Past",
        ]
    )
)


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    rate = numerator / denominator * 100
    return rate.replace([np.inf, -np.inf], np.nan).fillna(0)


def clean_statslook(
    df: pd.DataFrame | None = None,
    source_path: str | Path | None = None,
    reference_date: date | str | None = None,
) -> pd.DataFrame:
    result = load_statslook(source_path) if df is None else df.copy()

    for column in STATSLOOK_NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)

    result["dateOfBirth"] = pd.to_datetime(result["dateOfBirth"], errors="coerce")
    result["player_name_norm"] = result["playerName"].map(normalize_person_name)
    result["display_name_norm"] = result["displayName"].map(normalize_person_name)
    result["team_name_norm"] = result["teamName"].map(normalize_team_name)
    result["position_group"] = result.apply(
        lambda row: normalize_position_group(row.get("Detailed_Position"), row.get("Position")),
        axis=1,
    )
    result["statslook_row_id"] = result.index.astype(int)

    result = add_age(result, reference_date=reference_date)
    result = add_per90(result)

    result["pass_accuracy"] = _safe_rate(result["Accurate_Passes"], result["Passes"])
    result["duel_success_rate"] = _safe_rate(result["Duels_Won"], result["Total_Duels"])
    result["aerial_win_rate"] = _safe_rate(result["Aerials_Won"], result["Aerials"])
    result["dribble_success_rate"] = _safe_rate(
        result["Successful_Dribbles"], result["Dribble_Attempts"]
    )
    result["tackle_success_rate"] = _safe_rate(result["Tackles_Won"], result["Tackles"])
    result["shot_accuracy"] = _safe_rate(result["Shots_on_Target"], result["Shots_Total"])

    return result


def save_statslook_clean(
    output_path: str | Path = "data/processed/statslook_clean.parquet",
    **kwargs,
) -> Path:
    df = clean_statslook(**kwargs)
    return write_table(df, output_path)

