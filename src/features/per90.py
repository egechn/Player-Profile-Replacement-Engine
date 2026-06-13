from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PER90_COLUMNS = [
    "Gol",
    "Asist",
    "xG",
    "xGoT",
    "Shots_Off_Target",
    "Shots_Total",
    "Shots_on_Target",
    "Shots_Blocked",
    "Key_Passes",
    "Chances_Created",
    "Big_Chances_Created",
    "Big_Chances_Missed",
    "Passes",
    "Accurate_Passes",
    "Passes_Final_Third",
    "Backward_Passes",
    "Long_Balls",
    "Long_Balls_Won",
    "Through_Balls",
    "Through_Balls_Won",
    "Total_Crosses",
    "Accurate_Crosses",
    "Dribble_Attempts",
    "Successful_Dribbles",
    "Tackles",
    "Tackles_Won",
    "Interception",
    "Ball_Recovery",
    "Clearences",
    "Blocked_Shots",
    "Total_Duels",
    "Duels_Won",
    "Duels_Lost",
    "Aerials",
    "Aerials_Won",
    "Aerial_Lost",
    "Fouls",
    "Fouls_Drawn",
    "Dispossed",
    "Possession_Lost",
    "Turn_Over",
    "Touches",
]


def add_per90(
    df: pd.DataFrame,
    metric_columns: list[str] | None = None,
    minutes_col: str = "Min",
) -> pd.DataFrame:
    result = df.copy()
    metrics = metric_columns or DEFAULT_PER90_COLUMNS
    minutes = pd.to_numeric(result[minutes_col], errors="coerce").replace(0, np.nan)

    for column in metrics:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce").fillna(0)
        result[f"{column}_per90"] = (values / minutes * 90).replace([np.inf, -np.inf], np.nan)
        result[f"{column}_per90"] = result[f"{column}_per90"].fillna(0)

    return result

