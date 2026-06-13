from __future__ import annotations

import pandas as pd


def add_percentiles(
    df: pd.DataFrame,
    metric_columns: list[str],
    group_col: str = "position_group",
    suffix: str = "_pct",
) -> pd.DataFrame:
    result = df.copy()
    available_metrics = [column for column in metric_columns if column in result.columns]

    for column in available_metrics:
        values = pd.to_numeric(result[column], errors="coerce").fillna(0)
        if group_col in result.columns:
            result[f"{column}{suffix}"] = values.groupby(result[group_col]).rank(
                pct=True, method="average"
            ) * 100
        else:
            result[f"{column}{suffix}"] = values.rank(pct=True, method="average") * 100
        result[f"{column}{suffix}"] = result[f"{column}{suffix}"].fillna(50)

    return result

