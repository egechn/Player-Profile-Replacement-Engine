from __future__ import annotations

from datetime import date

import pandas as pd


def add_age(
    df: pd.DataFrame,
    date_col: str = "dateOfBirth",
    output_col: str = "age",
    reference_date: date | str | None = None,
) -> pd.DataFrame:
    result = df.copy()
    reference = pd.Timestamp(reference_date or date.today())
    born = pd.to_datetime(result[date_col], errors="coerce")
    result[output_col] = ((reference - born).dt.days / 365.25).round(1)
    return result

