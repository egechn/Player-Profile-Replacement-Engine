from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = resolve_project_path(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_table(path: str | Path) -> pd.DataFrame:
    path = resolve_project_path(path)
    if path.exists() and path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            fallback = path.with_suffix(".csv")
            if fallback.exists():
                return pd.read_csv(fallback, low_memory=False)
            raise
    if path.exists() and path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    parquet_fallback = path.with_suffix(".parquet")
    csv_fallback = path.with_suffix(".csv")
    if parquet_fallback.exists():
        try:
            return pd.read_parquet(parquet_fallback)
        except Exception:
            if csv_fallback.exists():
                return pd.read_csv(csv_fallback, low_memory=False)
            raise
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback, low_memory=False)
    raise FileNotFoundError(f"Table not found: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return path

    raise ValueError(f"Unsupported output format: {path.suffix}")

