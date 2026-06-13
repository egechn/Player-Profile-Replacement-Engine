from __future__ import annotations

import pandas as pd


def aggregate_player_valuations(valuations: pd.DataFrame) -> pd.DataFrame:
    if valuations.empty:
        return pd.DataFrame(
            columns=[
                "tm_player_id",
                "latest_market_value_eur",
                "market_value_date",
                "valuation_highest_market_value_eur",
            ]
        )

    result = valuations.copy()
    result["player_id"] = pd.to_numeric(result["player_id"], errors="coerce")
    result["market_value_in_eur"] = pd.to_numeric(
        result["market_value_in_eur"], errors="coerce"
    )
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["player_id"])

    latest = (
        result.sort_values(["player_id", "date"])
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id", "market_value_in_eur", "date"]]
        .rename(
            columns={
                "player_id": "tm_player_id",
                "market_value_in_eur": "latest_market_value_eur",
                "date": "market_value_date",
            }
        )
    )
    highest = (
        result.groupby("player_id", as_index=False)["market_value_in_eur"]
        .max()
        .rename(
            columns={
                "player_id": "tm_player_id",
                "market_value_in_eur": "valuation_highest_market_value_eur",
            }
        )
    )
    return latest.merge(highest, on="tm_player_id", how="outer")


def resolve_market_values(players: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    result = players.copy()
    aggregated = aggregate_player_valuations(valuations)
    result = result.merge(aggregated, on="tm_player_id", how="left")

    result["market_value_eur"] = result["market_value_eur"].combine_first(
        result["latest_market_value_eur"]
    )
    result["highest_market_value_eur"] = result["highest_market_value_eur"].combine_first(
        result["valuation_highest_market_value_eur"]
    )
    result["market_value_date"] = result["market_value_date"].astype("datetime64[ns]")
    return result.drop(
        columns=[
            "latest_market_value_eur",
            "valuation_highest_market_value_eur",
        ],
        errors="ignore",
    )

