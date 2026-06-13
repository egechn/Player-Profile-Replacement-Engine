from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.league_adjustment import (
    apply_league_strength_columns,
    league_similarity_weight,
    load_league_strength_config,
)
from src.features.profile_scores import compute_profile_scores, profile_vector_columns


def _cosine_similarity(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1)
    target_norm = np.linalg.norm(target)
    denominator = matrix_norm * target_norm
    denominator = np.where(denominator == 0, np.nan, denominator)
    scores = matrix @ target / denominator
    return np.nan_to_num(scores, nan=0.0)


def _euclidean_similarity(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(matrix - target, axis=1)
    max_distance = np.nanmax(distances) if len(distances) else 1
    if max_distance == 0:
        return np.ones(len(distances))
    return 1 - distances / max_distance


def ensure_profile_scores(df: pd.DataFrame) -> pd.DataFrame:
    columns = profile_vector_columns(df)
    if len(columns) >= 3:
        return df
    return compute_profile_scores(df)


def resolve_target_index(
    df: pd.DataFrame,
    player_name: str | None = None,
    statslook_row_id: int | None = None,
    tm_player_id: Any | None = None,
) -> int:
    if statslook_row_id is not None and "statslook_row_id" in df.columns:
        matches = df.index[df["statslook_row_id"].eq(statslook_row_id)].tolist()
        if matches:
            return matches[0]

    if tm_player_id is not None and "tm_player_id" in df.columns:
        matches = df.index[df["tm_player_id"].eq(tm_player_id)].tolist()
        if matches:
            return matches[0]

    if player_name is not None:
        name = player_name.lower().strip()
        for column in ["playerName", "displayName", "tm_player_name"]:
            if column in df.columns:
                matches = df.index[df[column].astype(str).str.lower().eq(name)].tolist()
                if matches:
                    return matches[0]
        for column in ["playerName", "displayName", "tm_player_name"]:
            if column in df.columns:
                matches = df.index[
                    df[column].astype(str).str.lower().str.contains(name, regex=False)
                ].tolist()
                if matches:
                    return matches[0]

    raise ValueError("Target player could not be resolved.")


def filter_candidates(
    df: pd.DataFrame,
    min_minutes: int | None = None,
    max_age: float | None = None,
    max_market_value_eur: float | None = None,
    leagues: list[str] | None = None,
    positions: list[str] | None = None,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if min_minutes is not None and "Min" in df.columns:
        mask &= pd.to_numeric(df["Min"], errors="coerce").fillna(0) >= min_minutes
    if max_age is not None and "age" in df.columns:
        mask &= pd.to_numeric(df["age"], errors="coerce").fillna(999) <= max_age
    if max_market_value_eur is not None and "market_value_eur" in df.columns:
        values = pd.to_numeric(df["market_value_eur"], errors="coerce")
        mask &= values.isna() | (values <= max_market_value_eur)
    if leagues:
        league_cols = [column for column in ["league_display_name", "league_name"] if column in df.columns]
        if league_cols:
            league_mask = pd.Series(False, index=df.index)
            for column in league_cols:
                league_mask |= df[column].isin(leagues)
            mask &= league_mask
    if positions:
        available_position_cols = [col for col in ["position_group", "Position"] if col in df.columns]
        if available_position_cols:
            position_mask = pd.Series(False, index=df.index)
            for column in available_position_cols:
                position_mask |= df[column].isin(positions)
            mask &= position_mask
    return mask


def exclude_same_player_identity(
    df: pd.DataFrame,
    mask: pd.Series,
    target_index: int,
) -> pd.Series:
    result = mask.copy()
    target = df.loc[target_index]

    if "tm_player_id" in df.columns and pd.notna(target.get("tm_player_id")):
        result &= ~df["tm_player_id"].eq(target.get("tm_player_id"))

    if "player_name_norm" in df.columns and pd.notna(target.get("player_name_norm")):
        same_name = df["player_name_norm"].eq(target.get("player_name_norm"))
        if "dateOfBirth" in df.columns and pd.notna(target.get("dateOfBirth")):
            same_name &= pd.to_datetime(df["dateOfBirth"], errors="coerce").eq(
                pd.to_datetime(target.get("dateOfBirth"), errors="coerce")
            )
        result &= ~same_name

    result.loc[target_index] = False
    return result


def find_similar_players(
    df: pd.DataFrame,
    player_name: str | None = None,
    statslook_row_id: int | None = None,
    tm_player_id: Any | None = None,
    top_n: int = 10,
    method: str = "cosine",
    center_profiles: bool = True,
    same_position_boost: float = 0.03,
    min_minutes: int | None = None,
    max_age: float | None = None,
    max_market_value_eur: float | None = None,
    leagues: list[str] | None = None,
    positions: list[str] | None = None,
    use_league_strength: bool = False,
    league_strength_weight: float | None = None,
) -> pd.DataFrame:
    scored = apply_league_strength_columns(ensure_profile_scores(df))
    target_index = resolve_target_index(scored, player_name, statslook_row_id, tm_player_id)
    vector_cols = profile_vector_columns(scored)
    if not vector_cols:
        raise ValueError("No profile score columns are available for similarity.")

    candidate_mask = filter_candidates(
        scored,
        min_minutes=min_minutes,
        max_age=max_age,
        max_market_value_eur=max_market_value_eur,
        leagues=leagues,
        positions=positions,
    )
    candidate_mask = exclude_same_player_identity(scored, candidate_mask, target_index)

    candidates = scored[candidate_mask].copy()
    target = scored.loc[target_index]
    if candidates.empty:
        return candidates

    matrix = candidates[vector_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    target_vector = pd.to_numeric(target[vector_cols], errors="coerce").fillna(0).to_numpy()
    if method == "euclidean":
        similarities = _euclidean_similarity(matrix, target_vector)
    else:
        if center_profiles:
            matrix = matrix - 50
            target_vector = target_vector - 50
        similarities = (_cosine_similarity(matrix, target_vector) + 1) / 2

    candidates["profile_similarity"] = similarities
    if "position_group" in candidates.columns and "position_group" in scored.columns:
        candidates["position_match"] = candidates["position_group"].eq(target["position_group"])
    else:
        candidates["position_match"] = False
    boost = candidates["position_match"].astype(float) * same_position_boost
    candidates["similarity_score"] = (
        candidates["profile_similarity"] + (1 - candidates["profile_similarity"]) * boost
    ).clip(0, 1)
    default_strength = float(load_league_strength_config().get("default_strength", 0.75))
    if "league_strength_coefficient" in candidates.columns:
        candidates["league_strength_coefficient"] = pd.to_numeric(
            candidates["league_strength_coefficient"],
            errors="coerce",
        ).fillna(default_strength)
    else:
        candidates["league_strength_coefficient"] = pd.Series(
            default_strength,
            index=candidates.index,
        )
    target_strength = pd.to_numeric(
        pd.Series([target.get("league_strength_coefficient", default_strength)]),
        errors="coerce",
    ).fillna(default_strength).iloc[0]
    candidates["target_league_strength_coefficient"] = target_strength
    candidates["league_strength_gap"] = (
        candidates["league_strength_coefficient"] - target_strength
    ).round(4)
    weight = league_similarity_weight() if league_strength_weight is None else league_strength_weight
    candidates["league_adjustment_factor"] = (
        1 + weight * candidates["league_strength_gap"]
    ).clip(0.96, 1.04)
    candidates["league_adjusted_similarity_score"] = (
        candidates["similarity_score"] * candidates["league_adjustment_factor"]
    )

    sort_column = "league_adjusted_similarity_score" if use_league_strength else "similarity_score"
    return candidates.sort_values(sort_column, ascending=False).head(top_n)
