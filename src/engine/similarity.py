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


DEFAULT_STRENGTH_THRESHOLD = 60.0
DEFAULT_MAX_FOCUS_PROFILES = 5
DEFAULT_MIN_FOCUS_PROFILES = 3
LEAGUE_ADJUSTMENT_MIN = 0.90
LEAGUE_ADJUSTMENT_MAX = 1.08

# Areas always kept in the search for these positions, even when they are not
# among the target's top strengths: replacing a defender means replacing their
# defending.
POSITION_REQUIRED_PROFILES = {
    "centre_back": ["ground_defence_score", "air_score"],
    "full_back": ["ground_defence_score"],
    "defensive_midfield": ["ground_defence_score"],
    "central_midfield": ["ground_defence_score"],
}
REQUIRED_PROFILE_MIN_WEIGHT = 0.6

# Which position groups a target is compared against when position_scope is
# "adjacent". Scores are percentiles against all players, so without this an
# attacking full back and a passing midfielder can look alike.
ADJACENT_POSITION_GROUPS = {
    "centre_back": ["centre_back", "defensive_midfield", "defender"],
    "full_back": ["full_back", "defender"],
    "defensive_midfield": ["defensive_midfield", "central_midfield", "centre_back", "midfielder"],
    "central_midfield": ["central_midfield", "defensive_midfield", "attacking_midfield", "midfielder"],
    "attacking_midfield": ["attacking_midfield", "central_midfield", "winger", "midfielder"],
    "winger": ["winger", "attacking_midfield", "forward"],
    "forward": ["forward", "winger"],
    "goalkeeper": ["goalkeeper"],
    # Players StatsLook only gives a broad position for.
    "defender": ["defender", "centre_back", "full_back"],
    "midfielder": ["midfielder", "defensive_midfield", "central_midfield", "attacking_midfield"],
}
POSITION_SCOPES = ("same", "adjacent", "all")


def position_scope_groups(target_group: Any, scope: str = "adjacent") -> list[str] | None:
    if scope == "all" or pd.isna(target_group):
        return None
    if scope == "same":
        return [target_group]
    return ADJACENT_POSITION_GROUPS.get(target_group, [target_group])


def _profile_label(column: str) -> str:
    return column.replace("_score", "").replace("_", " ")


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


def _weighted_level_similarity(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weights_sum = weights.sum()
    if weights_sum == 0:
        weights = np.ones_like(weights)
        weights_sum = weights.sum()
    per_dimension = 1 - np.abs(matrix - target) / 100
    per_dimension = np.clip(per_dimension, 0, 1)
    return per_dimension @ weights / weights_sum


def target_strength_profile_columns(
    target: pd.Series,
    vector_cols: list[str],
    strength_threshold: float = DEFAULT_STRENGTH_THRESHOLD,
    max_profiles: int = DEFAULT_MAX_FOCUS_PROFILES,
    min_profiles: int = DEFAULT_MIN_FOCUS_PROFILES,
) -> list[str]:
    scores = pd.to_numeric(target.reindex(vector_cols), errors="coerce").dropna()
    if scores.empty:
        return vector_cols

    ranked = scores.sort_values(ascending=False)
    selected = ranked[ranked >= strength_threshold].index.tolist()[:max_profiles]
    for column in ranked.index.tolist():
        if len(selected) >= min_profiles:
            break
        if column not in selected:
            selected.append(column)
    for column in required_profile_columns(target, vector_cols):
        if column not in selected:
            selected.append(column)
    return selected or vector_cols


def required_profile_columns(target: pd.Series, vector_cols: list[str]) -> list[str]:
    required = POSITION_REQUIRED_PROFILES.get(target.get("position_group"), [])
    return [column for column in required if column in vector_cols]


def profile_focus_text(profile_cols: list[str]) -> str:
    return ", ".join(_profile_label(column) for column in profile_cols)


def _target_strength_weights(target: pd.Series, focus_cols: list[str]) -> np.ndarray:
    values = pd.to_numeric(target.reindex(focus_cols), errors="coerce").fillna(50).to_numpy()
    weights = np.clip((values - 50) / 50, 0.2, 1.0)
    required = required_profile_columns(target, focus_cols)
    for position, column in enumerate(focus_cols):
        if column in required:
            weights[position] = max(weights[position], REQUIRED_PROFILE_MIN_WEIGHT)
    return weights.astype(float)


def explain_similarity_from_strengths(
    candidate: pd.Series,
    target: pd.Series,
    focus_cols: list[str],
    max_profiles: int = 3,
) -> str:
    differences = []
    for column in focus_cols:
        candidate_value = pd.to_numeric(pd.Series([candidate.get(column)]), errors="coerce").iloc[0]
        target_value = pd.to_numeric(pd.Series([target.get(column)]), errors="coerce").iloc[0]
        if pd.isna(candidate_value) or pd.isna(target_value):
            continue
        differences.append((column, abs(candidate_value - target_value), target_value))

    closest = [
        _profile_label(column)
        for column, _, _ in sorted(differences, key=lambda item: (item[1], -item[2]))[
            :max_profiles
        ]
    ]
    basis = profile_focus_text(focus_cols)
    if not closest:
        return f"Search focused on the target player's strongest profiles: {basis}."
    return f"Matches target strengths in {', '.join(closest)}. Search focused on: {basis}."


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
    position_scope: str = "adjacent",
    use_league_strength: bool = False,
    league_strength_weight: float | None = None,
    focus_on_target_strengths: bool = True,
    strength_threshold: float = DEFAULT_STRENGTH_THRESHOLD,
    max_focus_profiles: int = DEFAULT_MAX_FOCUS_PROFILES,
    min_focus_profiles: int = DEFAULT_MIN_FOCUS_PROFILES,
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
    target = scored.loc[target_index]
    scope_groups = position_scope_groups(target.get("position_group"), position_scope)
    if scope_groups is not None and not positions and "position_group" in scored.columns:
        candidate_mask &= scored["position_group"].isin(scope_groups)

    candidates = scored[candidate_mask].copy()
    if candidates.empty:
        return candidates

    focus_cols = (
        target_strength_profile_columns(
            target,
            vector_cols,
            strength_threshold=strength_threshold,
            max_profiles=max_focus_profiles,
            min_profiles=min_focus_profiles,
        )
        if focus_on_target_strengths
        else vector_cols
    )
    matrix = candidates[focus_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    target_vector = pd.to_numeric(target[focus_cols], errors="coerce").fillna(0).to_numpy()
    if method == "euclidean":
        if focus_on_target_strengths:
            weights = np.sqrt(_target_strength_weights(target, focus_cols))
            similarities = _euclidean_similarity(matrix * weights, target_vector * weights)
        else:
            similarities = _euclidean_similarity(matrix, target_vector)
    else:
        if center_profiles:
            shape_matrix = matrix - 50
            shape_target = target_vector - 50
        else:
            shape_matrix = matrix
            shape_target = target_vector
        if focus_on_target_strengths:
            weights = np.sqrt(_target_strength_weights(target, focus_cols))
            shape_similarity = (_cosine_similarity(shape_matrix * weights, shape_target * weights) + 1) / 2
            level_similarity = _weighted_level_similarity(
                matrix,
                target_vector,
                _target_strength_weights(target, focus_cols),
            )
            similarities = 0.45 * shape_similarity + 0.55 * level_similarity
        else:
            similarities = (_cosine_similarity(shape_matrix, shape_target) + 1) / 2

    candidates["profile_similarity"] = similarities
    candidates["similarity_basis"] = profile_focus_text(focus_cols)
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
    ).clip(LEAGUE_ADJUSTMENT_MIN, LEAGUE_ADJUSTMENT_MAX)
    candidates["league_adjusted_similarity_score"] = (
        candidates["similarity_score"] * candidates["league_adjustment_factor"]
    )

    sort_column = "league_adjusted_similarity_score" if use_league_strength else "similarity_score"
    top = candidates.sort_values(sort_column, ascending=False).head(top_n).copy()
    # Explanations are slow per row: only for the players returned.
    top["similarity_reason"] = top.apply(
        lambda row: explain_similarity_from_strengths(row, target, focus_cols),
        axis=1,
    )
    return top
