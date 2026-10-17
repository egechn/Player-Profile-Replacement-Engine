from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_yaml_config
from src.engine.similarity import (
    ensure_profile_scores,
    find_similar_players,
    target_strength_profile_columns,
)
from src.engine.similarity import resolve_target_index
from src.features.profile_scores import profile_vector_columns


DEFAULT_REPLACEMENT_WEIGHTS = {
    "profile_similarity": 0.57,
    "position_role_match": 0.15,
    "affordability": 0.10,
    "age_upside": 0.10,
    "league_strength": 0.08,
}

PROFILE_EXPLANATION_METRICS = {
    "ground_defence_score": ["Tackles_Won_per90", "Interception_per90", "duel_success_rate"],
    "air_score": ["Aerials_Won_per90", "aerial_win_rate", "height"],
    "build_up_score": ["Passes_per90", "pass_accuracy", "Long_Balls_Won_per90"],
    "orchestration_score": ["Passes_Final_Third_per90", "Through_Balls_Won_per90", "Touches_per90"],
    "take_on_score": ["Successful_Dribbles_per90", "Dribble_Attempts_per90", "dribble_success_rate"],
    "chance_creation_score": ["Key_Passes_per90", "Chances_Created_per90", "Accurate_Crosses_per90"],
    "finishing_score": ["Gol_per90", "xG_per90", "Shots_on_Target_per90"],
}


def load_replacement_weights() -> dict[str, float]:
    return load_yaml_config("config/replacement_weights.yaml", DEFAULT_REPLACEMENT_WEIGHTS)


def affordability_score(candidate_value: Any, target_value: Any) -> float:
    candidate = pd.to_numeric(pd.Series([candidate_value]), errors="coerce").iloc[0]
    target = pd.to_numeric(pd.Series([target_value]), errors="coerce").iloc[0]
    if pd.isna(candidate) or pd.isna(target) or target <= 0:
        return 0.5
    if candidate <= target * 0.5:
        return 1.0
    if candidate <= target:
        return 0.75
    return float(max(0.0, 1 - (candidate - target) / (target * 2)))


def age_upside_score(candidate_age: Any, target_age: Any) -> float:
    candidate = pd.to_numeric(pd.Series([candidate_age]), errors="coerce").iloc[0]
    target = pd.to_numeric(pd.Series([target_age]), errors="coerce").iloc[0]
    if pd.isna(candidate) or pd.isna(target):
        return 0.5
    return float(np.clip((target - candidate + 5) / 10, 0, 1))


def league_strength_score(candidate_value: Any, target_value: Any) -> float:
    candidate = pd.to_numeric(pd.Series([candidate_value]), errors="coerce").iloc[0]
    target = pd.to_numeric(pd.Series([target_value]), errors="coerce").iloc[0]
    if pd.isna(candidate):
        return 0.75
    if pd.isna(target):
        target = 0.75
    return float(np.clip(candidate + 0.5 * (candidate - target), 0, 1))


def position_role_match_score(candidate: pd.Series, target: pd.Series) -> float:
    if candidate.get("position_group") == target.get("position_group"):
        return 1.0
    if candidate.get("Position") == target.get("Position"):
        return 0.8
    return 0.3


def _profile_explanation(candidate: pd.Series, target: pd.Series, vector_cols: list[str]) -> str:
    focus_cols = target_strength_profile_columns(target, vector_cols)
    differences = []
    for column in focus_cols:
        candidate_value = pd.to_numeric(pd.Series([candidate.get(column)]), errors="coerce").iloc[0]
        target_value = pd.to_numeric(pd.Series([target.get(column)]), errors="coerce").iloc[0]
        if pd.isna(candidate_value) or pd.isna(target_value):
            continue
        differences.append((column, abs(candidate_value - target_value), target_value))

    close_profiles = [
        item[0].replace("_score", "").replace("_", " ")
        for item in sorted(differences, key=lambda item: (item[1], -item[2]))[:3]
    ]
    metric_candidates: list[str] = []
    for profile_col, _, _ in sorted(differences, key=lambda item: (item[1], -item[2]))[:3]:
        metric_candidates.extend(PROFILE_EXPLANATION_METRICS.get(profile_col, []))

    close_metrics = []
    seen_metrics = set()
    for metric in metric_candidates:
        if metric not in candidate.index or metric not in target.index:
            continue
        candidate_value = pd.to_numeric(pd.Series([candidate.get(metric)]), errors="coerce").iloc[0]
        target_value = pd.to_numeric(pd.Series([target.get(metric)]), errors="coerce").iloc[0]
        if pd.isna(candidate_value) or pd.isna(target_value):
            continue
        scale = max(abs(target_value), 1)
        if abs(candidate_value - target_value) / scale <= 0.25:
            metric_label = metric.replace("_per90", "/90").replace("_", " ")
            if metric_label in seen_metrics:
                continue
            close_metrics.append(metric_label)
            seen_metrics.add(metric_label)
        if len(close_metrics) == 3:
            break

    profile_text = ", ".join(close_profiles) if close_profiles else "target strengths"
    metric_text = ", ".join(close_metrics) if close_metrics else "the underlying role metrics"
    focus_text = ", ".join(
        column.replace("_score", "").replace("_", " ") for column in focus_cols
    )
    return (
        f"Similar in target strengths: {profile_text}; closest metric signals include "
        f"{metric_text}. Search focused on: {focus_text}."
    )


def recommend_replacements(
    df: pd.DataFrame,
    player_name: str | None = None,
    statslook_row_id: int | None = None,
    tm_player_id: Any | None = None,
    top_n: int = 10,
    candidate_pool_size: int = 100,
    min_minutes: int | None = 600,
    max_age: float | None = None,
    max_market_value_eur: float | None = None,
    leagues: list[str] | None = None,
    positions: list[str] | None = None,
    position_scope: str = "adjacent",
) -> pd.DataFrame:
    scored = ensure_profile_scores(df)
    target_index = resolve_target_index(scored, player_name, statslook_row_id, tm_player_id)
    target = scored.loc[target_index]
    candidates = find_similar_players(
        scored,
        statslook_row_id=target.get("statslook_row_id"),
        top_n=candidate_pool_size,
        min_minutes=min_minutes,
        max_age=max_age,
        max_market_value_eur=max_market_value_eur,
        leagues=leagues,
        positions=positions,
        position_scope=position_scope,
        use_league_strength=False,
    ).copy()
    if candidates.empty:
        return candidates

    weights = load_replacement_weights()
    candidates["position_role_match"] = candidates.apply(
        lambda row: position_role_match_score(row, target),
        axis=1,
    )
    candidates["affordability"] = candidates["market_value_eur"].map(
        lambda value: affordability_score(value, target.get("market_value_eur"))
    )
    candidates["age_upside"] = candidates["age"].map(
        lambda value: age_upside_score(value, target.get("age"))
    )
    candidates["league_strength"] = candidates["league_strength_coefficient"].map(
        lambda value: league_strength_score(value, target.get("league_strength_coefficient"))
    )
    candidates["replacement_score"] = (
        candidates["profile_similarity"] * weights.get("profile_similarity", 0.57)
        + candidates["position_role_match"] * weights.get("position_role_match", 0.15)
        + candidates["affordability"] * weights.get("affordability", 0.10)
        + candidates["age_upside"] * weights.get("age_upside", 0.10)
        + candidates["league_strength"] * weights.get("league_strength", 0.08)
    ) * 100
    vector_cols = profile_vector_columns(scored)
    candidates["recommendation_reason"] = candidates.apply(
        lambda row: _profile_explanation(row, target, vector_cols),
        axis=1,
    )
    return candidates.sort_values("replacement_score", ascending=False).head(top_n)


def segment_shortlists(recommendations: pd.DataFrame, target: pd.Series) -> dict[str, pd.DataFrame]:
    if recommendations.empty:
        return {"younger": recommendations, "cheaper": recommendations, "high_upside": recommendations}

    younger = recommendations[
        pd.to_numeric(recommendations["age"], errors="coerce")
        < pd.to_numeric(pd.Series([target.get("age")]), errors="coerce").iloc[0]
    ].copy()
    cheaper = recommendations[
        pd.to_numeric(recommendations["market_value_eur"], errors="coerce")
        < pd.to_numeric(pd.Series([target.get("market_value_eur")]), errors="coerce").iloc[0]
    ].copy()
    high_upside = recommendations.sort_values(
        ["age_upside", "replacement_score"], ascending=False
    ).head(10)
    return {
        "younger": younger.head(10),
        "cheaper": cheaper.head(10),
        "high_upside": high_upside,
    }
