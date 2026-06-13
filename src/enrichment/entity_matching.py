from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.cleaning.normalize_names import normalize_person_name
from src.cleaning.normalize_teams import normalize_team_name


try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


MATCH_METHODS = {
    "exact_name_dob": 100.0,
    "exact_display_name_dob": 98.0,
    "exact_name_team": 94.0,
    "fuzzy_name_dob": 0.0,
    "fuzzy_name_team": 0.0,
    "fuzzy_name_only": 0.0,
    "unmatched": 0.0,
}


@dataclass
class MatchCandidate:
    tm_index: int | None
    tm_player_id: Any
    confidence: float
    method: str
    name_score: float
    team_score: float
    dob_match: bool


def similarity_score(left: str, right: str) -> float:
    left = left or ""
    right = right or ""
    if not left or not right:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_sort_ratio(left, right))

    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio() * 100


def team_similarity_score(left: str, right: str) -> float:
    left = left or ""
    right = right or ""
    score = similarity_score(left, right)
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens and right_tokens:
        if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
            return max(score, 100.0)
        overlap = len(left_tokens & right_tokens) / max(min(len(left_tokens), len(right_tokens)), 1)
        if overlap >= 0.75:
            return max(score, 90.0)
    return score


def _date_key(value: Any) -> str:
    if value is None or value != value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _token_prefixes(value: str, length: int = 4) -> set[str]:
    tokens = [token for token in (value or "").split() if len(token) >= 2]
    prefixes = {token[:length] for token in tokens if len(token) >= length}
    if not prefixes and value:
        prefixes.add(value[:length])
    return prefixes


def _safe_float(value: Any, default: float = -1.0) -> float:
    try:
        if value is None or value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_matching_index(tm_records: list[dict[str, Any]]) -> dict[str, Any]:
    index: dict[str, Any] = {
        "name_dob": defaultdict(list),
        "name_team": defaultdict(list),
        "dob": defaultdict(list),
        "team": defaultdict(list),
        "team_prefix": defaultdict(list),
        "name_token_prefix": defaultdict(list),
    }

    for idx, row in enumerate(tm_records):
        name = row.get("tm_player_name_norm", "")
        team = row.get("tm_current_club_name_norm", "")
        dob = row.get("tm_dob_key", "")

        if name and dob:
            index["name_dob"][(name, dob)].append(idx)
        if name and team:
            index["name_team"][(name, team)].append(idx)
        if dob:
            index["dob"][dob].append(idx)
        if team:
            index["team"][team].append(idx)
            index["team_prefix"][team[:4]].append(idx)
        for prefix in _token_prefixes(name):
            index["name_token_prefix"][prefix].append(idx)

    return index


def _best_exact_candidate_from_indices(
    tm_records: list[dict[str, Any]],
    indices: list[int],
    stats_row: pd.Series,
    method: str,
    confidence: float,
) -> MatchCandidate | None:
    if not indices:
        return None

    stats_team = stats_row.get("team_name_norm", "")
    best_idx = None
    best_key = (-1.0, -1.0)
    for idx in indices:
        row = tm_records[idx]
        team_score = team_similarity_score(stats_team, row.get("tm_current_club_name_norm", ""))
        market_value = _safe_float(row.get("market_value_eur"))
        sort_key = (team_score, market_value)
        if sort_key > best_key:
            best_key = sort_key
            best_idx = idx

    if best_idx is None:
        return None

    chosen = tm_records[best_idx]
    return MatchCandidate(
        tm_index=int(best_idx),
        tm_player_id=chosen["tm_player_id"],
        confidence=confidence,
        method=method,
        name_score=100.0,
        team_score=float(best_key[0]),
        dob_match=_date_key(stats_row.get("dateOfBirth"))
        == _date_key(chosen.get("date_of_birth")),
    )


def _candidate_indices_from_index(
    stats_row: pd.Series,
    index: dict[str, Any],
) -> list[int]:
    candidate_indices: set[int] = set()
    dob_key = _date_key(stats_row.get("dateOfBirth"))
    team_norm = stats_row.get("team_name_norm", "")
    names = [
        stats_row.get("player_name_norm", ""),
        stats_row.get("display_name_norm", ""),
    ]

    if dob_key:
        candidate_indices.update(index["dob"].get(dob_key, []))
    if team_norm:
        candidate_indices.update(index["team"].get(team_norm, []))
        candidate_indices.update(index["team_prefix"].get(team_norm[:4], []))
    for name in names:
        for prefix in _token_prefixes(name):
            candidate_indices.update(index["name_token_prefix"].get(prefix, []))

    return list(candidate_indices)


def _best_fuzzy_candidate_from_indices(
    stats_row: pd.Series,
    tm_records: list[dict[str, Any]],
    index: dict[str, Any],
    minimum_fuzzy_score: int = 90,
    same_team_bonus: int = 5,
    same_dob_bonus: int = 8,
) -> MatchCandidate | None:
    indices = _candidate_indices_from_index(stats_row, index)
    if not indices:
        return None

    stats_names = [
        stats_row.get("player_name_norm", ""),
        stats_row.get("display_name_norm", ""),
    ]
    stats_team = stats_row.get("team_name_norm", "")
    stats_dob = _date_key(stats_row.get("dateOfBirth"))

    best_candidate = None
    best_key = (-1.0, False, -1.0, -1.0)
    for idx in indices:
        row = tm_records[idx]
        name_score = max(
            similarity_score(stats_name, row.get("tm_player_name_norm", ""))
            for stats_name in stats_names
        )
        if name_score < minimum_fuzzy_score:
            continue

        team_score = team_similarity_score(stats_team, row.get("tm_current_club_name_norm", ""))
        dob_match = bool(stats_dob and stats_dob == row.get("tm_dob_key", ""))
        confidence = float(name_score)
        if dob_match:
            confidence += same_dob_bonus
        if team_score >= 90:
            confidence += same_team_bonus
        confidence = min(confidence, 99.0)

        if dob_match:
            method = "fuzzy_name_dob"
        elif team_score >= 90:
            method = "fuzzy_name_team"
        else:
            method = "fuzzy_name_only"

        sort_key = (confidence, dob_match, team_score, name_score)
        if sort_key > best_key:
            best_key = sort_key
            best_candidate = MatchCandidate(
                tm_index=int(idx),
                tm_player_id=row["tm_player_id"],
                confidence=confidence,
                method=method,
                name_score=float(name_score),
                team_score=float(team_score),
                dob_match=dob_match,
            )

    return best_candidate


def _best_exact_candidate(
    candidates: pd.DataFrame,
    stats_row: pd.Series,
    method: str,
    confidence: float,
) -> MatchCandidate | None:
    if candidates.empty:
        return None

    candidates = candidates.copy()
    stats_team = stats_row.get("team_name_norm", "")
    candidates["team_score"] = candidates["tm_current_club_name_norm"].map(
        lambda value: similarity_score(stats_team, value)
    )
    chosen = candidates.sort_values(
        ["team_score", "market_value_eur"], ascending=[False, False], na_position="last"
    ).iloc[0]
    return MatchCandidate(
        tm_index=int(chosen.name),
        tm_player_id=chosen["tm_player_id"],
        confidence=confidence,
        method=method,
        name_score=100.0,
        team_score=float(chosen["team_score"]),
        dob_match=_date_key(stats_row.get("dateOfBirth"))
        == _date_key(chosen.get("date_of_birth")),
    )


def _candidate_pool(stats_row: pd.Series, transfermarkt_df: pd.DataFrame) -> pd.DataFrame:
    dob_key = _date_key(stats_row.get("dateOfBirth"))
    team_norm = stats_row.get("team_name_norm", "")
    name_norm = stats_row.get("player_name_norm", "")

    pools = []
    if dob_key:
        pools.append(transfermarkt_df[transfermarkt_df["tm_dob_key"] == dob_key])
    if team_norm:
        pools.append(
            transfermarkt_df[
                transfermarkt_df["tm_current_club_name_norm"].str[:2]
                == team_norm[:2]
            ]
        )
    if name_norm:
        pools.append(
            transfermarkt_df[
                transfermarkt_df["tm_player_name_norm"].str[:1]
                == name_norm[:1]
            ]
        )

    if not pools:
        return transfermarkt_df
    return pd.concat(pools).drop_duplicates("tm_player_id")


def _best_fuzzy_candidate(
    stats_row: pd.Series,
    transfermarkt_df: pd.DataFrame,
    minimum_fuzzy_score: int = 90,
    same_team_bonus: int = 5,
    same_dob_bonus: int = 8,
) -> MatchCandidate | None:
    pool = _candidate_pool(stats_row, transfermarkt_df)
    if pool.empty:
        return None

    stats_names = [
        stats_row.get("player_name_norm", ""),
        stats_row.get("display_name_norm", ""),
    ]
    stats_team = stats_row.get("team_name_norm", "")
    stats_dob = _date_key(stats_row.get("dateOfBirth"))

    scored_rows = []
    for idx, row in pool.iterrows():
        name_score = max(
            similarity_score(stats_name, row.get("tm_player_name_norm", ""))
            for stats_name in stats_names
        )
        if name_score < minimum_fuzzy_score:
            continue

        team_score = similarity_score(stats_team, row.get("tm_current_club_name_norm", ""))
        dob_match = bool(stats_dob and stats_dob == row.get("tm_dob_key", ""))
        confidence = name_score
        if dob_match:
            confidence += same_dob_bonus
        if team_score >= 90:
            confidence += same_team_bonus
        confidence = min(confidence, 99.0)

        if dob_match:
            method = "fuzzy_name_dob"
        elif team_score >= 90:
            method = "fuzzy_name_team"
        else:
            method = "fuzzy_name_only"

        scored_rows.append(
            MatchCandidate(
                tm_index=int(idx),
                tm_player_id=row["tm_player_id"],
                confidence=float(confidence),
                method=method,
                name_score=float(name_score),
                team_score=float(team_score),
                dob_match=dob_match,
            )
        )

    if not scored_rows:
        return None

    return sorted(
        scored_rows,
        key=lambda item: (
            item.confidence,
            item.dob_match,
            item.team_score,
            item.name_score,
        ),
        reverse=True,
    )[0]


def _needs_review(candidate: MatchCandidate, review_threshold: int = 90) -> bool:
    if candidate.method == "unmatched":
        return True
    if candidate.confidence < review_threshold:
        return True
    if candidate.method == "fuzzy_name_only":
        return True
    if not candidate.dob_match and candidate.method in {
        "exact_name_dob",
        "exact_display_name_dob",
        "fuzzy_name_dob",
    }:
        return True
    if candidate.team_score < 70 and candidate.method in {
        "exact_name_dob",
        "exact_display_name_dob",
        "fuzzy_name_only",
    }:
        return True
    return False


def match_players(
    statslook_df: pd.DataFrame,
    transfermarkt_df: pd.DataFrame,
    minimum_fuzzy_score: int = 90,
    same_team_bonus: int = 5,
    same_dob_bonus: int = 8,
    review_confidence_threshold: int = 90,
) -> pd.DataFrame:
    tm = transfermarkt_df.copy()
    if "tm_player_name_norm" not in tm.columns:
        tm["tm_player_name_norm"] = tm["tm_player_name"].map(normalize_person_name)
    if "tm_current_club_name_norm" not in tm.columns:
        tm["tm_current_club_name_norm"] = tm["current_club_name"].map(normalize_team_name)
    tm["tm_dob_key"] = tm["date_of_birth"].map(_date_key)
    tm = tm.reset_index(drop=True)
    tm_records = tm.to_dict("records")
    matching_index = _build_matching_index(tm_records)

    matches = []
    for _, stats_row in statslook_df.iterrows():
        player_name = stats_row.get("player_name_norm", "")
        display_name = stats_row.get("display_name_norm", "")
        team_name = stats_row.get("team_name_norm", "")
        dob_key = _date_key(stats_row.get("dateOfBirth"))

        candidate = None
        if player_name and dob_key:
            candidate = _best_exact_candidate_from_indices(
                tm_records,
                matching_index["name_dob"].get((player_name, dob_key), []),
                stats_row,
                "exact_name_dob",
                MATCH_METHODS["exact_name_dob"],
            )
        if candidate is None and display_name and dob_key:
            candidate = _best_exact_candidate_from_indices(
                tm_records,
                matching_index["name_dob"].get((display_name, dob_key), []),
                stats_row,
                "exact_display_name_dob",
                MATCH_METHODS["exact_display_name_dob"],
            )
        if candidate is None and player_name and team_name:
            candidate = _best_exact_candidate_from_indices(
                tm_records,
                matching_index["name_team"].get((player_name, team_name), []),
                stats_row,
                "exact_name_team",
                MATCH_METHODS["exact_name_team"],
            )
        if candidate is None:
            candidate = _best_fuzzy_candidate_from_indices(
                stats_row,
                tm_records,
                matching_index,
                minimum_fuzzy_score=minimum_fuzzy_score,
                same_team_bonus=same_team_bonus,
                same_dob_bonus=same_dob_bonus,
            )
        if candidate is None:
            candidate = MatchCandidate(
                tm_index=None,
                tm_player_id=pd.NA,
                confidence=0.0,
                method="unmatched",
                name_score=0.0,
                team_score=0.0,
                dob_match=False,
            )

        matches.append(
            {
                "statslook_row_id": stats_row.get("statslook_row_id"),
                "tm_player_id": candidate.tm_player_id,
                "match_confidence": round(candidate.confidence, 2),
                "match_method": candidate.method,
                "needs_review": _needs_review(candidate, review_confidence_threshold),
                "match_name_score": round(candidate.name_score, 2),
                "match_team_score": round(candidate.team_score, 2),
            }
        )

    return pd.DataFrame(matches)
