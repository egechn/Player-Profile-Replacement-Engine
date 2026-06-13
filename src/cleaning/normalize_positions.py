from __future__ import annotations

import re
from typing import Any

from src.cleaning.normalize_names import normalize_text


POSITION_GROUP_RULES = {
    "goalkeeper": ["goalkeeper", "keeper", "gk"],
    "centre_back": ["centre back", "center back", "centre-back", "center-back", "cb"],
    "full_back": [
        "left back",
        "right back",
        "full back",
        "wing back",
        "left-back",
        "right-back",
        "lb",
        "rb",
        "lwb",
        "rwb",
    ],
    "defensive_midfield": ["defensive midfield", "defensive midfielder", "dm", "cdm"],
    "central_midfield": [
        "central midfield",
        "central midfielder",
        "centre midfield",
        "centre midfielder",
        "cm",
    ],
    "attacking_midfield": ["attacking midfield", "attacking midfielder", "am", "cam"],
    "winger": [
        "winger",
        "left winger",
        "right winger",
        "wide midfielder",
        "left midfielder",
        "right midfielder",
        "lw",
        "rw",
        "lm",
        "rm",
    ],
    "forward": [
        "centre forward",
        "center forward",
        "second striker",
        "striker",
        "forward",
        "cf",
        "st",
        "ss",
    ],
}

POSITION_GROUP_LABELS = {
    "goalkeeper": "Goalkeeper",
    "centre_back": "Centre back",
    "full_back": "Full back",
    "defensive_midfield": "Defensive midfield",
    "central_midfield": "Central midfield",
    "attacking_midfield": "Attacking midfield",
    "winger": "Winger",
    "forward": "Forward",
    "defender": "Defender",
    "midfielder": "Midfielder",
    "unknown": "Unknown",
}


def _is_numeric_id(value: Any) -> bool:
    return bool(re.fullmatch(r"\d+(\.0+)?", str(value).strip()))


def normalize_position_group(detailed_position: Any, position: Any = None) -> str:
    text = normalize_text(detailed_position)
    fallback = normalize_text(position)
    haystack = f"{text} {fallback}".strip()
    if not haystack:
        return "unknown"

    tokens = set(haystack.split())
    for group, patterns in POSITION_GROUP_RULES.items():
        for pattern in patterns:
            normalized_pattern = normalize_text(pattern)
            if " " in normalized_pattern and normalized_pattern in haystack:
                return group
            if normalized_pattern in tokens:
                return group

    if fallback and not _is_numeric_id(position):
        return fallback.replace(" ", "_")
    return "unknown"


def position_group_label(group: Any) -> str:
    normalized_group = normalize_text(group).replace(" ", "_")
    return POSITION_GROUP_LABELS.get(
        normalized_group,
        normalized_group.replace("_", " ").title() if normalized_group else "Unknown",
    )
