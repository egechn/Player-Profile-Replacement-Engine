from __future__ import annotations

from typing import Any

from src.cleaning.normalize_names import normalize_text


CLUB_SUFFIXES = {
    "fc",
    "cf",
    "sc",
    "afc",
    "ac",
    "sk",
    "fk",
    "club",
    "football",
    "futbol",
    "voetbalvereniging",
    "calcio",
    "sociedad",
    "sad",
    "s",
    "a",
    "p",
    "d",
    "de",
    "the",
    "clube",
}

DEFAULT_TEAM_ALIASES = {
    "inter": "internazionale",
    "inter milan": "internazionale",
    "internazionale milano": "internazionale",
    "internazionale milano p": "internazionale",
    "sporting cp": "sporting portugal",
    "sporting portugal": "sporting portugal",
    "sporting lisbon": "sporting portugal",
    "sk rapid": "rapid wien",
    "sk rapid wien": "rapid wien",
    "sportklub rapid": "rapid wien",
}


def normalize_team_name(value: Any, aliases: dict[str, str] | None = None) -> str:
    text = normalize_text(value)
    aliases = {**DEFAULT_TEAM_ALIASES, **(aliases or {})}
    if text in aliases:
        return aliases[text]

    tokens = [token for token in text.split() if token not in CLUB_SUFFIXES]
    simplified = " ".join(tokens)
    return aliases.get(simplified, simplified)
