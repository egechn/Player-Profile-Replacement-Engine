from __future__ import annotations

import re
import unicodedata
from typing import Any


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(value != value):
            return True
    except TypeError:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"<na>", "nan", "none"}


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = strip_accents(str(value)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_person_name(value: Any) -> str:
    text = normalize_text(value)
    tokens = [
        token
        for token in text.split()
        if token not in {"jr", "junior", "sr", "ii", "iii", "iv"}
    ]
    return " ".join(tokens)
