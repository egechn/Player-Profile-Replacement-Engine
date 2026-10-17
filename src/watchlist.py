"""A watchlist of players, kept in data/watchlist.csv between sessions."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.cleaning.normalize_names import normalize_person_name
from src.io_utils import project_root

WATCHLIST_FILE = project_root() / "data" / "watchlist.csv"
COLUMNS = [
    "player_key",
    "Player",
    "Current club",
    "Pos",
    "Age",
    "Value (€m)",
    "Contract",
    "Season",
    "Similar to",
    "Added",
    "Note",
]


def player_key(row: pd.Series) -> str:
    birth = pd.to_datetime(pd.Series([row.get("dateOfBirth")]), errors="coerce").dt.strftime("%Y-%m-%d").fillna("").iloc[0]
    return f"{normalize_person_name(row.get('playerName'))}|{birth}"


def load_watchlist() -> pd.DataFrame:
    if not WATCHLIST_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(WATCHLIST_FILE, dtype=str).reindex(columns=COLUMNS).fillna("")


def save_watchlist(watchlist: pd.DataFrame) -> None:
    WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    watchlist.reindex(columns=COLUMNS).to_csv(WATCHLIST_FILE, index=False)


def add_to_watchlist(row: pd.Series, display: pd.Series, similar_to: str) -> bool:
    """row: the player's data row; display: the same player as shown in the table.
    Returns False when the player is already on the list."""
    watchlist = load_watchlist()
    key = player_key(row)
    if key in set(watchlist["player_key"]):
        return False
    entry = {
        "player_key": key,
        "Player": display.get("Player", ""),
        "Current club": display.get("Current club", ""),
        "Pos": display.get("Pos", ""),
        "Age": display.get("Age", ""),
        "Value (€m)": display.get("Value (€m)", ""),
        "Contract": display.get("Contract", ""),
        "Season": row.get("season", ""),
        "Similar to": similar_to,
        "Added": datetime.now().strftime("%Y-%m-%d"),
        "Note": "",
    }
    entry = {column: "" if pd.isna(value) else str(value) for column, value in entry.items()}
    save_watchlist(pd.concat([watchlist, pd.DataFrame([entry])], ignore_index=True))
    return True
