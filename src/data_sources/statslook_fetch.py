"""Fetch a StatsLook player table into data/raw/statslook/<season>/.

The scout page reads a public JSON API (no login), 100 players per page, with
stats keyed by numeric id. The id -> name catalog is embedded in the page HTML.
This pages through the API, names the stats, takes season totals and renames
the columns to the pipeline's names. Raw responses are saved next to the CSV.

    python -m src.data_sources.statslook_fetch            # season from config
    python -m src.data_sources.statslook_fetch --build    # fetch, then build
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import load_yaml_config
from src.data_sources.statslook_loader import REQUIRED_STATSLOOK_COLUMNS
from src.seasons import STATSLOOK_RAW_DIR

DEFAULT_PAGE_URL = "https://statslook.com/scout/players"
DEFAULT_API_URL = "https://statslook.com/api/players/stats"
DETAILED_POSITIONS = {
    148: "CB", 154: "RB", 155: "LB", 158: "RM", 157: "LM", 150: "AM",
    153: "CM", 149: "DM", 151: "ST", 152: "LW", 156: "RW", 163: "SS",
}
POSITIONS = {24: "Goalkeeper", 25: "Defender", 26: "Midfielder", 27: "Forward"}
# StatsLook's rate columns are left out: the pipeline computes rates from the counts.
SKIPPED_FORMATS = {"percentage"}
# The season "total" of a rating is a sum of match ratings, so the per-match average is kept.
AVERAGED_FORMATS = {"decimal1"}
FLIGHT_CHUNK = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', flags=re.S)
PLAYER_PAGE_URL = "https://statslook.com/players/p/{player_id}/overview"
PROFILE_TEAM = re.compile(r'"initialProfile":\{"team":\{"id":(\d+),"name":"((?:[^"\\]|\\.)*)"')


def load_fetch_config() -> dict[str, Any]:
    return load_yaml_config("config/statslook_fetch.yaml", {})


def api_season(season: str) -> str:
    """Our label "2026-27" -> StatsLook's "2026/2027"."""
    start = int(season[:4])
    return f"{start}/{start + 1}"


def _embedded_array(flight: str, key: str) -> list:
    start = flight.index(f'"{key}":[') + len(key) + 3
    return json.JSONDecoder().raw_decode(flight[start:])[0]


def page_metadata(session: requests.Session, page_url: str) -> dict[str, list]:
    """Season names, stat catalog, leagues and countries from the page's Next.js payload."""
    response = session.get(page_url, timeout=60)
    response.raise_for_status()
    flight = "".join(json.loads(f'"{chunk}"') for chunk in FLIGHT_CHUNK.findall(response.text))
    return {key: _embedded_array(flight, key) for key in ("seasonNames", "statCatalog", "leagues", "countries")}


def fetch_pages(session: requests.Session, api_url: str, params: dict[str, Any], pause_seconds: float) -> list[dict]:
    pages: list[dict] = []
    page, total = 1, None
    limit = int(params.get("limit", 100))
    while total is None or (page - 1) * limit < total:
        query = {**params, "page": page, "limit": limit}
        for attempt in range(5):
            response = session.get(api_url, params=query, timeout=60)
            if response.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        response.raise_for_status()
        body = response.json()
        total = int(body.get("total") or 0)
        players = body.get("players") or []
        pages.append({"url": response.url, "body": body})
        if not players:
            break
        # The API caps the page size; follow what it actually returns.
        if page == 1:
            limit = min(limit, len(players))
        page += 1
        time.sleep(pause_seconds)
    return pages


def _stat_columns(catalog: list[dict]) -> dict[str, dict]:
    return {
        str(stat["id"]): {"column": str(stat["code"]).replace("-", "_"), "format": stat.get("format")}
        for stat in catalog
    }


def player_table(pages: list[dict], metadata: dict[str, list]) -> pd.DataFrame:
    stats = _stat_columns(metadata["statCatalog"])
    countries = {country["id"]: country["name"] for country in metadata["countries"]}
    rows = []
    for page in pages:
        for player in page["body"].get("players") or []:
            row = {key: value for key, value in player.items() if key != "stats"}
            row["Detailed_Position"] = DETAILED_POSITIONS.get(player.get("detailedPositionId"))
            row["Position"] = POSITIONS.get(player.get("positionId"))
            row["nationality"] = countries.get(player.get("nationalityId"))
            for stat_id, values in (player.get("stats") or {}).items():
                stat = stats.get(stat_id, {"column": f"stat_{stat_id}", "format": None})
                if stat["format"] in SKIPPED_FORMATS:
                    continue
                field = "avg" if stat["format"] in AVERAGED_FORMATS else "total"
                row[stat["column"]] = values.get(field)
            rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit("StatsLook returned no players; check season and params in config/statslook_fetch.yaml.")
    table = table.drop_duplicates(subset="playerId")

    # A stat the player never registered is left out of the response: that is a zero.
    counting = [
        stat["column"]
        for stat in stats.values()
        if stat["format"] not in AVERAGED_FORMATS and stat["column"] in table.columns
    ]
    table[counting] = table[counting].fillna(0)
    return table


def _current_club(session: requests.Session, player_id: int) -> tuple[int, str] | None:
    try:
        response = session.get(PLAYER_PAGE_URL.format(player_id=player_id), timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        return None
    flight = "".join(json.loads(f'"{chunk}"') for chunk in FLIGHT_CHUNK.findall(response.text))
    match = PROFILE_TEAM.search(flight)
    return (int(match.group(1)), json.loads(f'"{match.group(2)}"')) if match else None


def fix_national_team_clubs(
    session: requests.Session, table: pd.DataFrame, metadata: dict[str, list], workers: int = 4
) -> pd.DataFrame:
    """The stats API lists some players under their national team, or no team.
    Their player page names the current club; use that instead."""
    countries = {country["name"] for country in metadata["countries"]}
    needs_club = table["teamName"].isna() | table["teamName"].isin(countries)
    player_ids = table.loc[needs_club, "playerId"].tolist()
    if not player_ids:
        return table
    print(f"Looking up the current club of {len(player_ids):,} players listed without one...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        clubs = dict(zip(player_ids, pool.map(lambda pid: _current_club(session, pid), player_ids)))
    table = table.copy()
    fixed = 0
    for index in table.index[needs_club]:
        club = clubs.get(table.at[index, "playerId"])
        if club and club[1] not in countries:
            table.at[index, "teamId"], table.at[index, "teamName"] = club
            fixed += 1
    print(f"Current club found for {fixed:,} of them.")
    return table


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def map_columns(table: pd.DataFrame, extra_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename StatsLook fields to the pipeline's names.

    Uses config/column_mapping.yaml (statslook_api: api name -> pipeline name),
    then column_map from config/statslook_fetch.yaml, then a case- and
    punctuation-insensitive match against the expected numeric columns
    (so `tackles_won` or `tacklesWon` becomes `Tackles_Won`).
    """
    mapping_config = load_yaml_config("config/column_mapping.yaml", {})
    rename = dict(mapping_config.get("statslook_api") or {})
    rename.update(extra_map or {})
    table = table.rename(columns={k: v for k, v in rename.items() if k in table.columns})

    expected = list(rename.values()) + list(mapping_config.get("numeric_columns") or [])
    expected += REQUIRED_STATSLOOK_COLUMNS
    by_norm = {_norm(name): name for name in expected}
    fuzzy = {
        column: by_norm[_norm(column)]
        for column in table.columns
        if column not in expected and _norm(column) in by_norm and by_norm[_norm(column)] not in table.columns
    }
    table = table.rename(columns=fuzzy)

    if {"Aerials_Won", "Aerial_Lost"} <= set(table.columns):
        table["Aerials"] = table["Aerials_Won"].fillna(0) + table["Aerial_Lost"].fillna(0)
    return table


def fetch_statslook(season: str | None = None, url: str | None = None) -> Path:
    config = load_fetch_config()
    season = season or config.get("season")
    api_url = url or config.get("api_url") or DEFAULT_API_URL
    if not season:
        raise SystemExit("Set season in config/statslook_fetch.yaml.")

    session = requests.Session()
    if config.get("user_agent"):
        session.headers["User-Agent"] = config["user_agent"]

    metadata = page_metadata(session, config.get("page_url") or DEFAULT_PAGE_URL)
    if api_season(season) not in metadata["seasonNames"]:
        raise SystemExit(
            f"StatsLook has no season {api_season(season)}. Offered: {', '.join(metadata['seasonNames'])}"
        )
    params = {**(config.get("params") or {}), "season": api_season(season)}
    pages = fetch_pages(session, api_url, params, float(config.get("pause_seconds", 0.3)))

    out_dir = STATSLOOK_RAW_DIR / season
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_responses.json").write_text(json.dumps(pages, ensure_ascii=False))
    (out_dir / "page_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1))

    table = fix_national_team_clubs(session, player_table(pages, metadata), metadata)
    table = map_columns(table, config.get("column_map") or {})
    out_path = out_dir / f"statslook_{season.replace('-', '_')}.csv"
    table.to_csv(out_path, index=False)

    reported = pages[0]["body"].get("total")
    missing = [column for column in REQUIRED_STATSLOOK_COLUMNS if column not in table.columns]
    print(
        f"Saved {len(table):,} rows (StatsLook reports {reported:,}) "
        f"and {len(table.columns)} columns to {out_path}"
    )
    if missing:
        print(
            "Missing columns the pipeline needs: "
            + ", ".join(missing)
            + ". Add them to column_map in config/statslook_fetch.yaml and fetch again. "
            + "Columns found: "
            + ", ".join(map(str, table.columns))
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch StatsLook player data for one season.")
    parser.add_argument("--season", default=None)
    parser.add_argument("--url", default=None, help="API URL override.")
    parser.add_argument("--build", action="store_true", help="Build the season's master table afterwards.")
    args = parser.parse_args()
    out_path = fetch_statslook(args.season, args.url)
    if args.build:
        from src.enrichment.transfermarkt_joiner import build_players_master
        from src.seasons import processed_dir_for

        season = args.season or load_fetch_config().get("season")
        master = build_players_master(
            statslook_path=out_path, processed_dir=processed_dir_for(season), season=season
        )
        print(f"{season}: built master table with {len(master):,} players.")


if __name__ == "__main__":
    main()
