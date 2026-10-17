from __future__ import annotations

import argparse

from src.enrichment.transfermarkt_joiner import build_players_master
from src.seasons import processed_dir_for, raw_seasons, statslook_source_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the player master table.")
    parser.add_argument(
        "--season",
        action="append",
        default=None,
        help="Season to build, e.g. 2026-27. Repeat for several. Default: every season in data/raw/statslook/.",
    )
    parser.add_argument("--statslook-path", default=None, help="Single-file mode; needs exactly one --season.")
    parser.add_argument("--transfermarkt-dir", default=None)
    parser.add_argument("--processed-dir", default=None, help="Default: data/processed/<season>.")
    parser.add_argument(
        "--strict-transfermarkt",
        action="store_true",
        help="Fail if Transfermarkt CSVs are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = args.season or raw_seasons()
    if not seasons:
        raise SystemExit("No StatsLook data found under data/raw/statslook/.")
    if args.statslook_path and len(seasons) != 1:
        raise SystemExit("--statslook-path needs exactly one --season.")
    for season in seasons:
        master = build_players_master(
            statslook_path=args.statslook_path or statslook_source_for(season),
            transfermarkt_dir=args.transfermarkt_dir,
            processed_dir=args.processed_dir or processed_dir_for(season),
            allow_missing_transfermarkt=not args.strict_transfermarkt,
            season=season,
        )
        print(f"{season}: built master table with {len(master):,} players.")


if __name__ == "__main__":
    main()

