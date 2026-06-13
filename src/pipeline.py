from __future__ import annotations

import argparse

from src.enrichment.transfermarkt_joiner import build_players_master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the player master table.")
    parser.add_argument("--statslook-path", default=None)
    parser.add_argument("--transfermarkt-dir", default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--strict-transfermarkt",
        action="store_true",
        help="Fail if Transfermarkt CSVs are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = build_players_master(
        statslook_path=args.statslook_path,
        transfermarkt_dir=args.transfermarkt_dir,
        processed_dir=args.processed_dir,
        allow_missing_transfermarkt=not args.strict_transfermarkt,
    )
    print(f"Built master table with {len(master):,} players.")


if __name__ == "__main__":
    main()

