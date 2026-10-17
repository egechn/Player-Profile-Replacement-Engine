# Player Profile Replacement Engine

This project builds player-level football profiles from Statslook performance data and enriches those players with Transfermarkt/Kaggle metadata such as nationality, market value, club, league, and contract information.

Statslook is the authoritative source for performance metrics. Transfermarkt data is used only for enrichment and must not replace Statslook performance metrics.

## Quick Start

This repository includes the minimal raw and processed data needed to inspect the prototype. To run the project on another computer, clone the repository, install the Python dependencies, and start the Streamlit dashboard. Rebuilding the processed master table is optional unless you replace the raw data.

```bash
git clone https://github.com/egechn/Player-Profile-Replacement-Engine.git
cd Player-Profile-Replacement-Engine
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The bundled minimal data files are:

```text
data/raw/statslook/Statslook_Scouting_Master_Final.csv
data/raw/transfermarkt/players.csv
data/raw/transfermarkt/clubs.csv
data/raw/transfermarkt/competitions.csv
data/raw/transfermarkt/player_valuations.csv
data/processed/players_master.parquet
```

Large optional Transfermarkt files such as `appearances.csv`, `game_events.csv`, and `game_lineups.csv` are not included because they are not required by the current pipeline and exceed normal GitHub-friendly repository size.

Run the dashboard directly:

```bash
streamlit run app.py
```

If you replace the raw data or want to rebuild the processed files, run:

```bash
python -m src.pipeline
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

If `data/processed/players_master.parquet` is missing, build it with `python -m src.pipeline` before starting the dashboard.

## Seasons

Each season is scored separately: percentiles and profiles are computed within the season.

In the sidebar, **Season** is the season of the selected player's profile and **Find similar players from** is the season candidates come from, so a player's 2026-27 form can be compared with 2025-26 players. The **2025-26 + 2026-27** option combines the last two seasons into one profile per player, weighting each season's scores by minutes played (steadier early in a season). Club, age, value and contract then come from the player's latest season.

1. Export the season from StatsLook (same columns as the current file) and save it as `data/raw/statslook/<season>/<any name>.csv`, for example `data/raw/statslook/2026-27/statslook_2026_27.csv`.
2. Build it: `python -m src.pipeline --season 2026-27` (without `--season`, every season found is built).
3. The table is written to `data/processed/<season>/players_master.parquet` and the season appears in the app's **Season** picker.

The original single file `data/raw/statslook/Statslook_Scouting_Master_Final.csv` and `data/processed/players_master.parquet` count as season `2025-26` (`legacy_season` in `config/seasons.yaml`).

For promoted and relegated clubs, add `config/season_team_leagues/<season>.yaml` with `"Team Name": League Name` lines; these override the team-to-league mapping for that season only.

Transfermarkt market values and contracts are the same snapshot for every season (the dataset's last snapshot is 6 July 2026; it is no longer refreshed).

The **Current club** shown in the tables is the player's club in the latest season's StatsLook data, whatever season the stats are from (`src/enrichment/current_club.py`). The legacy 2025-26 file has no StatsLook ids, so its players are matched on name + date of birth; players not in the latest season keep their own season's club. **League (stats)** is the league of the season the stats come from. A contract year is shown as `?` when the contract has already ended or Transfermarkt has the player at another club than the current one (a transfer after the snapshot).

### Fetching a season from StatsLook

The StatsLook page URL and season are in `config/statslook_fetch.yaml`. Either press **Fetch … from StatsLook** in the app sidebar, or run:

```bash
python -m src.data_sources.statslook_fetch --build
```

The scout page reads a public JSON API (`/api/players/stats`, no login, 100 players per page), so the fetcher calls it directly with `requests`. Stats arrive keyed by numeric id; their names come from the stat catalog embedded in the scout page. Season totals are kept, except for ratings (the per-match average). StatsLook's own percentage columns are left out: the pipeline computes rates from the counts. Players the API lists under a national team (or no team) get their club from their StatsLook player page. The rows are saved to `data/raw/statslook/<season>/statslook_<season>.csv` (plus `raw_responses.json` and `page_metadata.json` for checking), and `--build` builds the season. Field names are mapped to the pipeline's (`Min`, `Gol`, …) through the `statslook_api` section of `config/column_mapping.yaml`, then `column_map` in `config/statslook_fetch.yaml`, then a case-insensitive match. A full fetch takes a few minutes.

Stats are totals over every competition the player played in that season (league, cups, European and national-team matches).

### Small samples

Before ranking, each player's per-90 numbers and rates are pulled toward the typical value for their position, as if they had played 450 more minutes at that level: a player on 450 minutes sits halfway between their own numbers and the position's, one on 2,700 minutes keeps 86% of their own. This keeps 30-minute cameos out of the top of every profile area. The strength (`shrinkage_minutes`) and the tables' **Sample** indicator (●●● 900+ minutes, ●● 450+, ● below) are set in `config/reliability.yaml`.

### Minimum minutes

The sidebar **Minimum minutes** slider runs from 0 to the season's highest minutes. Its starting value comes from `min_minutes` in `config/seasons.yaml`; seasons not listed there start at 20% of a regular starter's minutes (90th percentile), so an early-season export gets a low default automatically.

## Data Layout

```text
data/
  raw/
    statslook/
      Statslook_Scouting_Master_Final.csv   # season 2025-26 (legacy location)
      2026-27/
        statslook_2026_27.csv               # one CSV per season folder
    transfermarkt/
      players.csv
      clubs.csv
      competitions.csv
      player_valuations.csv
      appearances.csv      # optional
      transfers.csv        # optional
  processed/
    statslook_clean.parquet
    transfermarkt_enrichment.parquet
    players_master.parquet
    unmatched_players.csv
```

The Statslook CSV has been copied into:

```text
data/raw/statslook/Statslook_Scouting_Master_Final.csv
```

Download the Kaggle Transfermarkt dataset manually from:

https://www.kaggle.com/datasets/davidcariboo/player-scores

Then place the required CSV files in:

```text
data/raw/transfermarkt/
```

## Matching Logic

Statslook players are matched to Transfermarkt players in this order:

1. `normalize(playerName) + dateOfBirth`
2. `normalize(displayName) + dateOfBirth`
3. `normalize(playerName) + normalize(teamName/current_club_name)`
4. Fuzzy name matching with RapidFuzz, with bonuses for same DOB and same/similar team
5. Unmatched players are written to `data/processed/unmatched_players.csv`

Match output columns:

```text
tm_player_id
tm_player_name
nationality_name
market_value_eur
highest_market_value_eur
market_value_date
current_club_id
current_club_name
league_id
league_name
league_country
league_display_name
league_strength_coefficient
opta_league_rank
opta_league_average_rating
domestic_competition_id
contract_until
transfermarkt_url
match_confidence
match_method
needs_review
```

Rows are marked `needs_review = True` when confidence is below 90, the match is name-only fuzzy, DOB conflicts, or the team looks suspicious.

## League Mapping

League assignment is not taken from Transfermarkt.

The authoritative league source is:

```text
config/team_league_mapping.yaml
```

This file maps Statslook `teamName` values to English league names. Teams missing from it are looked up in `config/team_league_mapping_inferred.yaml` at load time (`league_source = inferred_team_mapping`): clubs added by hand for 2025-26 (`source: manual`), plus teams whose own Transfermarkt club plays in a league that has a strength coefficient (`source: transfermarkt`). If a team is in neither file, the main league columns stay empty and `league_source` is set to `unmapped`.

Transfermarkt league values are preserved only as raw debug columns such as `tm_league_name` and `tm_league_country`; they are not used as the main league shown in the dashboard.

## League Strength

League strength coefficients live in:

```text
config/league_strength.yaml
```

The coefficients come from the Opta Power Rankings top 100 leagues (table as of 2026-10-13, Premier League rating 92.7). Leagues outside the top 100 and teams with no league get `default_strength` (0.72, just below the 100th league). The current formula is:

```text
league_strength_coefficient = Opta seasonAverageRating / Premier League seasonAverageRating
```

**League strength is applied to the metrics, not to the similarity score.** Before percentiles are computed, per-90 outputs (goals, xG, shots, assists, key passes, chances created, successful dribbles, final-third and accurate passes, crosses, aerials won, and possession-adjusted defensive actions) are multiplied by:

```text
league_factor = league_strength_coefficient ** metric_translation_exponent
```

With the default exponent of 2, Eredivisie output (0.85) counts at about 0.72 of the same output in the Premier League. Attempts, touches, rates and height are not translated. Raw per-90 values are kept for display; only the percentiles use translated values. The exponent is set in `config/league_strength.yaml`.

The old similarity multiplier (`league_adjusted_similarity_score = similarity_score × (1 + league_similarity_weight × league gap)`) is still computed but `league_similarity_weight` is now `0.0`, because league strength is already inside the metrics and the multiplier was large enough to override the profile match.

Teams with no league get `default_strength` (0.72), so their output is discounted heavily; keep `team_league_mapping.yaml` complete.

**Position scope.** "Most similar" and the replacement pool compare against the target's position group plus adjacent roles by default (`ADJACENT_POSITION_GROUPS` in `src/engine/similarity.py`). The sidebar "Compare against" option switches between same position, adjacent roles and all positions.

## Similarity Logic

Similarity is focused on the selected player's strongest profile scores, not every score equally.

For example, when searching around a player like Messi, the engine prioritizes high-identity strengths such as finishing, chance creation, take on and orchestration. Low or non-defining scores such as air are not used as explanation drivers just because two players are similarly low in those areas. For centre backs, full backs, defensive and central midfielders, ground defence (and air for centre backs) is always part of the search.

The dashboard exposes this through:

```text
similarity_basis
similarity_reason
```

`similarity_basis` shows which target strengths were used for the search, while `similarity_reason` explains which of those strengths are closest for each candidate.

## Defensive Actions View

The dashboard also includes a defensive actions radar and table for the selected player. This view breaks defense into more concrete signals:

```text
tackles/90
tackles won/90
interceptions/90
ball recoveries/90
clearances/90
blocks/90
duels won/90
aerials won/90
tackle success
duel success
aerial win rate
dribbled past control
```

These values are shown as position-group percentiles, so defenders are compared against players with a similar positional context instead of only raw totals.

## Profile Scores

Profiles follow the CIES Football Observatory layout, with one change: air defence and air attack are a single **Air** area, because StatsLook has no duel locations to tell them apart.

| Group | Score | Signal |
|---|---|---|
| Defence | `ground_defence_score` | Possession-adjusted tackles won, interceptions, recoveries and blocks; duel and tackle success; share of challenges lost to a dribble (inverse) |
| Air | `air_score` | Aerials won, aerial win rate, height |
| Possession | `build_up_score` | Passing volume, accurate passes, pass accuracy, long balls, inverse possession loss |
| Possession | `orchestration_score` | Final-third passes, through balls, touches, key passes |
| Possession | `take_on_score` | Successful and attempted dribbles, dribble success, fouls drawn |
| Possession | `chance_creation_score` | Key passes, chances created, big chances created, assists, crosses |
| Attack | `finishing_score` | Goals, xG, xGoT, shots, shots on target |

Each main score is a weighted average of metric percentiles against **all outfield players**, so scores are comparable across positions and drive similarity. A position-relative copy, `<area>_role_score` (for example `finishing_role_score`), ranks the same metrics within the player's position group, for "how good for their role" views.

**Possession adjustment.** Players in dominant teams get fewer chances to defend. Defensive volume is scaled to a 50/50 game (`metric × 50 / opponent possession`). StatsLook has no possession column, so team possession is estimated from the team's passes per match relative to its league average, clipped to 30–70% (`src/features/defensive_context.py`).

**Defence stays in the search.** Similarity normally focuses on the target's strongest areas. For centre backs, full backs, defensive and central midfielders, ground defence (and air for centre backs) is always included with a minimum weight, so a replacement for a defender has to defend too (`POSITION_REQUIRED_PROFILES` in `src/engine/similarity.py`).

Weights live in:

```text
config/profile_weights.yaml
config/replacement_weights.yaml
config/matching_config.yaml
```

## Build the Master Table

Install dependencies:

```bash
pip install -r requirements.txt
```

Build with Transfermarkt optional:

```bash
python -m src.pipeline
```

Require Transfermarkt files:

```bash
python -m src.pipeline --strict-transfermarkt
```

## Run the Dashboard

```bash
streamlit run app.py
```

The dashboard includes:

- Player selection
- Profile radar
- Similar players table
- Replacement shortlist
- Filters for minutes, age, market value, position group, and league
