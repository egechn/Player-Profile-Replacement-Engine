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

## Data Layout

```text
data/
  raw/
    statslook/
      Statslook_Scouting_Master_Final.csv
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

This file maps Statslook `teamName` values to English league names. If a team is not in this mapping, the main league columns stay empty and `league_source` is set to `unmapped`.

Transfermarkt league values are preserved only as raw debug columns such as `tm_league_name` and `tm_league_country`; they are not used as the main league shown in the dashboard.

## League Strength

League strength coefficients live in:

```text
config/league_strength.yaml
```

The coefficients are derived from the Opta Power Rankings league table embedded in the Opta Analyst article. The current formula is:

```text
league_strength_coefficient = Opta seasonAverageRating / Premier League seasonAverageRating
```

Similarity keeps the raw role/profile score as `similarity_score` in the 0-1 range and also outputs `league_adjusted_similarity_score` as a league-context comparison index. The "Most similar players" table is sorted by the raw profile score so league strength does not override role similarity. The adjusted score uses the selected player's league as the reference:

```text
league_strength_gap = candidate_league_coefficient - target_league_coefficient
league_adjustment_factor = 1 + league_similarity_weight * league_strength_gap
league_adjusted_similarity_score = similarity_score * league_adjustment_factor
```

The factor is clipped tightly to prevent league strength from overwhelming the player profile match, but the adjusted index itself can be slightly above 1.0 when a strong profile match comes from a stronger league.

## Profile Scores

The first version creates these profile scores:

| Score | Signal |
|---|---|
| `ball_progressor_score` | Dribbling, final-third passing, touches, through-ball success |
| `creator_score` | Assists, key passes, chances created, big chances created |
| `finisher_score` | Goals, xG, xGoT, shots, shots on target |
| `presser_score` | Tackles, interceptions, ball recoveries, fouls as pressure proxy |
| `ball_winner_score` | Tackles won, interceptions, recoveries, duels won |
| `aerial_score` | Aerials won, aerial win rate, height |
| `possession_security_score` | Pass accuracy, accurate passes, inverse possession loss/turnovers |
| `wide_creator_score` | Crosses, accurate crosses, dribbles, key passes |
| `box_threat_score` | Goals, xG, xGoT, shots, big-chance involvement |
| `deep_playmaker_score` | Passing volume, accurate passes, final-third passes, long/through balls |

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
