from __future__ import annotations

import numpy as np
import pandas as pd


POSSESSION_ADJUSTED_COLUMNS = [
    "Tackles_Won_per90",
    "Interception_per90",
    "Ball_Recovery_per90",
    "Blocked_Shots_per90",
]

POSSESSION_MIN = 30.0
POSSESSION_MAX = 70.0


def estimate_team_possession(
    df: pd.DataFrame,
    team_col: str = "teamName",
    league_col: str = "league_display_name",
) -> pd.Series:
    """Rough team possession (%) from passing volume.

    StatsLook has no possession column, so a team's passes per match relative
    to its league average stands in for it: a team passing 20% more than its
    league average is estimated at 60% possession. Unmapped leagues use the
    average over all teams.
    """
    minutes = pd.to_numeric(df["Min"], errors="coerce").fillna(0)
    passes = pd.to_numeric(df["Passes"], errors="coerce").fillna(0)
    played = minutes > 0
    team = (
        pd.DataFrame({"team": df[team_col], "passes": passes, "minutes": minutes})[played]
        .groupby("team")
        .agg(passes=("passes", "sum"), minutes=("minutes", "sum"))
    )
    team["passes_per_match"] = team["passes"] / team["minutes"] * 90 * 11

    if league_col in df.columns:
        team_league = df[played].groupby(team_col)[league_col].first()
        team["league"] = team_league.reindex(team.index)
        league_avg = team.groupby("league")["passes_per_match"].transform("mean")
    else:
        league_avg = pd.Series(np.nan, index=team.index)
    league_avg = league_avg.fillna(team["passes_per_match"].mean())

    possession = (50 * team["passes_per_match"] / league_avg).clip(POSSESSION_MIN, POSSESSION_MAX)
    return df[team_col].map(possession).fillna(50.0)


def add_defensive_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add possession-adjusted defensive volume and a duel-loss rate.

    Possession adjustment scales defensive actions to a 50/50 game:
    metric * 50 / opponent possession. Players in dominant teams get fewer
    chances to defend, so raw volume undersells them.
    """
    result = df.copy()
    result["team_possession_est"] = estimate_team_possession(result)
    opponent_possession = 100 - result["team_possession_est"]
    for column in POSSESSION_ADJUSTED_COLUMNS:
        if column in result.columns:
            result[f"{column}_padj"] = (
                pd.to_numeric(result[column], errors="coerce").fillna(0) * 50 / opponent_possession
            )

    # Share of 1v1 defensive challenges lost: dribbled past / (tackles + dribbled past).
    tackles = pd.to_numeric(result.get("Tackles"), errors="coerce").fillna(0)
    dribbled_past = pd.to_numeric(result.get("Dribbled_Past"), errors="coerce").fillna(0)
    challenges = (tackles + dribbled_past).replace(0, np.nan)
    result["challenge_loss_rate"] = (dribbled_past / challenges * 100).fillna(50.0)
    return result
