from __future__ import annotations

import pandas as pd

from src.features.profile_scores import profile_group, profile_vector_columns


PROFILE_GROUP_COLORS = {
    "defence": "#8d6e63",
    "air": "#9575cd",
    "possession": "#4a8fd4",
    "attack": "#d9534f",
}


DEFENSIVE_RADAR_METRICS = [
    ("Tackles/90", "Tackles_per90", False),
    ("Tackles Won/90", "Tackles_Won_per90", False),
    ("Interceptions/90", "Interception_per90", False),
    ("Recoveries/90", "Ball_Recovery_per90", False),
    ("Clearances/90", "Clearences_per90", False),
    ("Blocks/90", "Blocked_Shots_per90", False),
    ("Duels Won/90", "Duels_Won_per90", False),
    ("Aerials Won/90", "Aerials_Won_per90", False),
    ("Tackle Success", "tackle_success_rate", False),
    ("Duel Success", "duel_success_rate", False),
    ("Aerial Win Rate", "aerial_win_rate", False),
    ("Dribbled Past Control", "Dribbled_Past_per90", True),
]


def _numeric_value(value) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])


def _metric_percentile(
    df: pd.DataFrame,
    player_row: pd.Series,
    metric: str,
    inverse: bool = False,
) -> float:
    if metric not in df.columns:
        return 0.0

    comparison = df
    if "position_group" in df.columns and pd.notna(player_row.get("position_group")):
        same_group = df["position_group"].eq(player_row.get("position_group"))
        if same_group.sum() >= 5:
            comparison = df[same_group]

    values = pd.to_numeric(comparison[metric], errors="coerce").fillna(0)
    if inverse:
        values = -values
    percentiles = values.rank(pct=True, method="average") * 100
    if player_row.name not in percentiles.index:
        return 0.0
    return round(float(percentiles.loc[player_row.name]), 2)


def build_metric_percentile_table(
    df: pd.DataFrame,
    player_row: pd.Series,
    metric_specs: list[tuple[str, str, bool]] | None = None,
) -> pd.DataFrame:
    specs = metric_specs or DEFENSIVE_RADAR_METRICS
    rows = []
    for label, metric, inverse in specs:
        if metric not in df.columns:
            continue
        rows.append(
            {
                "Metric": label,
                "Value": round(_numeric_value(player_row.get(metric)), 2),
                "Percentile": _metric_percentile(df, player_row, metric, inverse=inverse),
            }
        )
    return pd.DataFrame(rows)


def build_metric_percentile_radar(
    df: pd.DataFrame,
    player_row: pd.Series,
    metric_specs: list[tuple[str, str, bool]] | None = None,
    title: str | None = None,
):
    import plotly.graph_objects as go

    table = build_metric_percentile_table(df, player_row, metric_specs)
    labels = table["Metric"].tolist()
    values = table["Percentile"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            name=player_row.get("playerName", "Player"),
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=32, r=32, t=48, b=32),
    )
    return fig


def build_profile_radar(
    player_row: pd.Series,
    score_columns: list[str] | None = None,
    title: str | None = None,
):
    import plotly.graph_objects as go

    columns = score_columns or profile_vector_columns()
    labels = [column.replace("_score", "").replace("_", " ").title() for column in columns]
    values = [
        float(pd.to_numeric(pd.Series([player_row.get(column)]), errors="coerce").fillna(0).iloc[0])
        for column in columns
    ]
    colors = [PROFILE_GROUP_COLORS.get(profile_group(column), "#888888") for column in columns]

    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=values,
            theta=labels,
            marker_color=colors,
            marker_line_color="white",
            marker_line_width=2,
            opacity=0.85,
            text=[f"{value:.0f}" for value in values],
            hovertemplate="%{theta}: %{r:.0f}<extra></extra>",
            name=player_row.get("playerName", "Player"),
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=32, r=32, t=48, b=32),
    )
    return fig


def build_comparison_radar(
    target_row: pd.Series,
    candidate_row: pd.Series,
    score_columns: list[str] | None = None,
    title: str | None = None,
):
    import plotly.graph_objects as go

    columns = score_columns or profile_vector_columns()
    labels = [column.replace("_score", "").replace("_", " ").title() for column in columns]

    def values(row: pd.Series) -> list[float]:
        return [
            float(pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").fillna(0).iloc[0])
            for column in columns
        ]

    target_values = values(target_row)
    candidate_values = values(candidate_row)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=target_values + target_values[:1],
            theta=labels + labels[:1],
            fill="toself",
            line_color="#f28e2b",
            opacity=0.6,
            name=target_row.get("playerName", "Target"),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=candidate_values + candidate_values[:1],
            theta=labels + labels[:1],
            fill="toself",
            line_color="#4a8fd4",
            opacity=0.6,
            name=candidate_row.get("playerName", "Candidate"),
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        margin=dict(l=32, r=32, t=48, b=32),
    )
    return fig
