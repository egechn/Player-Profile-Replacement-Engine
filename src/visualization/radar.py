from __future__ import annotations

import pandas as pd

from src.features.profile_scores import profile_vector_columns


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
            name=target_row.get("playerName", "Target"),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=candidate_values + candidate_values[:1],
            theta=labels + labels[:1],
            fill="toself",
            name=candidate_row.get("playerName", "Candidate"),
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        margin=dict(l=32, r=32, t=48, b=32),
    )
    return fig

