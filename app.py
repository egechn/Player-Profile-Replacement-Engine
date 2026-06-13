from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import streamlit as st

from src.cleaning.normalize_positions import position_group_label
from src.engine.replacement import recommend_replacements
from src.engine.similarity import find_similar_players, resolve_target_index
from src.enrichment.transfermarkt_joiner import build_players_master
from src.features.league_adjustment import apply_league_strength_columns
from src.features.profile_scores import compute_profile_scores, profile_vector_columns
from src.io_utils import read_table, resolve_project_path
from src.visualization.radar import build_profile_radar
from src.visualization.tables import format_player_table


MASTER_PATH = Path("data/processed/players_master.parquet")


def format_metric(value, number_format: str = "{:.1f}", missing: str = "n/a") -> str:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return missing
    return number_format.format(value)


def stretch_kwargs(streamlit_func) -> dict[str, str | bool]:
    if "width" in inspect.signature(streamlit_func).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


@st.cache_data(show_spinner=False)
def load_master(rebuild: bool = False) -> pd.DataFrame:
    master_path = resolve_project_path(MASTER_PATH)
    csv_fallback = master_path.with_suffix(".csv")
    if rebuild or (not master_path.exists() and not csv_fallback.exists()):
        df = build_players_master()
    else:
        df = read_table(master_path)
    return compute_profile_scores(apply_league_strength_columns(df))


st.set_page_config(page_title="Player Profile Replacement Engine", layout="wide")
st.title("Player Profile Replacement Engine")

with st.sidebar:
    rebuild = st.button("Rebuild master table")
    df = load_master(rebuild=rebuild)

    min_minutes = st.slider("Minimum minutes", 0, 3000, 600, 100)
    max_age = st.slider("Maximum age", 16, 40, 32, 1)
    max_market_value = st.number_input(
        "Maximum market value EUR",
        min_value=0,
        value=0,
        step=500000,
    )
    position_values = sorted(
        value
        for value in df["position_group"].dropna().astype(str).unique().tolist()
        if not value.replace("_", "").replace(".", "").isdigit()
    )
    position_labels = {position_group_label(value): value for value in position_values}
    selected_position_labels = st.multiselect("Position groups", sorted(position_labels))
    leagues = sorted(df["league_display_name"].dropna().unique().tolist())
    selected_leagues = st.multiselect("Leagues", leagues)

player_options = (
    df.sort_values("playerName")["playerName"].dropna().astype(str).unique().tolist()
)
default_player = (
    df.sort_values("Min", ascending=False)["playerName"].dropna().astype(str).iloc[0]
)
default_index = player_options.index(default_player) if default_player in player_options else 0
selected_player = st.selectbox("Player", player_options, index=default_index)
target_index = resolve_target_index(df, player_name=selected_player)
target = df.loc[target_index]

status_cols = st.columns(5)
status_cols[0].metric("Players", f"{len(df):,}")
status_cols[1].metric("Selected age", format_metric(target.get("age")))
status_cols[2].metric("Minutes", format_metric(target.get("Min"), "{:,.0f}"))
match_confidence = target.get("match_confidence")
status_cols[3].metric(
    "TM match",
    "missing" if pd.isna(match_confidence) or match_confidence == 0 else f"{match_confidence:.0f}",
)
status_cols[4].metric(
    "League coeff",
    format_metric(target.get("league_strength_coefficient"), "{:.2f}"),
)

profile_cols = profile_vector_columns(df)
left, right = st.columns([1, 1])
with left:
    st.plotly_chart(
        build_profile_radar(target, profile_cols, title=f"{selected_player} profile"),
        **stretch_kwargs(st.plotly_chart),
    )

with right:
    profile_table = target[profile_cols].rename(
        index=lambda value: value.replace("_score", "").replace("_", " ").title()
    )
    st.dataframe(profile_table.reset_index().rename(columns={"index": "Profile", target.name: "Score"}))

max_value_filter = max_market_value if max_market_value > 0 else None
position_filter = [position_labels[label] for label in selected_position_labels] or None
league_filter = selected_leagues or None

similar = find_similar_players(
    df,
    player_name=selected_player,
    top_n=10,
    min_minutes=min_minutes,
    max_age=max_age,
    max_market_value_eur=max_value_filter,
    leagues=league_filter,
    positions=position_filter,
)
st.subheader("Most similar players")
st.dataframe(format_player_table(similar), **stretch_kwargs(st.dataframe))

shortlist = recommend_replacements(
    df,
    player_name=selected_player,
    top_n=10,
    min_minutes=min_minutes,
    max_age=max_age,
    max_market_value_eur=max_value_filter,
    leagues=league_filter,
    positions=position_filter,
)
st.subheader("Replacement shortlist")
st.dataframe(format_player_table(shortlist), **stretch_kwargs(st.dataframe))

if df["match_method"].eq("unmatched").all():
    st.info(
        "Transfermarkt files are not loaded yet. Add the Kaggle CSVs under "
        "data/raw/transfermarkt/ and rebuild the master table."
    )
