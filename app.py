from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from src.cleaning.normalize_positions import position_group_label
from src.data_sources.statslook_fetch import fetch_statslook, load_fetch_config
from src.engine.replacement import recommend_replacements
from src.cleaning.normalize_names import normalize_text
from src.engine.similarity import find_similar_players
from src.enrichment.transfermarkt_joiner import build_players_master
from src.features.combined_seasons import combine_season_profiles, combined_label, combined_parts
from src.features.league_adjustment import apply_league_strength_columns
from src.features.profile_scores import (
    compute_profile_scores,
    profile_group,
    profile_vector_columns,
    role_column,
)
from src.io_utils import read_table
from src.seasons import (
    available_seasons,
    default_min_minutes,
    base_season,
    default_season,
    is_league_only,
    master_path_for,
    processed_dir_for,
    statslook_source_for,
)
from src.visualization.radar import build_comparison_radar, build_profile_radar
from src.visualization.tables import focus_areas, format_player_table
from src.watchlist import add_to_watchlist, load_watchlist, save_watchlist


def format_metric(value, number_format: str = "{:.1f}", missing: str = "n/a") -> str:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return missing
    return number_format.format(value)


def stretch_kwargs(streamlit_func) -> dict[str, str | bool]:
    # Older Streamlit has an integer-only `width`; only newer versions drop use_container_width.
    if "use_container_width" in inspect.signature(streamlit_func).parameters:
        return {"use_container_width": True}
    return {"width": "stretch"}


@st.cache_data(show_spinner=False)
def load_master(season: str, rebuild: bool = False) -> pd.DataFrame:
    parts = combined_parts(season)
    if len(parts) > 1:
        return combine_season_profiles([load_master(part, rebuild) for part in parts], season)
    master_path = master_path_for(season)
    csv_fallback = master_path.with_suffix(".csv")
    if rebuild or (not master_path.exists() and not csv_fallback.exists()):
        df = build_players_master(
            statslook_path=statslook_source_for(season),
            processed_dir=processed_dir_for(season),
            season=season,
        )
    else:
        df = read_table(master_path)
    if "season" not in df.columns:
        df.insert(0, "season", season)
    # Percentiles are computed within the loaded season only.
    return compute_profile_scores(apply_league_strength_columns(df))


def player_table_with_comparison(
    players: pd.DataFrame, key: str, target: pd.Series, profile_cols: list[str]
) -> None:
    """Player table; clicking a row shows that player's profile next to the target's."""
    if focus_areas(players):
        st.caption(f"Search focused on the selected player's strengths: {focus_areas(players)}.")
    table = format_player_table(players)
    event = st.dataframe(
        table,
        hide_index=True,
        # A new target resets the selection.
        key=f"{key}_{target.get('season')}_{target.get('statslook_row_id')}",
        on_select="rerun",
        selection_mode="single-row",
        **stretch_kwargs(st.dataframe),
    )
    target_name = target.get("displayName") or target.get("playerName")
    st.download_button(
        "Download table (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"{key.replace('_table', '')}_{target_name}_{target.get('season')}.csv".replace(" ", "_"),
        mime="text/csv",
        key=f"download_{key}",
    )
    rows = event.selection.rows if event else []
    if not rows or rows[0] >= len(players):
        st.caption("Click a row to compare that player's profile with the selected player.")
        return
    candidate = players.iloc[rows[0]]
    candidate_name = candidate.get("displayName") or candidate.get("playerName")
    if st.button(f"☆ Add {candidate_name} to watchlist", key=f"watch_{key}"):
        if add_to_watchlist(candidate, table.iloc[rows[0]], similar_to=f"{target_name} ({target.get('season')})"):
            st.success(f"{candidate_name} added to the watchlist.")
        else:
            st.info(f"{candidate_name} is already on the watchlist.")
    target_label = f"{target_name} ({target.get('season')})"
    candidate_label = f"{candidate_name} ({candidate.get('season')})"
    labelled_target, labelled_candidate = target.copy(), candidate.copy()
    labelled_target["playerName"], labelled_candidate["playerName"] = target_label, candidate_label
    chart_left, chart_right = st.columns([1, 1])
    with chart_left:
        st.plotly_chart(
            build_profile_radar(candidate, profile_cols, title=f"{candidate_label} profile"),
            **stretch_kwargs(st.plotly_chart),
        )
    with chart_right:
        comparison = build_comparison_radar(
            labelled_target, labelled_candidate, profile_cols, title="Profile comparison"
        )
        comparison.update_layout(legend=dict(orientation="h", x=0, y=-0.12))
        st.plotly_chart(comparison, **stretch_kwargs(st.plotly_chart))


st.set_page_config(page_title="Player Profile Replacement Engine", layout="wide")
st.title("Player Profile Replacement Engine")

with st.sidebar:
    seasons = available_seasons()
    if not seasons:
        st.error("No StatsLook data found under data/raw/statslook/.")
        st.stop()
    # Last two seasons together: minutes-weighted profile, steadier early in a season.
    full_seasons = [s for s in seasons if not is_league_only(s)]
    # Newest first, each season followed by its league-only variant, the combination last.
    seasons = sorted(seasons, key=lambda s: (base_season(s), not is_league_only(s)))
    if len(full_seasons) >= 2:
        seasons = [combined_label(full_seasons[-2:])] + seasons
    season = st.selectbox(
        "Season",
        seasons[::-1],
        index=seasons[::-1].index(default_season()),
    )
    rebuild = st.button("Rebuild master table")
    fetch_config = load_fetch_config()
    if st.button(f"Fetch {fetch_config.get('season', '')} from StatsLook"):
        with st.spinner("Fetching from StatsLook and building the season..."):
            try:
                fetched = fetch_statslook()
                build_players_master(
                    statslook_path=fetched,
                    processed_dir=processed_dir_for(fetch_config["season"]),
                    season=fetch_config["season"],
                )
                st.cache_data.clear()
                st.rerun()
            except (SystemExit, Exception) as error:
                st.error(f"Fetch failed: {error}")
    df = load_master(season, rebuild=rebuild)
    candidate_season = st.selectbox(
        "Find similar players from",
        seasons[::-1],
        index=seasons[::-1].index(season),
        help="Compare the selected player's profile in the season above with players from this season.",
    )
    pool = df if candidate_season == season else load_master(candidate_season)

    season_minutes = pd.to_numeric(pool["Min"], errors="coerce").fillna(0)
    max_minutes = int(max(season_minutes.max(), 100) // 50 * 50 + 50)
    min_minutes = st.slider(
        "Minimum minutes",
        0,
        max_minutes,
        min(default_min_minutes(candidate_season, season_minutes), max_minutes),
        50,
        key=f"min_minutes_{candidate_season}",
        help="Default comes from config/seasons.yaml (min_minutes) or 20% of a regular starter's minutes this season.",
    )
    max_age = st.slider("Maximum age", 16, 40, 32, 1)
    max_market_value = st.number_input(
        "Maximum market value EUR",
        min_value=0,
        value=0,
        step=500000,
    )
    position_values = sorted(
        value
        for value in pool["position_group"].dropna().astype(str).unique().tolist()
        if not value.replace("_", "").replace(".", "").isdigit()
    )
    position_labels = {position_group_label(value): value for value in position_values}
    selected_position_labels = st.multiselect("Position groups", sorted(position_labels))
    scope_labels = {
        "Same position": "same",
        "Same and adjacent roles": "adjacent",
        "All positions": "all",
    }
    position_scope = scope_labels[
        st.radio(
            "Compare against",
            list(scope_labels),
            index=1,
            help="Ignored when position groups are selected above.",
        )
    ]
    leagues = sorted(pool["league_display_name"].dropna().unique().tolist())
    selected_leagues = st.multiselect("Leagues", leagues)

def search_text(series: pd.Series) -> pd.Series:
    return series.map(normalize_text)


search = st.text_input(
    "Search player",
    placeholder="Name or club, e.g. vinicius, real madrid",
    help="Accents and capitals don't matter. Matches the short name, the full name and the club.",
)
by_minutes = df.sort_values("Min", ascending=False)
matches = by_minutes
if search.strip():
    haystack = (
        search_text(by_minutes["displayName"]) + " " + search_text(by_minutes["playerName"])
        + " " + search_text(by_minutes.get("current_team", by_minutes["teamName"]))
    )
    tokens = normalize_text(search).split()
    matches = by_minutes[pd.concat([haystack.str.contains(t, regex=False) for t in tokens], axis=1).all(axis=1)]
    # Players whose short name matches come first (most minutes first within each group).
    short_name = search_text(matches["displayName"])
    in_short_name = pd.concat([short_name.str.contains(t, regex=False) for t in tokens], axis=1).all(axis=1)
    matches = pd.concat([matches[in_short_name], matches[~in_short_name]])
    if matches.empty:
        st.warning(f"No player matches '{search}'. Showing all players.")
        matches = by_minutes
searched = search.strip() and len(matches) < len(by_minutes)
labels = (
    matches["displayName"].fillna(matches["playerName"]).astype(str)
    + " · " + matches.get("current_team", matches["teamName"]).fillna(matches["teamName"]).fillna("no club").astype(str)
    + " · " + matches["Detailed_Position"].fillna("?").astype(str)
    + " · " + pd.to_numeric(matches["Min"], errors="coerce").fillna(0).map("{:,.0f} min".format)
).to_dict()
target_index = st.selectbox(
    f"Player ({len(matches):,} found)" if searched else "Player",
    matches.index.tolist(),
    format_func=labels.get,
    key=f"player_{season}",
)
selected_player = df.loc[target_index].get("displayName") or df.loc[target_index].get("playerName")
target = df.loc[target_index]

status_cols = st.columns(5)
status_cols[0].metric(f"Players {season}", f"{len(df):,}")
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
    profile_table = pd.DataFrame(
        {
            "Area": [value.replace("_score", "").replace("_", " ").title() for value in profile_cols],
            "Group": [(profile_group(value) or "").title() for value in profile_cols],
            "Vs all players": [target.get(value) for value in profile_cols],
            "Vs position": [target.get(role_column(value)) for value in profile_cols],
        }
    )
    st.dataframe(profile_table, hide_index=True)

max_value_filter = max_market_value if max_market_value > 0 else None
position_filter = [position_labels[label] for label in selected_position_labels] or None
league_filter = selected_leagues or None

# Profile scores are percentiles within each season, so a player's standing in one
# season can be compared with players' standing in another.
if candidate_season == season:
    search_df, search_target_id = df, target.get("statslook_row_id")
else:
    search_target_id = -1
    search_df = pd.concat(
        [pool, df.loc[[target_index]].assign(statslook_row_id=search_target_id)],
        ignore_index=True,
    )
    st.caption(
        f"Comparing {selected_player}'s {season} profile with {candidate_season} players "
        f"(the player's own {candidate_season} season is left out)."
    )

similar = find_similar_players(
    search_df,
    statslook_row_id=search_target_id,
    top_n=10,
    min_minutes=min_minutes,
    max_age=max_age,
    max_market_value_eur=max_value_filter,
    leagues=league_filter,
    positions=position_filter,
    position_scope=position_scope,
    use_league_strength=True,
)
st.subheader("Most similar players")
player_table_with_comparison(similar, "similar_table", target, profile_cols)

shortlist = recommend_replacements(
    search_df,
    statslook_row_id=search_target_id,
    top_n=10,
    min_minutes=min_minutes,
    max_age=max_age,
    max_market_value_eur=max_value_filter,
    leagues=league_filter,
    positions=position_filter,
    position_scope=position_scope,
)
st.subheader("Replacement shortlist")
player_table_with_comparison(shortlist, "shortlist_table", target, profile_cols)

st.subheader("Watchlist")
watchlist = load_watchlist()
if watchlist.empty:
    st.caption("No players yet. Click a row in the tables above and press ☆ Add to watchlist.")
else:
    st.caption("Edit notes or delete rows (select a row, then press Delete), then save.")
    edited = st.data_editor(
        watchlist.drop(columns=["player_key"]),
        hide_index=True,
        num_rows="dynamic",
        disabled=[column for column in watchlist.columns if column not in ("Note", "player_key")],
        key="watchlist_editor",
        **stretch_kwargs(st.data_editor),
    )
    save_col, download_col = st.columns([1, 1])
    if save_col.button("Save watchlist changes"):
        # Rows deleted in the editor are dropped; rows added there are ignored.
        kept = watchlist.loc[watchlist.index.intersection(edited.index)].copy()
        kept["Note"] = edited.loc[kept.index, "Note"].fillna("")
        save_watchlist(kept)
        st.success("Watchlist saved.")
        st.rerun()
    download_col.download_button(
        "Download watchlist (CSV)",
        watchlist.drop(columns=["player_key"]).to_csv(index=False).encode("utf-8"),
        file_name="watchlist.csv",
        mime="text/csv",
    )

if df["match_method"].eq("unmatched").all():
    st.info(
        "Transfermarkt files are not loaded yet. Add the Kaggle CSVs under "
        "data/raw/transfermarkt/ and rebuild the master table."
    )
