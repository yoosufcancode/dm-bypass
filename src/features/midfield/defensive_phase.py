"""
Defensive phase features for midfielders.

These features measure midfielder behaviour WHEN THE OPPONENT HAS POSSESSION —
directly relevant to predicting how often the opponent bypasses the midfield.

All functions return a pd.Series indexed by player_id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext

# Pitch zone constants (StatsBomb 120x80 coordinate system)
_MIDFIELD_X_LOW = 40.0
_MIDFIELD_X_HIGH = 80.0


def defensive_midfield_actions(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count defensive actions each midfielder performs inside the midfield zone
    (x = 40–80) while the opponent is in possession.

    Covers: Pressure, Interception, Ball Recovery, Tackle (Duel), Block.
    This measures how actively each midfielder defends the zone that bypasses
    are designed to skip through.

    Returns
    -------
    pd.Series
        Indexed by player_id, counts of defensive midfield zone actions.
    """
    DEFENSIVE_TYPES = {"Pressure", "Interception", "Ball Recovery", "Duel", "Block"}

    # Opponent-possession events
    opp_poss = ctx.events[
        (ctx.events["possession_team_id"] != ctx.team_id)
        & ctx.events["possession_team_id"].notna()
    ]

    if opp_poss.empty:
        return ctx.players_series(default=0.0)

    # Barcelona midfielders acting during opponent possession
    mid_def = opp_poss[
        opp_poss["player_id"].isin(ctx.midfielder_ids)
        & opp_poss["type_name"].isin(DEFENSIVE_TYPES)
        & opp_poss["x"].between(_MIDFIELD_X_LOW, _MIDFIELD_X_HIGH, inclusive="both")
    ]

    counts = mid_def.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def transition_pressure_rate(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count defensive actions each midfielder makes in the 10 seconds immediately
    after Barcelona loses possession (transition-to-defence window).

    A high value means the midfielder engages quickly when possession is lost —
    the direct counter to a bypass opportunity opening up.

    Returns
    -------
    pd.Series
        Indexed by player_id, counts of transition defensive actions.
    """
    DEFENSIVE_TYPES = {"Pressure", "Interception", "Ball Recovery", "Duel", "Block",
                       "Clearance", "Foul Committed"}
    TRANSITION_WINDOW = 10.0  # seconds

    events = ctx.events.copy()
    if "timestamp_seconds" not in events.columns:
        from .context import _timestamp_to_seconds
        events["timestamp_seconds"] = _timestamp_to_seconds(events["timestamp"])

    # Identify possession transitions: rows where possession_team changes to opponent
    events = events.sort_values("timestamp_seconds").reset_index(drop=True)
    events["prev_poss_team"] = events["possession_team_id"].shift(1)
    transitions = events[
        (events["prev_poss_team"] == ctx.team_id)
        & (events["possession_team_id"] != ctx.team_id)
        & events["possession_team_id"].notna()
    ]

    if transitions.empty:
        return ctx.players_series(default=0.0)

    counts = {pid: 0.0 for pid in ctx.midfielder_ids}

    for _, t_row in transitions.iterrows():
        t0 = t_row["timestamp_seconds"]
        window = events[
            (events["timestamp_seconds"] > t0)
            & (events["timestamp_seconds"] <= t0 + TRANSITION_WINDOW)
            & events["player_id"].isin(ctx.midfielder_ids)
            & events["type_name"].isin(DEFENSIVE_TYPES)
        ]
        for pid in window["player_id"].dropna().astype(int):
            if pid in counts:
                counts[pid] += 1.0

    return ctx.ensure_index(pd.Series(counts, dtype=float), fill_value=0.0)


def midfield_zone_coverage_x(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Average x-position of each midfielder during opponent possession sequences.

    Lower x = midfielder sits deeper when defending (own half).
    Higher x = midfielder stays high (pressing, not tracking back).

    Reflects defensive positioning — teams that sit deep give the opponent
    more room to bypass in the x=40–80 zone.

    Returns
    -------
    pd.Series
        Indexed by player_id, mean x-position during opponent possession.
    """
    opp_poss = ctx.events[
        (ctx.events["possession_team_id"] != ctx.team_id)
        & ctx.events["possession_team_id"].notna()
        & ctx.events["player_id"].isin(ctx.midfielder_ids)
        & ctx.events["x"].notna()
    ]

    if opp_poss.empty:
        return ctx.players_series(default=np.nan)

    means = opp_poss.groupby("player_id")["x"].mean().astype(float)
    return ctx.ensure_index(means, fill_value=np.nan)


def press_on_deep_opponent_possession(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count pressures by each midfielder when the opponent has possession
    in their own defensive zone (x < 40) — i.e., pressing the buildup
    BEFORE it becomes a bypass opportunity.

    This is the most direct counter to bypass: if midfielders press the
    opponent deep, the opponent cannot build up the pace and space needed
    to skip the midfield zone.

    Returns
    -------
    pd.Series
        Indexed by player_id, counts of deep-opposition pressures.
    """
    deep_opp = ctx.events[
        (ctx.events["possession_team_id"] != ctx.team_id)
        & ctx.events["possession_team_id"].notna()
        & ctx.events["x"].lt(_MIDFIELD_X_LOW)  # opponent in their defensive zone
    ]

    if deep_opp.empty:
        return ctx.players_series(default=0.0)

    actions = deep_opp[
        (deep_opp["player_id"].isin(ctx.midfielder_ids))
        & (deep_opp["type_name"] == "Pressure")
    ]

    counts = actions.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def midfield_presence_on_deep_opp_possession(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count the number of events each midfielder has in the midfield zone
    (x = 40–80) during opponent possessions that STARTED in the defensive
    zone (x < 40) — i.e., active presence in the bypass channel when
    a bypass threat is live.

    High value = midfielder holds position in the bypass channel when
    the opponent is building from deep.

    Returns
    -------
    pd.Series
        Indexed by player_id, counts of midfield-zone events during bypass-threat windows.
    """
    events = ctx.events.copy()
    if "timestamp_seconds" not in events.columns:
        from .context import _timestamp_to_seconds
        events["timestamp_seconds"] = _timestamp_to_seconds(events["timestamp"])

    events = events.sort_values("timestamp_seconds").reset_index(drop=True)

    # Identify possessions where the opponent starts deep
    opp_poss = events[
        (events["possession_team_id"] != ctx.team_id)
        & events["possession_team_id"].notna()
    ]

    deep_start_possessions: set = set()
    for poss_id, grp in opp_poss.groupby("possession"):
        first_x = grp.sort_values("timestamp_seconds")["x"].iloc[0]
        if pd.notna(first_x) and first_x < _MIDFIELD_X_LOW:
            deep_start_possessions.add(poss_id)

    if not deep_start_possessions:
        return ctx.players_series(default=0.0)

    # Midfielder events in midfield zone during these possessions
    threat_events = events[
        events["possession"].isin(deep_start_possessions)
        & events["player_id"].isin(ctx.midfielder_ids)
        & events["x"].between(_MIDFIELD_X_LOW, _MIDFIELD_X_HIGH, inclusive="both")
    ]

    counts = threat_events.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def bypass_channel_defensive_actions(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count defensive actions (Pressure, Interception, Tackle, Block, Ball Recovery)
    performed by each midfielder specifically in the midfield zone (x = 40–80)
    during opponent possessions that STARTED deep (x < 40).

    This is more focused than defensive_midfield_actions because it only counts
    actions during actual bypass-threat possessions, not all opponent possession.

    Returns
    -------
    pd.Series
        Indexed by player_id, counts of in-channel defensive actions during bypass threats.
    """
    DEFENSIVE_TYPES = {"Pressure", "Interception", "Ball Recovery", "Duel", "Block"}

    events = ctx.events.copy()
    if "timestamp_seconds" not in events.columns:
        from .context import _timestamp_to_seconds
        events["timestamp_seconds"] = _timestamp_to_seconds(events["timestamp"])

    opp_poss = events[
        (events["possession_team_id"] != ctx.team_id)
        & events["possession_team_id"].notna()
    ]

    deep_start_possessions: set = set()
    for poss_id, grp in opp_poss.groupby("possession"):
        first_x = grp.sort_values("timestamp_seconds")["x"].iloc[0]
        if pd.notna(first_x) and first_x < _MIDFIELD_X_LOW:
            deep_start_possessions.add(poss_id)

    if not deep_start_possessions:
        return ctx.players_series(default=0.0)

    actions = events[
        events["possession"].isin(deep_start_possessions)
        & events["player_id"].isin(ctx.midfielder_ids)
        & events["type_name"].isin(DEFENSIVE_TYPES)
        & events["x"].between(_MIDFIELD_X_LOW, _MIDFIELD_X_HIGH, inclusive="both")
    ]

    counts = actions.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def avg_defensive_x_on_deep_opp(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Average x-position of each midfielder's events during opponent possessions
    that STARTED deep (x < 40).

    Lower x = midfielder drops deeper during bypass threats (tracking back).
    Higher x = midfielder stays high / presses (tries to prevent buildup).

    Unlike midfield_zone_coverage_x (all opponent possession), this only
    looks at the specific scenario the bypass definition targets.

    Returns
    -------
    pd.Series
        Indexed by player_id, mean x-position during deep-start opponent possessions.
    """
    events = ctx.events.copy()
    if "timestamp_seconds" not in events.columns:
        from .context import _timestamp_to_seconds
        events["timestamp_seconds"] = _timestamp_to_seconds(events["timestamp"])

    opp_poss = events[
        (events["possession_team_id"] != ctx.team_id)
        & events["possession_team_id"].notna()
    ]

    deep_start_possessions: set = set()
    for poss_id, grp in opp_poss.groupby("possession"):
        first_x = grp.sort_values("timestamp_seconds")["x"].iloc[0]
        if pd.notna(first_x) and first_x < _MIDFIELD_X_LOW:
            deep_start_possessions.add(poss_id)

    if not deep_start_possessions:
        return ctx.players_series(default=np.nan)

    mid_events = events[
        events["possession"].isin(deep_start_possessions)
        & events["player_id"].isin(ctx.midfielder_ids)
        & events["x"].notna()
    ]

    if mid_events.empty:
        return ctx.players_series(default=np.nan)

    means = mid_events.groupby("player_id")["x"].mean().astype(float)
    return ctx.ensure_index(means, fill_value=np.nan)


def defensive_shape_compactness(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Variance of midfielder x-positions sampled at moments when the opponent
    has possession in the midfield zone (x = 40–80).

    Low variance = midfield line is compact and narrow → hard to bypass.
    High variance = midfield is stretched → gaps for bypasses to exploit.

    Computed as a match-half constant spread equally across all midfielders
    (since compactness is a collective property, not an individual one).

    Returns
    -------
    pd.Series
        Indexed by player_id, all sharing the same compactness value.
    """
    # Opponent in possession and in the midfield zone
    opp_mid = ctx.events[
        (ctx.events["possession_team_id"] != ctx.team_id)
        & ctx.events["possession_team_id"].notna()
        & ctx.events["x"].between(_MIDFIELD_X_LOW, _MIDFIELD_X_HIGH, inclusive="both")
    ]

    if opp_mid.empty:
        return ctx.players_series(default=np.nan)

    # For each such moment, record x-positions of all midfielders with events nearby
    events_all = ctx.events.copy()
    if "timestamp_seconds" not in events_all.columns:
        from .context import _timestamp_to_seconds
        events_all["timestamp_seconds"] = _timestamp_to_seconds(events_all["timestamp"])

    mid_events = events_all[
        events_all["player_id"].isin(ctx.midfielder_ids)
        & events_all["x"].notna()
    ].sort_values("timestamp_seconds")

    if mid_events.empty:
        return ctx.players_series(default=np.nan)

    # Sample midfielder x-positions at each opponent-in-midfield-zone moment
    x_snapshots = []
    if "timestamp_seconds" not in opp_mid.columns:
        from .context import _timestamp_to_seconds
        opp_mid = opp_mid.copy()
        opp_mid["timestamp_seconds"] = _timestamp_to_seconds(opp_mid["timestamp"])

    for _, opp_row in opp_mid.iterrows():
        t = opp_row.get("timestamp_seconds", np.nan)
        if pd.isna(t):
            continue
        # Closest event per midfielder within ±5s window
        nearby = mid_events[
            (mid_events["timestamp_seconds"] >= t - 5)
            & (mid_events["timestamp_seconds"] <= t + 5)
        ]
        if len(nearby["player_id"].unique()) >= 2:
            snapshot_xs = (
                nearby.groupby("player_id")["x"].mean().values
            )
            x_snapshots.append(np.var(snapshot_xs))

    if not x_snapshots:
        return ctx.players_series(default=np.nan)

    compactness = float(np.mean(x_snapshots))
    result = ctx.players_series(default=compactness)
    return result
