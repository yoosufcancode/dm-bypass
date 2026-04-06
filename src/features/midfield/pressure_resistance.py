from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def pressured_touches(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count touches while under pressure for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pressured touch counts.
    """
    df = ctx.player_events[ctx.player_events["under_pressure"] == True]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def pressured_touch_retention_rate(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate retention rate of touches while under pressure.

    A touch is retained if it is not immediately followed by a dispossession
    or miscontrol within 2 seconds.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pressured touch retention rates (0.0 to 1.0).
        Returns NaN for players with no pressured touches.
    """
    pressured = ctx.player_events[ctx.player_events["under_pressure"] == True]
    if pressured.empty:
        return ctx.players_series(default=np.nan)

    events_sorted = ctx.team_events.sort_values("timestamp_seconds")
    retention = {}
    for player_id, group in pressured.groupby("player_id"):
        retained = 0
        for _, row in group.iterrows():
            t = row["timestamp_seconds"]
            next_events = events_sorted[
                (events_sorted["timestamp_seconds"] > t)
                & (events_sorted["timestamp_seconds"] <= t + 2)
            ]
            same_player = next_events[next_events["player_id"] == player_id]
            if same_player.empty:
                retained += 1
            else:
                next_types = same_player["type_name"].tolist()
                if not any(evt in ["Dispossessed", "Miscontrol"] for evt in next_types):
                    retained += 1
        total = len(group)
        retention[player_id] = retained / total if total else np.nan
    series = pd.Series(retention, dtype=float)
    return ctx.ensure_index(series, fill_value=np.nan)

