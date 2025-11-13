from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def shot_creating_actions(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count shot creating actions (last 2 actions before a shot) for each midfielder.

    Actions within 5 seconds before a shot in the same possession are counted.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with shot creating action counts.
    """
    events = ctx.team_events.sort_values("timestamp_seconds")
    shots = events[events["type_name"] == "Shot"]
    if shots.empty:
        return ctx.players_series(default=0.0)

    counts = ctx.players_series(default=0.0)
    for _, shot in shots.iterrows():
        poss = shot["possession"]
        shot_time = shot["timestamp_seconds"]
        window = events[
            (events["possession"] == poss)
            & (events["timestamp_seconds"] < shot_time)
            & (events["timestamp_seconds"] >= shot_time - 5)
        ]
        last_two = window.tail(2)
        for _, action in last_two.iterrows():
            player_id = action["player_id"]
            if player_id in counts.index:
                counts.loc[player_id] += 1.0
    return counts


def expected_threat_added(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Placeholder: Expected threat model not provided. Return NaN for all players.
    """
    return ctx.players_series(default=np.nan)

