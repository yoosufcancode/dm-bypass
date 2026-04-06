from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def secondary_shot_assists(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count secondary shot assists (pre-assists) for each midfielder.

    A pre-assist is the second-to-last action before a shot in the same possession.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with secondary shot assist counts.
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
        ].tail(2)
        if len(window) >= 2:
            pre_assist = window.iloc[-2]
            if pre_assist["player_id"] in counts.index:
                counts.loc[pre_assist["player_id"]] += 1.0
    return counts


def expected_assists(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate expected assists (xA) for each midfielder based on shot xG.

    Sums the xG of shots that were assisted by each player's passes.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with expected assists (xA) values.
        Returns NaN if shot xG data is not available.
    """
    passes = ctx.player_events[
        (ctx.player_events["type_name"] == "Pass")
        & (ctx.player_events.get("pass.shot_assist") == True)
    ]
    if passes.empty or "shot.statsbomb_xg" not in ctx.team_events.columns:
        return ctx.players_series(default=np.nan)

    xg_map = ctx.team_events[
        ctx.team_events["type_name"] == "Shot"
    ][["id", "shot.statsbomb_xg"]]

    assist_map = {}
    for _, shot in ctx.team_events[ctx.team_events["type_name"] == "Shot"].iterrows():
        key_pass_id = shot.get("shot.key_pass_id")
        if key_pass_id:
            assist_map[key_pass_id] = shot.get("shot.statsbomb_xg")

    xa = {}
    for _, row in passes.iterrows():
        key = row.get("id")
        player_id = row["player_id"]
        if key and key in assist_map:
            xa[player_id] = xa.get(player_id, 0.0) + float(assist_map[key])

    return ctx.ensure_index(pd.Series(xa, dtype=float), fill_value=np.nan)


def xg_chain(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate xG chain contribution for each midfielder.

    Sums the xG from shots in possessions where the player participated.
    All players who touched the ball in a possession leading to a shot
    receive credit for the full xG of that shot.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with xG chain contribution values.
        Returns NaN if shot xG data is not available.
    """
    shots = ctx.team_events[ctx.team_events["type_name"] == "Shot"]
    if shots.empty or "shot.statsbomb_xg" not in shots.columns:
        return ctx.players_series(default=np.nan)

    contributions = {}
    events = ctx.team_events.sort_values("timestamp_seconds")
    for _, shot in shots.iterrows():
        poss = shot["possession"]
        xg = shot.get("shot.statsbomb_xg", np.nan)
        if pd.isna(xg):
            continue
        possession_events = events[
            (events["possession"] == poss)
            & (events["timestamp_seconds"] <= shot["timestamp_seconds"])
        ]
        players = possession_events["player_id"].dropna().unique()
        for player_id in players:
            if player_id in ctx.midfielder_ids:
                contributions[player_id] = contributions.get(player_id, 0.0) + float(xg)

    return ctx.ensure_index(pd.Series(contributions, dtype=float), fill_value=np.nan)

