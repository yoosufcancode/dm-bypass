from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def pressures_applied(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count pressure events applied by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pressure counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "Pressure"]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def ball_recoveries(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count ball recoveries credited to each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with ball recovery counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "Ball Recovery"]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def interceptions(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count interceptions credited to each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with interception counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "Interception"]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def tackles_won(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count tackles won (successful tackle duels) by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with tackle win counts.
    """
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Duel")
        & (ctx.player_events.get("duel.type.name") == "Tackle")
        & (
            ctx.player_events.get("duel.outcome.name").str.contains("Won", na=False)
        )
    ]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def press_to_interception_chain(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count pressures that lead to an interception within 5 seconds in the same possession.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with press-to-interception chain counts.
    """
    pressures = ctx.player_events[ctx.player_events["type_name"] == "Pressure"]
    interceptions_df = ctx.team_events[ctx.team_events["type_name"] == "Interception"]
    if pressures.empty or interceptions_df.empty:
        return ctx.players_series(default=0.0)

    counts = {}
    for player_id, group in pressures.groupby("player_id"):
        success = 0
        for _, row in group.iterrows():
            t = row["timestamp_seconds"]
            possession = row["possession"]
            intercept = interceptions_df[
                (interceptions_df["possession"] == possession)
                & (interceptions_df["timestamp_seconds"] >= t)
                & (interceptions_df["timestamp_seconds"] <= t + 5)
            ]
            if not intercept.empty:
                success += 1
        counts[player_id] = float(success)
    series = pd.Series(counts, dtype=float)
    return ctx.ensure_index(series, fill_value=0.0)


def counterpress_actions(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count counterpressing actions (events marked as counterpress) by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with counterpress action counts.
    """
    df = ctx.player_events[ctx.player_events["counterpress"] == True]
    counts = df.groupby("player_id")["counterpress"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def pressure_to_self_recovery(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count pressures that lead to the same player recovering the ball within 5 seconds.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pressure-to-self-recovery counts.
    """
    pressures = ctx.player_events[ctx.player_events["type_name"] == "Pressure"]
    recoveries = ctx.player_events[ctx.player_events["type_name"] == "Ball Recovery"]
    if pressures.empty or recoveries.empty:
        return ctx.players_series(default=0.0)

    recoveries = recoveries.sort_values("timestamp_seconds")
    counts = {}
    for player_id, group in pressures.groupby("player_id"):
        success = 0
        player_recoveries = recoveries[recoveries["player_id"] == player_id]
        for _, row in group.iterrows():
            t = row["timestamp_seconds"]
            possession = row["possession"]
            window = player_recoveries[
                (player_recoveries["possession"] == possession)
                & (player_recoveries["timestamp_seconds"] >= t)
                & (player_recoveries["timestamp_seconds"] <= t + 5)
            ]
            if not window.empty:
                success += 1
        counts[player_id] = float(success)
    return ctx.ensure_index(pd.Series(counts, dtype=float), fill_value=0.0)


def blocked_passes(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count pass blocks by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pass block counts.
    """
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Block")
        & (ctx.player_events.get("block.block_type") == "Pass Block")
    ]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def blocked_shots(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count shot blocks by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with shot block counts.
    """
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Block")
        & (ctx.player_events.get("block.block_type") == "Shot Block")
    ]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def clearance_followed_by_recovery(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count clearances that are followed by a team ball recovery within 5 seconds.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with clearance-to-recovery chain counts.
    """
    clearances = ctx.player_events[ctx.player_events["type_name"] == "Clearance"]
    recoveries = ctx.team_events[ctx.team_events["type_name"] == "Ball Recovery"]
    if clearances.empty or recoveries.empty:
        return ctx.players_series(default=0.0)

    counts = {}
    for player_id, group in clearances.groupby("player_id"):
        success = 0
        for _, row in group.iterrows():
            t = row["timestamp_seconds"]
            possession = row["possession"]
            window = recoveries[
                (recoveries["possession"] == possession)
                & (recoveries["timestamp_seconds"] > t)
                & (recoveries["timestamp_seconds"] <= t + 5)
            ]
            if not window.empty:
                success += 1
        counts[player_id] = float(success)
    return ctx.ensure_index(pd.Series(counts, dtype=float), fill_value=0.0)


def pressures_to_turnover_rate(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate the rate at which pressures force opponent turnovers within 3 seconds.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pressure-to-turnover rates (0.0 to 1.0).
        Returns NaN for players with no pressures applied.
    """
    pressures = pressures_applied(ctx)
    if pressures.empty or pressures.sum() == 0:
        return ctx.players_series(default=np.nan)

    opponent_turnovers = ctx.opponent_events[
        ctx.opponent_events["type_name"].isin(["Dispossessed", "Miscontrol"])
    ]
    if opponent_turnovers.empty:
        return ctx.players_series(default=np.nan)

    counts = {}
    for player_id, group in ctx.player_events[
        ctx.player_events["type_name"] == "Pressure"
    ].groupby("player_id"):
        forced = 0
        for _, row in group.iterrows():
            t = row["timestamp_seconds"]
            possession = row["possession"]
            window = opponent_turnovers[
                (opponent_turnovers["possession"] == possession)
                & (opponent_turnovers["timestamp_seconds"] > t)
                & (opponent_turnovers["timestamp_seconds"] <= t + 3)
            ]
            if not window.empty:
                forced += 1
        counts[player_id] = forced

    forced_series = pd.Series(counts, dtype=float)
    rate = forced_series / pressures.replace(0.0, np.nan)
    return ctx.ensure_index(rate, fill_value=np.nan)

