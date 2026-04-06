from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def possessions_involved(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count the number of unique possessions where each midfielder touched the ball.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with counts of unique possessions per player.
    """
    df = ctx.player_events.dropna(subset=["player_id", "possession"])
    counts = (
        df.groupby("player_id")["possession"]
        .nunique()
        .astype(float)
    )
    return ctx.ensure_index(counts, fill_value=0.0)


def possession_time_seconds(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate total on-ball time in seconds for each midfielder.

    Computes time between consecutive touches within the same possession,
    capping gaps at 10 seconds to avoid inflated durations from long breaks.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with total possession time in seconds.
    """
    df = ctx.player_events.dropna(subset=["player_id", "possession", "timestamp_seconds"])
    df = df.sort_values(["player_id", "possession", "timestamp_seconds"])
    next_time = df.groupby(["player_id", "possession"])["timestamp_seconds"].shift(-1)
    durations = (next_time - df["timestamp_seconds"]).clip(lower=0.0)
    durations = durations.fillna(0.0).clip(upper=10.0)
    df = df.assign(duration_temp=durations)
    totals = df.groupby("player_id")["duration_temp"].sum()
    return ctx.ensure_index(totals, fill_value=0.0)


def tempo_index(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate tempo index: passes + carries per on-ball minute.

    Higher values indicate faster involvement and quicker decision-making.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with tempo index (actions per minute).
        Returns NaN for players with no possession time.
    """
    possession_time = possession_time_seconds(ctx)
    passes = ctx.player_events[ctx.player_events["type_name"] == "Pass"]
    carries = ctx.player_events[ctx.player_events["type_name"] == "Carry"]
    pass_counts = passes.groupby("player_id")["type_name"].count().astype(float)
    carry_counts = carries.groupby("player_id")["type_name"].count().astype(float)
    actions = pass_counts.add(carry_counts, fill_value=0.0)
    tempo = actions / (possession_time.replace(0.0, np.nan) / 60.0)
    return ctx.ensure_index(tempo, fill_value=np.nan)


def turnovers(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count dispossessions and miscontrols committed by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with turnover counts.
    """
    mask = ctx.player_events["type_name"].isin(["Dispossessed", "Miscontrol"])
    counts = (
        ctx.player_events[mask]
        .groupby("player_id")["type_name"]
        .count()
        .astype(float)
    )
    return ctx.ensure_index(counts, fill_value=0.0)

