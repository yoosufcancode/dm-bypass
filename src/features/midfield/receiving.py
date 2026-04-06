from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def _receipts(ctx: MidfieldFeatureContext) -> pd.DataFrame:
    """
    Helper function to filter player events to only ball receipts.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only ball receipt events for midfielders.
    """
    return ctx.player_events[ctx.player_events["type_name"] == "Ball Receipt*"]


def ball_receipts_total(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count total ball receipts for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with total ball receipt counts.
    """
    df = _receipts(ctx)
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def central_lane_receipts(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count ball receipts in the central lane (y 35-45) for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with central lane receipt counts.
    """
    df = _receipts(ctx).dropna(subset=["y"])
    mask = df["y"].between(35, 45)
    counts = df[mask].groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def one_touch_passes(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count one-touch passes (passes received and immediately passed within 1s) for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with one-touch pass counts.
    """
    events = ctx.team_events.sort_values("timestamp_seconds")
    counts = ctx.players_series(default=0.0)

    for _, possession in events.groupby("possession"):
        possession = possession.reset_index(drop=True)
        for i in range(len(possession) - 1):
            current = possession.loc[i]
            nxt = possession.loc[i + 1]
            if (
                current["team_id"] == ctx.team_id
                and current["type_name"] == "Pass"
                and nxt["team_id"] == ctx.team_id
                and nxt["player_id"] in counts.index
                and nxt["timestamp_seconds"] - current["timestamp_seconds"] <= 1.0
                and nxt["type_name"] != "Carry"
            ):
                counts.loc[nxt["player_id"]] += 1.0
    return counts


def weak_foot_pass_share(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate share of passes with weak foot.

    Since roster metadata is not available, we infer the dominant foot for each player
    based on which foot they use more frequently. The foot used less is considered
    the weak foot.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with weak foot pass share (0.0 to 1.0).
        Returns NaN for players with insufficient pass data to determine dominant foot.
    """
    passes = ctx.player_events[ctx.player_events["type_name"] == "Pass"]
    if passes.empty:
        return ctx.players_series(default=np.nan)
    
    # Get body_part column
    body_part = passes.get("pass.body_part.name")
    if body_part is None:
        body_part = passes.get("pass_body_part_name")
    if body_part is None:
        return ctx.players_series(default=np.nan)
    
    # Calculate weak foot share for each player
    results = {}
    for player_id, player_passes in passes.groupby("player_id"):
        player_body_parts = body_part.loc[player_passes.index]
        
        # Count left and right foot passes
        left_foot = (player_body_parts == "Left Foot").sum()
        right_foot = (player_body_parts == "Right Foot").sum()
        total_foot_passes = left_foot + right_foot
        
        if total_foot_passes == 0:
            results[player_id] = np.nan
        else:
            # Determine weak foot (the one used less)
            if left_foot < right_foot:
                weak_foot_count = left_foot
            elif right_foot < left_foot:
                weak_foot_count = right_foot
            else:
                # Equal usage - can't determine, return NaN
                results[player_id] = np.nan
                continue
            
            # Calculate share
            results[player_id] = weak_foot_count / total_foot_passes
    
    series = pd.Series(results, dtype=float)
    return ctx.ensure_index(series, fill_value=np.nan)


def pressured_retention_rate(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate retention rate when receiving under pressure (delegates to pressured_touch_retention_rate).

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pressured retention rates (0.0 to 1.0).
        Returns NaN for players with no pressured touches.
    """
    from .pressure_resistance import pressured_touch_retention_rate

    return pressured_touch_retention_rate(ctx)

