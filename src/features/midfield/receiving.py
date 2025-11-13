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
    Calculate share of passes with weak foot (requires roster metadata, returns NaN).

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with weak foot pass share (0.0 to 1.0).
        Returns NaN for all players as roster metadata is not available.
    """
    return ctx.players_series(default=np.nan)


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

