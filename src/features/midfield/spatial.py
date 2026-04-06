from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def _base_series(ctx: MidfieldFeatureContext, column: str, agg: str) -> pd.Series:
    """
    Helper function to compute aggregated series for a column.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.
    column : str
        Column name to aggregate.
    agg : str
        Aggregation type: 'mean', 'var', or 'count'.

    Returns
    -------
    pd.Series
        Series indexed by player_id with aggregated values.
    """
    df = ctx.player_events.dropna(subset=["player_id", column])
    grouped = df.groupby("player_id")[column]
    if agg == "mean":
        series = grouped.mean()
    elif agg == "var":
        series = grouped.var()
    elif agg == "count":
        series = grouped.count()
    else:
        raise ValueError(f"Unsupported aggregation {agg}")
    return ctx.ensure_index(series.astype(float), fill_value=np.nan if agg != "count" else 0.0)


def average_position_x(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate average x-coordinate (lengthwise position) for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with average x-coordinates.
    """
    return _base_series(ctx, "x", "mean")


def average_position_y(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate average y-coordinate (widthwise position) for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with average y-coordinates.
    """
    return _base_series(ctx, "y", "mean")


def width_variance(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate variance in y-coordinate (width variance) for each midfielder.

    Higher values indicate more lateral movement across the pitch width.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with y-coordinate variance.
    """
    return _base_series(ctx, "y", "var")


def zone_entries(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count events in the central zone (y between 35-45) for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with central zone event counts.
    """
    df = ctx.player_events.dropna(subset=["player_id", "y"])
    mask = df["y"].between(35, 45)
    counts = df[mask].groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)

