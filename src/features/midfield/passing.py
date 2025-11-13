from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext, _extract_coordinate


def _passes(ctx: MidfieldFeatureContext) -> pd.DataFrame:
    """
    Helper function to filter player events to only passes.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only pass events for midfielders.
    """
    return ctx.player_events[ctx.player_events["type_name"] == "Pass"]


def passes_attempted(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count total passes attempted by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pass attempt counts.
    """
    df = _passes(ctx)
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def pass_completion_rate(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate pass completion rate (completed / attempted) for each midfielder.

    A pass is considered completed if it has no outcome (outcome is NaN).

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with completion rates (0.0 to 1.0).
        Returns NaN for players with no pass attempts.
    """
    df = _passes(ctx)
    if df.empty:
        return ctx.players_series(default=np.nan)
    outcome_col = "pass.outcome.name" if "pass.outcome.name" in df.columns else "pass_outcome_name"
    completed_mask = df[outcome_col].isna()
    completed = df[completed_mask].groupby("player_id")["type_name"].count().astype(float)
    attempted = passes_attempted(ctx)
    rate = completed / attempted.replace(0.0, np.nan)
    return ctx.ensure_index(rate, fill_value=np.nan)


def progressive_passes(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count progressive passes: passes that advance the ball at least 10 meters forward.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with progressive pass counts.
    """
    df = _passes(ctx).copy()
    if df.empty:
        return ctx.players_series(default=0.0)
    df["end_x"] = df["pass_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    df["start_x"] = df["x"]
    progressive = df[(df["end_x"] - df["start_x"]) >= 10]
    counts = progressive.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def final_third_entries_by_pass(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count passes that end in the final third (x > 80).

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with final third entry pass counts.
    """
    df = _passes(ctx).copy()
    if df.empty:
        return ctx.players_series(default=0.0)
    df["end_x"] = df["pass_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    mask = df["end_x"] > 80
    counts = df[mask].groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def key_passes(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count key passes: passes that lead to a shot assist or goal assist.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with key pass counts.
    """
    df = _passes(ctx)
    if df.empty:
        return ctx.players_series(default=0.0)
    mask = (df.get("pass.shot_assist", False) == True) | (df.get("pass.goal_assist", False) == True)
    counts = df[mask].groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def under_pressure_pass_share(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate the share of passes attempted while under pressure.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pressured pass share (0.0 to 1.0).
        Returns NaN for players with no pass attempts.
    """
    df = _passes(ctx)
    if df.empty:
        return ctx.players_series(default=np.nan)
    pressured = df[df["under_pressure"] == True].groupby("player_id")["type_name"].count().astype(float)
    attempted = passes_attempted(ctx)
    share = pressured / attempted.replace(0.0, np.nan)
    return ctx.ensure_index(share, fill_value=np.nan)

