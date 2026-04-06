from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext, _extract_coordinate


def line_breaking_receipts(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count ball receipts that break from defensive third (x < 40) to midfield (x >= 40).

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with line-breaking receipt counts.
    """
    events = ctx.events.sort_values(["possession", "timestamp"])
    grouped = events.groupby("possession")
    events = events.assign(
        prev_team_id=grouped["team_id"].shift(1),
        prev_player_id=grouped["player_id"].shift(1),
        prev_x=grouped["x"].shift(1),
        prev_type=grouped["type_name"].shift(1),
    )
    receipts = events[
        (events["team_id"] == ctx.team_id)
        & (events["type_name"] == "Ball Receipt*")
        & (events["x"] >= 40)
        & (events["prev_team_id"] == ctx.team_id)
        & (events["prev_x"] < 40)
    ]
    counts = receipts.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def zone14_touches(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count touches in Zone 14 (x 78-102, y 35-55) for each midfielder.

    Zone 14 is the central attacking zone just outside the penalty area.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with Zone 14 touch counts.
    """
    df = ctx.player_events.dropna(subset=["x", "y"])
    mask = df["x"].between(78, 102) & df["y"].between(35, 55)
    counts = df[mask].groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def penalty_area_deliveries(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count passes delivered into the penalty area (x >= 102, y 18-62) for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with penalty area delivery counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "Pass"].copy()
    if df.empty:
        return ctx.players_series(default=0.0)
    df["end_x"] = df["pass_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    df["end_y"] = df["pass_end_location"].apply(lambda loc: _extract_coordinate(loc, 1))
    mask = (df["end_x"] >= 102) & df["end_y"].between(18, 62)
    counts = df[mask].groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def switches_completed(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count completed switch passes for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with switch pass counts.
    """
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Pass")
        & (ctx.player_events.get("pass.switch") == True)
    ]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def cross_accuracy(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate cross accuracy for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with cross accuracy (0.0 to 1.0).
        Returns NaN for players with no cross attempts.
    """
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Pass")
        & (ctx.player_events.get("pass.cross") == True)
    ]
    if df.empty:
        return ctx.players_series(default=np.nan)
    outcome_col = "pass.outcome.name" if "pass.outcome.name" in df.columns else "pass_outcome_name"
    completed = df[df[outcome_col].isna()].groupby("player_id")["type_name"].count().astype(float)
    attempted = df.groupby("player_id")["type_name"].count().astype(float)
    accuracy = completed / attempted.replace(0.0, np.nan)
    return ctx.ensure_index(accuracy, fill_value=np.nan)

