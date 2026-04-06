from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def _set_piece_mask(events: pd.DataFrame) -> pd.Series:
    """
    Helper function to identify set piece events.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame.

    Returns
    -------
    pd.Series
        Boolean series indicating set piece events.
    """
    return events["play_pattern.name"].isin(
        ["From Corner", "From Free Kick", "From Throw In", "From Goal Kick"]
    )


def set_piece_involvements(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count set piece involvements for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with set piece involvement counts.
    """
    mask = _set_piece_mask(ctx.player_events)
    df = ctx.player_events[mask]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def corner_delivery_accuracy(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate corner delivery accuracy for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with corner delivery accuracy (0.0 to 1.0).
        Returns NaN for players with no corner attempts.
    """
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Pass")
        & (ctx.player_events.get("pass.type.name") == "Corner")
    ]
    if df.empty:
        return ctx.players_series(default=np.nan)
    outcome_col = "pass.outcome.name" if "pass.outcome.name" in df.columns else "pass_outcome_name"
    completed = df[df[outcome_col].isna()].groupby("player_id")["type_name"].count().astype(float)
    attempted = df.groupby("player_id")["type_name"].count().astype(float)
    accuracy = completed / attempted.replace(0.0, np.nan)
    return ctx.ensure_index(accuracy, fill_value=np.nan)


def set_piece_duels_won(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count set piece duels won by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with set piece duel win counts.
    """
    mask = _set_piece_mask(ctx.player_events)
    df = ctx.player_events[
        mask
        & (ctx.player_events["type_name"] == "Duel")
        & (ctx.player_events.get("duel.outcome.name").str.contains("Won", na=False))
    ]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def defensive_set_piece_clearances(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count defensive clearances on set pieces by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with defensive set piece clearance counts.
    """
    mask = _set_piece_mask(ctx.team_events)
    df = ctx.team_events[
        mask & (ctx.team_events["type_name"] == "Clearance")
    ]
    df = df[df["player_id"].isin(ctx.midfielder_ids)]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)

