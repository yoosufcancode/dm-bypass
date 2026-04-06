from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext


def _duels(ctx: MidfieldFeatureContext) -> pd.DataFrame:
    """
    Helper function to filter player events to only duels.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only duel events for midfielders.
    """
    return ctx.player_events[ctx.player_events["type_name"] == "Duel"]


def aerial_duels_contested(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count aerial duels contested by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with aerial duel counts.
    """
    df = _duels(ctx)
    type_series = df.get("duel.type.name")
    if type_series is None:
        return ctx.players_series(default=0.0)
    df = df[type_series.str.contains("Aerial", na=False)]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def aerial_duel_win_rate(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate aerial duel win rate for each midfielder.

    Data limitation: StatsBomb records "Aerial Lost" events for the losing player
    but does NOT record a corresponding "Aerial Won" event for the winner. This means
    win rate cannot be calculated from positive evidence. If duel.outcome.name contains
    "Won" strings in the data, those are used; otherwise 0.0 is returned for all players.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with aerial duel win rates (0.0 to 1.0).
        Returns 0.0 for all players when win data is not available in the dataset.
    """
    df = _duels(ctx)
    type_series = df.get("duel.type.name")
    if type_series is None:
        return ctx.players_series(default=0.0)
    mask = type_series.str.contains("Aerial", na=False)
    df = df[mask]
    if df.empty:
        return ctx.players_series(default=0.0)
    
    # Check if any wins are recorded
    outcome_series = df.get("duel.outcome.name")
    if outcome_series is not None:
        wins = df[outcome_series.str.contains("Won", na=False)]
        if not wins.empty:
            win_counts = wins.groupby("player_id")["type_name"].count().astype(float)
            contested = df.groupby("player_id")["type_name"].count().astype(float)
            rate = win_counts / contested.replace(0.0, np.nan)
            return ctx.ensure_index(rate, fill_value=0.0)
    
    # If no wins recorded (only losses), return 0.0 for all
    return ctx.players_series(default=0.0)


def fifty_fiftys_won(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count 50/50 duels won by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with 50/50 duel win counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "50/50"].copy()
    outcome = df.get("50_50.outcome.name")
    if outcome is None:
        return ctx.players_series(default=0.0)
    mask = outcome.str.contains("Won", na=False)
    df = df[mask]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def sliding_tackles(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count sliding tackles attempted by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with sliding tackle counts.
    """
    # Filter to tackle-type duels (same pattern as tackles_won)
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Duel")
        & (ctx.player_events.get("duel.type.name") == "Tackle")
    ]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def sliding_tackle_success_rate(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate sliding tackle success rate for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with sliding tackle success rates (0.0 to 1.0).
        Returns NaN for players with no sliding tackle attempts.
    """
    # Filter to tackle-type duels (same pattern as tackles_won)
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Duel")
        & (ctx.player_events.get("duel.type.name") == "Tackle")
    ]
    if df.empty:
        return ctx.players_series(default=np.nan)
    
    wins = df[df.get("duel.outcome.name").str.contains("Won", na=False)]
    win_counts = wins.groupby("player_id")["type_name"].count().astype(float)
    attempts = df.groupby("player_id")["type_name"].count().astype(float)
    rate = win_counts / attempts.replace(0.0, np.nan)
    return ctx.ensure_index(rate, fill_value=np.nan)

