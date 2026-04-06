from __future__ import annotations

import pandas as pd

from .context import MidfieldFeatureContext


def fouls_committed(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count fouls committed by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with foul counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "Foul Committed"]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def fouls_suffered(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count fouls suffered (fouls won) by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with fouls won counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "Foul Won"]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def tactical_fouls(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count tactical/professional fouls committed by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with tactical foul counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "Foul Committed"]
    if df.empty:
        return ctx.players_series(default=0.0)
    
    # Check for yellow cards (tactical fouls often result in yellow cards)
    card_name = df.get("foul_committed.card.name")
    if card_name is None:
        # Try alternative column name
        card_name = df.get("foul_committed_card_name")
    
    if card_name is not None:
        # Count fouls with yellow cards as tactical fouls
        tactical = df[card_name == "Yellow Card"]
        counts = tactical.groupby("player_id")["type_name"].count().astype(float)
    else:
        # Also check if foul_committed.type.name exists (for newer data)
        foul_type = df.get("foul_committed.type.name")
        if foul_type is not None:
            tactical = df[foul_type.isin(["Tactical", "Professional Foul"])]
            counts = tactical.groupby("player_id")["type_name"].count().astype(float)
        else:
            counts = pd.Series(dtype=float)
    
    return ctx.ensure_index(counts, fill_value=0.0)


def advantage_fouls_won(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count fouls won where advantage was played by each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with advantage foul counts.
    """
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Foul Won")
        & (ctx.player_events.get("foul_won.advantage") == True)
    ]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)

