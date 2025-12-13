from __future__ import annotations

import numpy as np
import pandas as pd

from .context import MidfieldFeatureContext, _extract_coordinate


def _carries(ctx: MidfieldFeatureContext) -> pd.DataFrame:
    """
    Helper function to filter player events to only carries.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only carry events for midfielders.
    """
    return ctx.player_events[ctx.player_events["type_name"] == "Carry"]


def carries_attempted(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count total carries attempted by each midfielder.
    
    Filters out very short carries (< 10 meters) as StatsBomb includes
    many trivial movements that don't represent meaningful ball progression.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with carry attempt counts.
    """
    df = _carries(ctx).copy()
    if df.empty:
        return ctx.players_series(default=0.0)
    
    # Calculate carry distances and filter out very short ones (< 10 meters)
    df["start_x"] = df["x"]
    df["start_y"] = df["y"]
    df["end_x"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    df["end_y"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 1))
    
    # Calculate distance in meters (pitch is ~105m x 68m, coordinates are 0-120 x 0-80)
    df["distance"] = ((df["end_x"] - df["start_x"])**2 + (df["end_y"] - df["start_y"])**2)**0.5
    
    # Filter out carries less than 10 meters
    meaningful_carries = df[df["distance"] >= 10.0]
    
    counts = meaningful_carries.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def progressive_carries(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count progressive carries: carries that advance the ball at least 10 meters forward.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with progressive carry counts.
    """
    df = _carries(ctx).copy()
    if df.empty:
        return ctx.players_series(default=0.0)
    df["end_x"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    progressive = df[(df["end_x"] - df["x"]) >= 10]
    counts = progressive.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def carry_distance_total(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate total distance covered by all carries for each midfielder.
    
    Filters out very short carries (< 10 meters) as StatsBomb includes
    many trivial movements that don't represent meaningful ball progression.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with total carry distance in meters.
    """
    df = _carries(ctx).copy()
    if df.empty:
        return ctx.players_series(default=0.0)

    # Calculate distances and filter out very short carries (< 10 meters)
    df["start_x"] = df["x"]
    df["start_y"] = df["y"]
    df["end_x"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    df["end_y"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 1))
    
    # Calculate distance in meters
    df["distance"] = ((df["end_x"] - df["start_x"])**2 + (df["end_y"] - df["start_y"])**2)**0.5
    
    # Filter out carries less than 10 meters
    meaningful_carries = df[df["distance"] >= 10.0]
    
    totals = meaningful_carries.groupby("player_id")["distance"].sum()
    return ctx.ensure_index(totals, fill_value=0.0)


def successful_dribbles(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count successful dribbles (Dribble events with Complete outcome) for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with successful dribble counts.
    """
    df = ctx.player_events[ctx.player_events["type_name"] == "Dribble"]
    if df.empty:
        return ctx.players_series(default=0.0)
    # Check for dribble.outcome.name == "Complete"
    dribble_outcome = df.get("dribble.outcome.name")
    if dribble_outcome is None:
        # Try alternative column name
        dribble_outcome = df.get("dribble_outcome_name")
    if dribble_outcome is not None:
        successful = df[dribble_outcome == "Complete"]
        counts = successful.groupby("player_id")["type_name"].count().astype(float)
    else:
        # If outcome column doesn't exist, return zeros
        counts = pd.Series(dtype=float)
    return ctx.ensure_index(counts, fill_value=0.0)


def carries_leading_to_shot(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count carries that directly lead to a shot within the same possession.

    Since StatsBomb data may not include carry_id references, we use possession
    and timing: a carry leads to a shot if a shot occurs in the same possession
    within 5 seconds after the carry ends.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with carry-to-shot counts.
    """
    df = _carries(ctx)
    if df.empty:
        return ctx.players_series(default=0.0)
    
    # Get shots in the same team
    shots = ctx.team_events[ctx.team_events["type_name"] == "Shot"].copy()
    if shots.empty:
        return ctx.players_series(default=0.0)
    
    # Sort by timestamp
    df = df.sort_values("timestamp_seconds")
    shots = shots.sort_values("timestamp_seconds")
    
    counts = {}
    for player_id, player_carries in df.groupby("player_id"):
        count = 0
        for _, carry_row in player_carries.iterrows():
            carry_time = carry_row["timestamp_seconds"]
            possession = carry_row["possession"]
            
            # Find shots in same possession within 5 seconds
            matching_shots = shots[
                (shots["possession"] == possession)
                & (shots["timestamp_seconds"] > carry_time)
                & (shots["timestamp_seconds"] <= carry_time + 5)
            ]
            if not matching_shots.empty:
                count += 1
        counts[player_id] = float(count)
    
    series = pd.Series(counts, dtype=float)
    return ctx.ensure_index(series, fill_value=0.0)


def carries_leading_to_key_pass(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count carries that directly lead to a key pass within the same possession.

    Since StatsBomb data may not include carry_id references, we use possession
    and timing: a carry leads to a key pass if a key pass occurs in the same
    possession within 3 seconds after the carry ends.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with carry-to-key-pass counts.
    """
    df = _carries(ctx)
    if df.empty:
        return ctx.players_series(default=0.0)
    
    # Get key passes (passes with shot_assist or goal_assist)
    key_passes = ctx.team_events[
        (ctx.team_events["type_name"] == "Pass")
        & (
            (ctx.team_events.get("pass.shot_assist") == True)
            | (ctx.team_events.get("pass.goal_assist") == True)
        )
    ].copy()
    if key_passes.empty:
        return ctx.players_series(default=0.0)
    
    # Sort by timestamp
    df = df.sort_values("timestamp_seconds")
    key_passes = key_passes.sort_values("timestamp_seconds")
    
    counts = {}
    for player_id, player_carries in df.groupby("player_id"):
        count = 0
        for _, carry_row in player_carries.iterrows():
            carry_time = carry_row["timestamp_seconds"]
            possession = carry_row["possession"]
            
            # Find key passes in same possession within 3 seconds
            matching_passes = key_passes[
                (key_passes["possession"] == possession)
                & (key_passes["timestamp_seconds"] > carry_time)
                & (key_passes["timestamp_seconds"] <= carry_time + 3)
            ]
            if not matching_passes.empty:
                count += 1
        counts[player_id] = float(count)
    
    series = pd.Series(counts, dtype=float)
    return ctx.ensure_index(series, fill_value=0.0)


def final_third_carries(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count carries that end in the final third (x > 80).

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with final third carry counts.
    """
    df = _carries(ctx).copy()
    if df.empty:
        return ctx.players_series(default=0.0)
    
    # Calculate carry distances and filter out very short ones (< 10 meters)
    df["start_x"] = df["x"]
    df["start_y"] = df["y"]
    df["end_x"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    df["end_y"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 1))
    
    # Calculate distance in meters
    df["distance"] = ((df["end_x"] - df["start_x"])**2 + (df["end_y"] - df["start_y"])**2)**0.5
    
    # Filter: must be >= 10m AND end in final third (x >= 80)
    mask = (df["distance"] >= 10.0) & (df["end_x"] >= 80)
    counts = df[mask].groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def penalty_area_carries(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count carries that end in the penalty area (x >= 102, y between 18-62).

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with penalty area carry counts.
    """
    df = _carries(ctx).copy()
    if df.empty:
        return ctx.players_series(default=0.0)
    df["end_x"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    df["end_y"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 1))
    mask = (df["end_x"] >= 102) & df["end_y"].between(18, 62)
    counts = df[mask].groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def pressured_carry_success_rate(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Calculate success rate of carries attempted while under pressure.

    A carry is considered successful if it is not immediately followed by
    a dispossession or miscontrol within 2 seconds.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with pressured carry success rates (0.0 to 1.0).
        Returns NaN for players with no pressured carries.
    """
    df = _carries(ctx).copy()
    if df.empty:
        return ctx.players_series(default=np.nan)
    pressured = df[df["under_pressure"] == True]
    if pressured.empty:
        return ctx.players_series(default=np.nan)
    pressured = pressured.sort_values(["player_id", "timestamp_seconds"])
    next_events = ctx.team_events.sort_values("timestamp_seconds")

    success_counts = {}
    for player_id, group in pressured.groupby("player_id"):
        success = 0
        for _, row in group.iterrows():
            t_end = row["timestamp_seconds"]
            window = next_events[
                (next_events["timestamp_seconds"] > t_end)
                & (next_events["timestamp_seconds"] <= t_end + 2)
            ]
            player_window = window[window["player_id"] == player_id]
            if player_window.empty:
                success += 1
            else:
                next_types = player_window["type_name"].tolist()
                if not any(evt in ["Dispossessed", "Miscontrol"] for evt in next_types):
                    success += 1
        total = len(group)
        success_counts[player_id] = success / total if total else np.nan

    series = pd.Series(success_counts, dtype=float)
    return ctx.ensure_index(series, fill_value=np.nan)

