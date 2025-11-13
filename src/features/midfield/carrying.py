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

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with carry attempt counts.
    """
    df = _carries(ctx)
    counts = df.groupby("player_id")["type_name"].count().astype(float)
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

    def _distance(row):
        end_loc = row.get("carry_end_location")
        if isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2:
            dx = end_loc[0] - row["x"]
            dy = end_loc[1] - row["y"]
            return float(np.sqrt(dx * dx + dy * dy))
        return 0.0

    df["distance"] = df.apply(_distance, axis=1)
    totals = df.groupby("player_id")["distance"].sum()
    return ctx.ensure_index(totals, fill_value=0.0)


def successful_dribbles(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count successful dribbles (Take On events won) for each midfielder.

    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing player events and midfielder IDs.

    Returns
    -------
    pd.Series
        Series indexed by player_id with successful dribble counts.
    """
    df = ctx.player_events[
        (ctx.player_events["type_name"] == "Take On")
        & (ctx.player_events.get("take_on.outcome.name") == "Won")
    ]
    counts = df.groupby("player_id")["type_name"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def carries_leading_to_shot(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count carries that directly lead to a shot within the same possession.

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
    if df.empty or "carry.id" not in df.columns:
        return ctx.players_series(default=0.0)
    carry_ids = df[["player_id", "carry.id"]].dropna()
    shot_df = ctx.team_events[ctx.team_events["type_name"] == "Shot"]
    if "shot.carry_id" not in shot_df.columns:
        return ctx.players_series(default=0.0)
    merged = carry_ids.merge(
        shot_df[["shot.carry_id"]],
        left_on="carry.id",
        right_on="shot.carry_id",
        how="inner",
    )
    counts = merged.groupby("player_id")["carry.id"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


def carries_leading_to_key_pass(ctx: MidfieldFeatureContext) -> pd.Series:
    """
    Count carries that directly lead to a key pass within the same possession.

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
    if df.empty or "carry.id" not in df.columns:
        return ctx.players_series(default=0.0)
    carry_ids = df[["player_id", "carry.id"]].dropna()
    pass_df = ctx.team_events[
        (ctx.team_events["type_name"] == "Pass")
        & (ctx.team_events.get("pass.shot_assist") == True)
    ]
    if "pass.carry_id" not in pass_df.columns:
        return ctx.players_series(default=0.0)
    merged = carry_ids.merge(
        pass_df[["pass.carry_id"]],
        left_on="carry.id",
        right_on="pass.carry_id",
        how="inner",
    )
    counts = merged.groupby("player_id")["carry.id"].count().astype(float)
    return ctx.ensure_index(counts, fill_value=0.0)


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
    df["end_x"] = df["carry_end_location"].apply(lambda loc: _extract_coordinate(loc, 0))
    mask = df["end_x"] > 80
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

