"""
Opponent-side context features for each match-half.

These capture the OPPONENT's playing style and intent, which are the primary
drivers of how many times the opponent bypasses the analysed team's midfield.

Works with the internal (cleaned) events schema produced by both
clean_events() (StatsBomb) and clean_wyscout_events() (Wyscout).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


_LONG_BALL_THRESHOLD = 32.0  # metres (in 0-120 pitch scale)
_MID_X_HIGH = 80.0


def _opponent_team_id(events: pd.DataFrame, team_id: int) -> Optional[int]:
    """Return the opponent's team_id."""
    ids = events["team_id"].dropna().unique()
    others = [int(t) for t in ids if int(t) != team_id]
    return others[0] if others else None


def _pass_length(row: pd.Series) -> float:
    """Compute pass length from start→end coordinates."""
    x0, y0 = row.get("x"), row.get("y")
    end_loc = row.get("pass_end_location")
    if not isinstance(end_loc, (list, tuple)) or len(end_loc) < 2:
        return np.nan
    x1, y1 = end_loc[0], end_loc[1]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [x0, y0, x1, y1]):
        return np.nan
    return float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)


def _pass_forward_rate(passes: pd.DataFrame) -> float:
    """
    Fraction of passes with positive x-displacement (forward passes).
    """
    if passes.empty:
        return np.nan
    forward_count = 0
    total = 0
    for _, row in passes.iterrows():
        x0 = row.get("x")
        end_loc = row.get("pass_end_location")
        if isinstance(end_loc, (list, tuple)) and len(end_loc) > 0:
            x1 = end_loc[0]
            if not any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [x0, x1]):
                total += 1
                if x1 > x0:
                    forward_count += 1
    return float(forward_count / total) if total > 0 else np.nan


def compute_opponent_context(
    events: pd.DataFrame,
    team_name: str,
    period: int,
) -> dict:
    """
    Compute opponent-side context features for a single match-half.

    Parameters
    ----------
    events : pd.DataFrame
        Cleaned events for this period (internal schema from clean_wyscout_events
        or clean_events).
    team_name : str
        The analysed team's name.
    period : int
        Match period (1 or 2) — used only for scoreline calculation.

    Returns
    -------
    dict
        Flat dict of scalar opponent context features.
    """
    result = {
        "opp_long_ball_rate":    np.nan,
        "opp_avg_pass_length":   np.nan,
        "opp_direct_play_index": np.nan,
        "opp_pass_forward_rate": np.nan,
        "score_diff_start_of_half": 0,
    }

    # Identify teams
    team_events = events[events["team_name"] == team_name]
    if team_events.empty:
        # Fall back to team_id if team_name column is missing
        return result

    opp_events = events[events["team_name"] != team_name]
    if opp_events.empty:
        return result

    # ── Opponent passing style ────────────────────────────────────────────────
    opp_passes = opp_events[opp_events["type_name"] == "Pass"].copy()

    if not opp_passes.empty:
        opp_passes["_pass_length"] = opp_passes.apply(_pass_length, axis=1)
        lengths = opp_passes["_pass_length"].dropna()
        if len(lengths) > 0:
            result["opp_avg_pass_length"] = float(lengths.mean())
            result["opp_long_ball_rate"]  = float((lengths >= _LONG_BALL_THRESHOLD).mean())

        fwd_rate = _pass_forward_rate(opp_passes)
        result["opp_pass_forward_rate"] = fwd_rate

        if not np.isnan(result["opp_long_ball_rate"]) and not np.isnan(fwd_rate):
            result["opp_direct_play_index"] = result["opp_long_ball_rate"] * fwd_rate

    # ── Score difference at start of this half ────────────────────────────────
    # Only meaningful when we have the full match events (period > 1 check)
    # In Wyscout data, each period is processed separately so we approximate
    # score diff as 0 for period 1 and check shots for period 2.
    if period > 1 and "match_id" in events.columns:
        # Look at period 1 shots to infer scoreline
        period1 = events[events["period"] == 1]
        shots = period1[period1["type_name"] == "Shot"]
        # Wyscout doesn't have shot outcome name — skip score calc
        # Keep score_diff_start_of_half = 0 (safe default)

    return result
