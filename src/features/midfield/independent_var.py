"""
Calculate the number of bypasses per match (independent variable).

A bypass occurs when ALL three conditions are met:
  1. Opponent possession STARTS in their defensive zone (x < 40) — true transition from deep.
  2. The ball reaches the attacking team's final third (x >= 80) within the time
     window (default 10 s) and within max_passes passes.
  3. No ball-control events (Pass, Carry, Ball Receipt*) occur in the midfield
     zone (x = 40–80) during the possession window — the zone was truly SKIPPED,
     not played through.

This module provides a function to calculate bypasses that integrates with
the feature engineering pipeline in main_feature.py.
"""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import yaml

from src.features.midfield.context import MidfieldFeatureContext


# Default configuration values
DEFAULT_TIME_WINDOW = 10  # seconds
DEFAULT_MAX_PASSES = 4
DEFAULT_FINAL_THIRD_X = 80   # target zone: opponent reaches x >= 80
DEFAULT_DEFENSIVE_ZONE_X = 40  # possession must START below x=40 (deep origin)
DEFAULT_MIDFIELD_X_LOW = 40    # midfield zone lower bound
DEFAULT_MIDFIELD_X_HIGH = 80   # midfield zone upper bound

# Event types that constitute "ball control" in the midfield zone.
# "Ball Receipt*" removed: not available in Wyscout data.
MIDFIELD_CONTROL_TYPES = {"Pass", "Carry"}


def load_bypass_config(config_path: Optional[Path] = None) -> dict:
    """
    Load bypass configuration from YAML file or use defaults.
    
    Parameters
    ----------
    config_path : Optional[Path]
        Path to config file (default: config/labels.yaml).
    
    Returns
    -------
    dict
        Configuration dictionary with bypass parameters.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "labels.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Extract bypass config or use defaults
            if config and "bypass" in config:
                return {
                    "time_seconds": config["bypass"].get("time_seconds", DEFAULT_TIME_WINDOW),
                    "max_passes": config["bypass"].get("max_passes", DEFAULT_MAX_PASSES),
                    "final_third_x": config["pitch"].get("final_third_x", DEFAULT_FINAL_THIRD_X)
                }
        
    # Return defaults
    return {
        "time_seconds": DEFAULT_TIME_WINDOW,
        "max_passes": DEFAULT_MAX_PASSES,
        "final_third_x": DEFAULT_FINAL_THIRD_X
    }


def calculate_bypasses_per_match(
    ctx: MidfieldFeatureContext,
    config: Optional[dict] = None
) -> int:
    """
    Calculate the number of bypasses in a match.
    
    A bypass occurs when an opponent possession reaches the final third
    (x >= final_third_x) within the time window (time_seconds) and within
    max_passes from the start of the possession.
    
    This function works with the MidfieldFeatureContext used in the feature
    engineering pipeline, making it compatible with main_feature.py.
    
    Parameters
    ----------
    ctx : MidfieldFeatureContext
        Context containing events and team information for the match.
    config : Optional[dict]
        Configuration dictionary with bypass parameters. If None, loads from
        config file or uses defaults.
        Expected keys: 'time_seconds', 'max_passes', 'final_third_x'
    
    Returns
    -------
    int
        Number of bypasses in the match.
    """
    if config is None:
        config = load_bypass_config()

    time_window = config.get("time_seconds", DEFAULT_TIME_WINDOW)
    max_passes = config.get("max_passes", DEFAULT_MAX_PASSES)
    final_third_x = config.get("final_third_x", DEFAULT_FINAL_THIRD_X)
    defensive_zone_x = config.get("defensive_zone_x", DEFAULT_DEFENSIVE_ZONE_X)
    midfield_x_low = config.get("midfield_x_low", DEFAULT_MIDFIELD_X_LOW)
    midfield_x_high = config.get("midfield_x_high", DEFAULT_MIDFIELD_X_HIGH)
    
    events = ctx.events.copy()
    
    if events.empty:
        return 0
    
    # Convert timestamp to seconds for easier calculation
    if "timestamp" in events.columns and events["timestamp"].dtype.name.startswith("timedelta"):
        # timestamp is a timedelta, convert to total seconds
        events["timestamp_seconds"] = events["timestamp"].dt.total_seconds()
        # Adjust for period (period 1 = 0-2700s, period 2 = 2700-5400s, etc.)
        events["timestamp_seconds"] = (
            (events["period"] - 1) * 45 * 60 + 
            events["timestamp_seconds"]
        )
    elif "minute" in events.columns and "second" in events.columns:
        # Fallback: use minute and second if timestamp not available
        events["timestamp_seconds"] = (
            (events["period"] - 1) * 45 * 60 +
            events["minute"] * 60 +
            events["second"]
        )
    else:
        # If no time information available, return 0
        return 0
    
    # Sort events by timestamp
    events = events.sort_values(["period", "timestamp_seconds"]).copy()
    
    # Identify passes
    events["is_pass"] = (events["type_name"] == "Pass").astype(int)
    
    # Get opponent possessions (not the team we're analyzing)
    opponent_possessions = events[
        (events["possession_team_id"] != ctx.team_id) & 
        (events["possession_team_id"].notna())
    ].copy()
    
    if opponent_possessions.empty:
        return 0
    
    bypass_count = 0

    # Group by possession
    for (poss_id, poss_team_id), poss_events in opponent_possessions.groupby(
        ["possession", "possession_team_id"]
    ):
        if poss_events.empty:
            continue

        poss_events = poss_events.sort_values("timestamp_seconds")
        start_time = poss_events["timestamp_seconds"].min()

        # ── Condition 1: possession must start in the defensive zone (x < 40) ──
        # Use the x-coordinate of the very first event in the possession.
        first_x = poss_events["x"].iloc[0] if "x" in poss_events.columns else np.nan
        if pd.isna(first_x) or first_x >= defensive_zone_x:
            continue  # possession started in midfield or final third — not a deep bypass

        # Get events within the time window
        window_events = poss_events[
            poss_events["timestamp_seconds"] <= start_time + time_window
        ].copy()

        if window_events.empty:
            continue

        # Count passes cumulatively
        window_events["pass_count"] = window_events["is_pass"].cumsum()

        # Extract x coordinates
        window_events["x_coord"] = window_events.get("x", np.nan)

        if "pass_end_location" in window_events.columns:
            window_events["pass_end_x"] = window_events["pass_end_location"].apply(
                lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else np.nan
            )
        else:
            window_events["pass_end_x"] = np.nan

        # ── Condition 2: final third reached within time window and max passes ──
        reached_final_third = window_events[
            (window_events["x_coord"] >= final_third_x) |
            (window_events["pass_end_x"] >= final_third_x)
        ]

        if reached_final_third.empty:
            continue

        first_reach = reached_final_third.iloc[0]
        pass_count_at_reach = window_events[
            window_events["timestamp_seconds"] <= first_reach["timestamp_seconds"]
        ]["pass_count"].max()

        if pass_count_at_reach > max_passes:
            continue

        # ── Condition 3: midfield zone was SKIPPED — no ball control in x=40–80 ──
        # Consider only events BEFORE the ball reached the final third.
        pre_reach = window_events[
            window_events["timestamp_seconds"] < first_reach["timestamp_seconds"]
        ]
        midfield_control = pre_reach[
            pre_reach["type_name"].isin(MIDFIELD_CONTROL_TYPES)
            & (
                pre_reach["x_coord"].between(midfield_x_low, midfield_x_high, inclusive="both")
                | pre_reach["pass_end_x"].between(midfield_x_low, midfield_x_high, inclusive="both")
            )
        ]

        if not midfield_control.empty:
            continue  # ball was played through the midfield — not a true bypass

        bypass_count += 1

    return bypass_count
