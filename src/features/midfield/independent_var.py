"""
Calculate the number of bypasses per match (independent variable).

A bypass occurs when an opponent possession reaches the team's final third
(x >= 80) within 10 seconds and within 4 passes from the start of the possession.

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
DEFAULT_FINAL_THIRD_X = 80  # opponent entering defensive final third if x >= 80


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
                    "final_third_x": config.get("pitch", {}).get("final_third_x", DEFAULT_FINAL_THIRD_X)
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
        
        # Get start time of possession
        start_time = poss_events["timestamp_seconds"].min()
        
        # Get events within time window
        window_events = poss_events[
            poss_events["timestamp_seconds"] <= start_time + time_window
        ].copy()
        
        if window_events.empty:
            continue
        
        # Sort by timestamp
        window_events = window_events.sort_values("timestamp_seconds")
        
        # Count passes cumulatively
        window_events["pass_count"] = window_events["is_pass"].cumsum()
        
        # Extract x coordinates from location and pass_end_location
        window_events["x_coord"] = window_events.get("x", np.nan)
        
        # Extract pass end x coordinate
        if "pass_end_location" in window_events.columns:
            window_events["pass_end_x"] = window_events["pass_end_location"].apply(
                lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else np.nan
            )
        else:
            window_events["pass_end_x"] = np.nan
        
        # Check if final third reached (either by event location or pass end location)
        reached_final_third = window_events[
            (window_events["x_coord"] >= final_third_x) | 
            (window_events["pass_end_x"] >= final_third_x)
        ]
        
        if not reached_final_third.empty:
            # Check if reached within max_passes
            first_reach = reached_final_third.iloc[0]
            pass_count_at_reach = window_events[
                window_events["timestamp_seconds"] <= first_reach["timestamp_seconds"]
            ]["pass_count"].max()
            
            if pass_count_at_reach <= max_passes:
                bypass_count += 1
    
    return bypass_count
