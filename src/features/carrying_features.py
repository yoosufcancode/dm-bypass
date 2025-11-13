"""
Carrying Through Midfield Features

Computes features related to preventing and controlling opponent carries through midfield.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max):
    """Helper to get opponent carries in midfield."""
    return events[
        (events['type_name'] == 'Carry') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ].copy()


def _get_carries_with_end_location(opponent_carries_midfield):
    """Helper to get carries with end location and extract coordinates."""
    if 'carry.end_location' not in opponent_carries_midfield.columns:
        return pd.DataFrame()
    
    carries_with_end = opponent_carries_midfield[
        opponent_carries_midfield['carry.end_location'].notna()
    ].copy()
    
    if len(carries_with_end) > 0:
        def get_end_x(end_loc):
            if isinstance(end_loc, list) and len(end_loc) > 0:
                return end_loc[0]
            return None
        
        def get_end_y(end_loc):
            if isinstance(end_loc, list) and len(end_loc) > 1:
                return end_loc[1]
            return None
        
        carries_with_end['end_x'] = carries_with_end.get('carry.end_location', pd.Series()).apply(get_end_x)
        carries_with_end['end_y'] = carries_with_end.get('carry.end_location', pd.Series()).apply(get_end_y)
        carries_with_end = carries_with_end[
            carries_with_end['end_x'].notna() & carries_with_end['end_y'].notna()
        ]
        
        carries_with_end['carry_distance'] = np.sqrt(
            (carries_with_end['end_x'] - carries_with_end['x'])**2 +
            (carries_with_end['end_y'] - carries_with_end['y'])**2
        )
    
    return carries_with_end


def compute_carries_allowed_midfield_total(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Total opponent carries in midfield."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    return len(opponent_carries_midfield)


def compute_carries_interrupted_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of carries interrupted by defensive actions within 2 seconds."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(opponent_carries_midfield) > 0:
        interrupted_count = 0
        for idx, carry in opponent_carries_midfield.iterrows():
            carry_time = carry.get('timestamp')
            if pd.notna(carry_time):
                time_window = pd.Timedelta(seconds=2)
                defensive_actions = events[
                    (events.get('team.id', pd.Series()) == team_id) &
                    (events['timestamp'] >= carry_time) &
                    (events['timestamp'] <= carry_time + time_window) &
                    (events['type_name'].isin(['Interception', 'Duel', 'Ball Recovery', 'Block', 'Clearance']))
                ]
                if len(defensive_actions) > 0:
                    interrupted_count += 1
        return interrupted_count
    return 0


def compute_carry_distance_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average distance of carries allowed."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'carry_distance' in carries_with_end.columns:
        return carries_with_end['carry_distance'].mean()
    return None


def compute_carries_entering_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of carries entering midfield from defensive third."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'end_x' in carries_with_end.columns:
        entering_midfield = carries_with_end[
            (carries_with_end['x'] < 40) & (carries_with_end['end_x'] >= 40)
        ]
        return len(entering_midfield)
    return 0


def compute_carries_exiting_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of carries exiting midfield to final third."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'end_x' in carries_with_end.columns:
        exiting_midfield = carries_with_end[
            (carries_with_end['x'] <= 80) & (carries_with_end['end_x'] > 80)
        ]
        return len(exiting_midfield)
    return 0


def compute_carries_through_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of carries that traverse entire midfield zone."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'end_x' in carries_with_end.columns:
        through_midfield = carries_with_end[
            (carries_with_end['x'] < 40) & (carries_with_end['end_x'] > 80)
        ]
        return len(through_midfield)
    return 0


def compute_carry_progression_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average forward progression of carries (end_x - start_x)."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'end_x' in carries_with_end.columns:
        forward_progression = carries_with_end['end_x'] - carries_with_end['x']
        return forward_progression.mean()
    return None


def compute_carries_forward_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of forward carries (end_x > start_x)."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'end_x' in carries_with_end.columns:
        forward_carries = carries_with_end[carries_with_end['end_x'] > carries_with_end['x']]
        return len(forward_carries)
    return 0


def compute_carries_lateral_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of lateral carries (more y movement than x movement)."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'end_x' in carries_with_end.columns and 'end_y' in carries_with_end.columns:
        lateral_carries = carries_with_end[
            (carries_with_end['end_y'] - carries_with_end['y']).abs() > 
            (carries_with_end['end_x'] - carries_with_end['x']).abs()
        ]
        return len(lateral_carries)
    return 0


def compute_carries_central_lane(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of carries through central lane (35 ≤ y ≤ 45)."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'y' in carries_with_end.columns:
        central_lane = carries_with_end[
            (carries_with_end['y'] >= 35) & (carries_with_end['y'] <= 45)
        ]
        return len(central_lane)
    return 0


def compute_carries_wide_areas(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of carries through wide areas (y < 35 or y > 45)."""
    opponent_carries_midfield = _get_opponent_carries_midfield(events, team_id, midfield_x_min, midfield_x_max)
    carries_with_end = _get_carries_with_end_location(opponent_carries_midfield)
    
    if len(carries_with_end) > 0 and 'y' in carries_with_end.columns:
        wide_areas = carries_with_end[
            (carries_with_end['y'] < 35) | (carries_with_end['y'] > 45)
        ]
        return len(wide_areas)
    return 0


def compute_carrying_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all carrying through midfield features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, possession_team.id,
        x, y, timestamp, possession, carry.end_location
    team_id : int
        Team ID for Barcelona (default: 217)
    midfield_x_min : float
        Minimum x-coordinate for midfield (default: 40.0)
    midfield_x_max : float
        Maximum x-coordinate for midfield (default: 80.0)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of feature names and values
    """
    features = {}
    
    features['carries_allowed_midfield_total'] = compute_carries_allowed_midfield_total(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carries_interrupted_midfield'] = compute_carries_interrupted_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carry_distance_allowed'] = compute_carry_distance_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carries_entering_midfield'] = compute_carries_entering_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carries_exiting_midfield'] = compute_carries_exiting_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carries_through_midfield'] = compute_carries_through_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carry_progression_allowed'] = compute_carry_progression_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carries_forward_allowed'] = compute_carries_forward_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carries_lateral_allowed'] = compute_carries_lateral_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carries_central_lane'] = compute_carries_central_lane(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['carries_wide_areas'] = compute_carries_wide_areas(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
