"""
Zone-Specific Features

Computes features related to transitions between defensive third, midfield, and final third.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict


def _get_opponent_events(events, team_id):
    """Helper to get opponent events."""
    return events[
        events.get('possession_team.id', pd.Series()) != team_id
    ].copy()


def _get_end_x_from_location(end_loc):
    """Helper to extract x coordinate from location."""
    if isinstance(end_loc, list) and len(end_loc) > 0:
        return end_loc[0]
    return None


def compute_defensive_to_midfield_passes_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of passes from defensive third (< 40) entering midfield (≥ 40)."""
    opponent_events = _get_opponent_events(events, team_id)
    opponent_passes = opponent_events[opponent_events['type_name'] == 'Pass'].copy()
    
    if len(opponent_passes) > 0:
        passes_with_end = opponent_passes[
            opponent_passes.get('pass.end_location', pd.Series()).notna()
        ].copy()
        
        passes_with_end['end_x'] = passes_with_end.get('pass.end_location', pd.Series()).apply(_get_end_x_from_location)
        passes_with_end = passes_with_end[passes_with_end['end_x'].notna()]
        
        defensive_to_midfield_passes = passes_with_end[
            (passes_with_end['x'] < 40) & (passes_with_end['end_x'] >= 40)
        ]
        return len(defensive_to_midfield_passes)
    return 0


def compute_defensive_to_midfield_carries_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of carries from defensive third entering midfield."""
    opponent_events = _get_opponent_events(events, team_id)
    opponent_carries = opponent_events[opponent_events['type_name'] == 'Carry'].copy()
    
    if 'carry.end_location' in opponent_carries.columns:
        carries_with_end = opponent_carries[
            opponent_carries['carry.end_location'].notna()
        ].copy()
        
        if len(carries_with_end) > 0:
            carries_with_end['end_x'] = carries_with_end.get('carry.end_location', pd.Series()).apply(_get_end_x_from_location)
            carries_with_end = carries_with_end[carries_with_end['end_x'].notna()]
            
            defensive_to_midfield_carries = carries_with_end[
                (carries_with_end['x'] < 40) & (carries_with_end['end_x'] >= 40)
            ]
            return len(defensive_to_midfield_carries)
    return 0


def compute_defensive_to_midfield_prevention_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of defensive-to-midfield transitions prevented."""
    passes_allowed = compute_defensive_to_midfield_passes_allowed(events, team_id, midfield_x_min, midfield_x_max)
    carries_allowed = compute_defensive_to_midfield_carries_allowed(events, team_id, midfield_x_min, midfield_x_max)
    total_transition_attempts = passes_allowed + carries_allowed
    
    if total_transition_attempts > 0:
        interceptions_at_entry = events[
            (events['type_name'] == 'Interception') &
            (events.get('team.id', pd.Series()) == team_id) &
            (events['x'] >= 38) & (events['x'] <= 42) &
            events['x'].notna()
        ]
        prevented = len(interceptions_at_entry)
        return prevented / total_transition_attempts
    return 0.0


def compute_defensive_to_midfield_interception_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Interception rate at midfield entry point."""
    passes_allowed = compute_defensive_to_midfield_passes_allowed(events, team_id, midfield_x_min, midfield_x_max)
    carries_allowed = compute_defensive_to_midfield_carries_allowed(events, team_id, midfield_x_min, midfield_x_max)
    total_transition_attempts = passes_allowed + carries_allowed
    
    if total_transition_attempts > 0:
        interceptions_at_entry = events[
            (events['type_name'] == 'Interception') &
            (events.get('team.id', pd.Series()) == team_id) &
            (events['x'] >= 38) & (events['x'] <= 42) &
            events['x'].notna()
        ]
        return len(interceptions_at_entry) / total_transition_attempts
    return 0.0


def compute_midfield_to_final_passes_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of passes from midfield entering final third (> 80)."""
    opponent_events = _get_opponent_events(events, team_id)
    opponent_passes = opponent_events[opponent_events['type_name'] == 'Pass'].copy()
    
    if len(opponent_passes) > 0:
        passes_with_end = opponent_passes[
            opponent_passes.get('pass.end_location', pd.Series()).notna()
        ].copy()
        
        passes_with_end['end_x'] = passes_with_end.get('pass.end_location', pd.Series()).apply(_get_end_x_from_location)
        passes_with_end = passes_with_end[passes_with_end['end_x'].notna()]
        
        midfield_to_final_passes = passes_with_end[
            (passes_with_end['x'] >= 40) & (passes_with_end['x'] <= 80) & 
            (passes_with_end['end_x'] > 80)
        ]
        return len(midfield_to_final_passes)
    return 0


def compute_midfield_to_final_carries_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of carries from midfield entering final third."""
    opponent_events = _get_opponent_events(events, team_id)
    opponent_carries = opponent_events[opponent_events['type_name'] == 'Carry'].copy()
    
    if 'carry.end_location' in opponent_carries.columns:
        carries_with_end = opponent_carries[
            opponent_carries['carry.end_location'].notna()
        ].copy()
        
        if len(carries_with_end) > 0:
            carries_with_end['end_x'] = carries_with_end.get('carry.end_location', pd.Series()).apply(_get_end_x_from_location)
            carries_with_end = carries_with_end[carries_with_end['end_x'].notna()]
            
            midfield_to_final_carries = carries_with_end[
                (carries_with_end['x'] >= 40) & (carries_with_end['x'] <= 80) & 
                (carries_with_end['end_x'] > 80)
            ]
            return len(midfield_to_final_carries)
    return 0


def compute_midfield_to_final_prevention_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of midfield-to-final transitions prevented."""
    passes_allowed = compute_midfield_to_final_passes_allowed(events, team_id, midfield_x_min, midfield_x_max)
    carries_allowed = compute_midfield_to_final_carries_allowed(events, team_id, midfield_x_min, midfield_x_max)
    total_exit_attempts = passes_allowed + carries_allowed
    
    if total_exit_attempts > 0:
        interceptions_at_exit = events[
            (events['type_name'] == 'Interception') &
            (events.get('team.id', pd.Series()) == team_id) &
            (events['x'] >= 78) & (events['x'] <= 82) &
            events['x'].notna()
        ]
        prevented = len(interceptions_at_exit)
        return prevented / total_exit_attempts
    return 0.0


def compute_midfield_to_final_interception_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Interception rate at midfield exit point."""
    passes_allowed = compute_midfield_to_final_passes_allowed(events, team_id, midfield_x_min, midfield_x_max)
    carries_allowed = compute_midfield_to_final_carries_allowed(events, team_id, midfield_x_min, midfield_x_max)
    total_exit_attempts = passes_allowed + carries_allowed
    
    if total_exit_attempts > 0:
        interceptions_at_exit = events[
            (events['type_name'] == 'Interception') &
            (events.get('team.id', pd.Series()) == team_id) &
            (events['x'] >= 78) & (events['x'] <= 82) &
            events['x'].notna()
        ]
        return len(interceptions_at_exit) / total_exit_attempts
    return 0.0


def compute_bypass_attempts_total(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of possessions that bypass midfield (defensive third to final third)."""
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    bypass_attempts = 0
    for poss in opponent_possessions:
        poss_events = events[events['possession'] == poss]
        if len(poss_events) > 0:
            start_x = poss_events.iloc[0].get('x', 0)
            max_x = poss_events['x'].max() if 'x' in poss_events.columns else 0
            
            if start_x < 40 and max_x > 80:
                bypass_attempts += 1
    
    return bypass_attempts


def compute_bypass_prevention_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of bypass attempts prevented."""
    bypass_attempts = compute_bypass_attempts_total(events, team_id, midfield_x_min, midfield_x_max)
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    if len(opponent_possessions) > 0:
        return 1 - (bypass_attempts / len(opponent_possessions))
    return 0.0


def compute_zone_specific_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all zone-specific features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, possession_team.id,
        x, y, timestamp, possession, pass.end_location, carry.end_location
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
    
    features['defensive_to_midfield_passes_allowed'] = compute_defensive_to_midfield_passes_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['defensive_to_midfield_carries_allowed'] = compute_defensive_to_midfield_carries_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['defensive_to_midfield_prevention_rate'] = compute_defensive_to_midfield_prevention_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['defensive_to_midfield_interception_rate'] = compute_defensive_to_midfield_interception_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_to_final_passes_allowed'] = compute_midfield_to_final_passes_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_to_final_carries_allowed'] = compute_midfield_to_final_carries_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_to_final_prevention_rate'] = compute_midfield_to_final_prevention_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_to_final_interception_rate'] = compute_midfield_to_final_interception_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['bypass_attempts_total'] = compute_bypass_attempts_total(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['bypass_prevention_rate'] = compute_bypass_prevention_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
