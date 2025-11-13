"""
Passing Through Midfield Features

Computes features related to preventing and controlling opponent passes through midfield.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max):
    """Helper to get opponent passes in midfield."""
    return events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ].copy()


def _get_passes_with_end_location(opponent_passes_midfield):
    """Helper to get passes with end location and extract coordinates."""
    passes_with_end = opponent_passes_midfield[
        opponent_passes_midfield.get('pass.end_location', pd.Series()).notna()
    ].copy()
    
    if len(passes_with_end) > 0:
        def get_end_x(end_loc):
            if isinstance(end_loc, list) and len(end_loc) > 0:
                return end_loc[0]
            return None
        
        def get_end_y(end_loc):
            if isinstance(end_loc, list) and len(end_loc) > 1:
                return end_loc[1]
            return None
        
        passes_with_end['end_x'] = passes_with_end.get('pass.end_location', pd.Series()).apply(get_end_x)
        passes_with_end = passes_with_end[passes_with_end['end_x'].notna()]
        
        passes_with_end['end_y'] = passes_with_end.get('pass.end_location', pd.Series()).apply(get_end_y)
        passes_with_end = passes_with_end[passes_with_end['end_y'].notna()]
    
    return passes_with_end


def compute_passes_allowed_midfield_total(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Total opponent passes in midfield."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    return len(opponent_passes_midfield)


def compute_passes_intercepted_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of incomplete/blocked passes (intercepted)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    incomplete_passes = opponent_passes_midfield[
        opponent_passes_midfield.get('outcome_name', pd.Series()).isin(['Incomplete', 'Out', 'Blocked'])
    ]
    return len(incomplete_passes)


def compute_pass_completion_rate_allowed_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Pass completion rate allowed (proportion of completed passes)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(opponent_passes_midfield) > 0:
        completed_passes = opponent_passes_midfield[
            opponent_passes_midfield.get('outcome_name', pd.Series()) == 'Complete'
        ]
        return len(completed_passes) / len(opponent_passes_midfield)
    return None


def compute_passes_allowed_forward(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of forward passes allowed (end_x > start_x)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    passes_with_end = _get_passes_with_end_location(opponent_passes_midfield)
    
    if len(passes_with_end) > 0:
        forward_passes = passes_with_end[passes_with_end['end_x'] > passes_with_end['x']]
        return len(forward_passes)
    return 0


def compute_passes_allowed_backward(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of backward passes allowed (end_x < start_x)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    passes_with_end = _get_passes_with_end_location(opponent_passes_midfield)
    
    if len(passes_with_end) > 0:
        backward_passes = passes_with_end[passes_with_end['end_x'] < passes_with_end['x']]
        return len(backward_passes)
    return 0


def compute_passes_allowed_lateral(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of lateral passes allowed (more y movement than x movement)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    passes_with_end = _get_passes_with_end_location(opponent_passes_midfield)
    
    if len(passes_with_end) > 0:
        lateral_passes = passes_with_end[
            (passes_with_end['end_y'] - passes_with_end['y']).abs() > 
            (passes_with_end['end_x'] - passes_with_end['x']).abs()
        ]
        return len(lateral_passes)
    return 0


def compute_average_pass_length_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average pass length allowed."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(opponent_passes_midfield) > 0:
        pass_lengths = opponent_passes_midfield.get('pass.length', pd.Series())
        if pass_lengths.notna().any():
            return pass_lengths.mean()
    return None


def compute_long_passes_allowed_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of long passes allowed (> 20m)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(opponent_passes_midfield) > 0:
        long_passes = opponent_passes_midfield[
            opponent_passes_midfield.get('pass.length', pd.Series()) > 20
        ]
        return len(long_passes)
    return 0


def compute_passes_before_interception(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Average number of passes before first interception per possession."""
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    passes_before_interception = []
    for poss in opponent_possessions:
        poss_events = events[events['possession'] == poss].sort_values('timestamp')
        poss_passes = poss_events[poss_events['type_name'] == 'Pass']
        poss_interceptions = poss_events[
            (poss_events['type_name'] == 'Interception') &
            (poss_events.get('team.id', pd.Series()) == team_id)
        ]
        
        if len(poss_interceptions) > 0:
            first_interception_idx = poss_interceptions.index[0]
            passes_before = len(poss_passes[poss_passes.index < first_interception_idx])
            passes_before_interception.append(passes_before)
    
    if passes_before_interception:
        return np.mean(passes_before_interception)
    return 0.0


def compute_consecutive_passes_allowed_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Average consecutive passes allowed per possession."""
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    consecutive_passes = []
    for poss in opponent_possessions:
        poss_events = events[events['possession'] == poss].sort_values('timestamp')
        poss_passes = poss_events[poss_events['type_name'] == 'Pass']
        poss_interceptions = poss_events[
            (poss_events['type_name'] == 'Interception') &
            (poss_events.get('team.id', pd.Series()) == team_id)
        ]
        
        if len(poss_interceptions) > 0:
            first_interception_idx = poss_interceptions.index[0]
            passes_before = len(poss_passes[poss_passes.index < first_interception_idx])
            consecutive_passes.append(passes_before)
        else:
            consecutive_passes.append(len(poss_passes))
    
    if consecutive_passes:
        return np.mean(consecutive_passes)
    return 0.0


def compute_pass_sequence_length_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Average pass sequence length per possession."""
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    pass_sequences = []
    for poss in opponent_possessions:
        poss_events = events[events['possession'] == poss].sort_values('timestamp')
        poss_passes = poss_events[poss_events['type_name'] == 'Pass']
        
        if len(poss_passes) > 0:
            pass_sequences.append(len(poss_passes))
    
    if pass_sequences:
        return np.mean(pass_sequences)
    return 0.0


def compute_pass_chain_break_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of pass sequences broken by interceptions."""
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    broken_sequences = 0
    for poss in opponent_possessions:
        poss_events = events[events['possession'] == poss].sort_values('timestamp')
        poss_passes = poss_events[poss_events['type_name'] == 'Pass']
        poss_interceptions = poss_events[
            (poss_events['type_name'] == 'Interception') &
            (poss_events.get('team.id', pd.Series()) == team_id)
        ]
        
        if len(poss_interceptions) > 0 and len(poss_passes) > 0:
            first_interception_idx = poss_interceptions.index[0]
            passes_before = len(poss_passes[poss_passes.index < first_interception_idx])
            if passes_before < len(poss_passes):
                broken_sequences += 1
    
    if len(opponent_possessions) > 0:
        return broken_sequences / len(opponent_possessions)
    return 0.0


def compute_passes_allowed_left_to_right(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of passes from left to right (end_y > start_y)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    passes_with_end = _get_passes_with_end_location(opponent_passes_midfield)
    
    if len(passes_with_end) > 0 and 'end_y' in passes_with_end.columns:
        left_to_right = passes_with_end[passes_with_end['end_y'] > passes_with_end['y']]
        return len(left_to_right)
    return 0


def compute_passes_allowed_right_to_left(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of passes from right to left (end_y < start_y)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    passes_with_end = _get_passes_with_end_location(opponent_passes_midfield)
    
    if len(passes_with_end) > 0 and 'end_y' in passes_with_end.columns:
        right_to_left = passes_with_end[passes_with_end['end_y'] < passes_with_end['y']]
        return len(right_to_left)
    return 0


def compute_passes_allowed_center_to_wide(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of passes from center (35-45) to wide areas."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    passes_with_end = _get_passes_with_end_location(opponent_passes_midfield)
    
    if len(passes_with_end) > 0 and 'end_y' in passes_with_end.columns:
        center_to_wide = passes_with_end[
            (passes_with_end['y'] >= 35) & (passes_with_end['y'] <= 45) &
            ((passes_with_end['end_y'] < 35) | (passes_with_end['end_y'] > 45))
        ]
        return len(center_to_wide)
    return 0


def compute_passes_allowed_wide_to_center(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of passes from wide areas to center (35-45)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    passes_with_end = _get_passes_with_end_location(opponent_passes_midfield)
    
    if len(passes_with_end) > 0 and 'end_y' in passes_with_end.columns:
        wide_to_center = passes_with_end[
            ((passes_with_end['y'] < 35) | (passes_with_end['y'] > 45)) &
            (passes_with_end['end_y'] >= 35) & (passes_with_end['end_y'] <= 45)
        ]
        return len(wide_to_center)
    return 0


def compute_passes_allowed_defensive_to_final(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of passes from defensive third (< 40) through midfield to final third (> 80)."""
    opponent_passes_midfield = _get_opponent_passes_midfield(events, team_id, midfield_x_min, midfield_x_max)
    passes_with_end = _get_passes_with_end_location(opponent_passes_midfield)
    
    if len(passes_with_end) > 0:
        defensive_to_final = passes_with_end[
            (passes_with_end['x'] < 40) & (passes_with_end['end_x'] > 80)
        ]
        return len(defensive_to_final)
    return 0


def compute_passing_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all passing through midfield features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, possession_team.id,
        x, y, timestamp, possession, pass.length, pass.end_location,
        pass.outcome.name
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
    
    features['passes_allowed_midfield_total'] = compute_passes_allowed_midfield_total(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_intercepted_midfield'] = compute_passes_intercepted_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['pass_completion_rate_allowed_midfield'] = compute_pass_completion_rate_allowed_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_allowed_forward'] = compute_passes_allowed_forward(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_allowed_backward'] = compute_passes_allowed_backward(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_allowed_lateral'] = compute_passes_allowed_lateral(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['average_pass_length_allowed'] = compute_average_pass_length_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['long_passes_allowed_midfield'] = compute_long_passes_allowed_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_before_interception'] = compute_passes_before_interception(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['consecutive_passes_allowed_midfield'] = compute_consecutive_passes_allowed_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['pass_sequence_length_allowed'] = compute_pass_sequence_length_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['pass_chain_break_rate'] = compute_pass_chain_break_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_allowed_left_to_right'] = compute_passes_allowed_left_to_right(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_allowed_right_to_left'] = compute_passes_allowed_right_to_left(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_allowed_center_to_wide'] = compute_passes_allowed_center_to_wide(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_allowed_wide_to_center'] = compute_passes_allowed_wide_to_center(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['passes_allowed_defensive_to_final'] = compute_passes_allowed_defensive_to_final(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
