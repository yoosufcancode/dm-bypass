"""
Access Control Features

Computes features related to preventing progressive passes, through balls, and switches.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_progressive_passes_allowed_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of progressive passes allowed through midfield (pass length > 10m AND end_x > start_x + 5m)."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.length' not in opponent_passes.columns or 'pass.end_location' not in opponent_passes.columns:
        return 0
    
    progressive_count = 0
    for _, pass_event in opponent_passes.iterrows():
        pass_length = pass_event.get('pass.length')
        pass_end_loc = pass_event.get('pass.end_location')
        pass_x = pass_event.get('x')
        
        if (pd.notna(pass_length) and pd.notna(pass_end_loc) and 
            pd.notna(pass_x) and isinstance(pass_end_loc, list) and len(pass_end_loc) > 0):
            if pass_length > 10 and pass_end_loc[0] > pass_x + 5:
                progressive_count += 1
    
    return progressive_count


def compute_progressive_pass_prevention_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Percentage of progressive passes intercepted/blocked."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.length' not in opponent_passes.columns or 'pass.end_location' not in opponent_passes.columns:
        return 0.0
    
    progressive_passes = []
    for idx, pass_event in opponent_passes.iterrows():
        pass_length = pass_event.get('pass.length')
        pass_end_loc = pass_event.get('pass.end_location')
        pass_x = pass_event.get('x')
        
        if (pd.notna(pass_length) and pd.notna(pass_end_loc) and 
            pd.notna(pass_x) and isinstance(pass_end_loc, list) and len(pass_end_loc) > 0):
            if pass_length > 10 and pass_end_loc[0] > pass_x + 5:
                progressive_passes.append(idx)
    
    progressive = opponent_passes.loc[progressive_passes] if progressive_passes else pd.DataFrame()
    
    if len(progressive) > 0:
        outcome_col = 'pass.outcome.name'
        if outcome_col in progressive.columns:
            intercepted = progressive[progressive[outcome_col] == 'Incomplete']
            return len(intercepted) / len(progressive)
    return 0.0


def compute_progressive_passes_central_lane(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Progressive passes through central lane (35 ≤ y ≤ 45)."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.length' not in opponent_passes.columns or 'pass.end_location' not in opponent_passes.columns or 'y' not in opponent_passes.columns:
        return 0
    
    progressive_passes = []
    for idx, pass_event in opponent_passes.iterrows():
        pass_length = pass_event.get('pass.length')
        pass_end_loc = pass_event.get('pass.end_location')
        pass_x = pass_event.get('x')
        pass_y = pass_event.get('y')
        
        if (pd.notna(pass_length) and pd.notna(pass_end_loc) and 
            pd.notna(pass_x) and pd.notna(pass_y) and isinstance(pass_end_loc, list) and len(pass_end_loc) > 0):
            if pass_length > 10 and pass_end_loc[0] > pass_x + 5 and 35 <= pass_y <= 45:
                progressive_passes.append(idx)
    
    return len(progressive_passes)


def compute_progressive_passes_wide(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Progressive passes through wide areas."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.length' not in opponent_passes.columns or 'pass.end_location' not in opponent_passes.columns or 'y' not in opponent_passes.columns:
        return 0
    
    progressive_passes = []
    for idx, pass_event in opponent_passes.iterrows():
        pass_length = pass_event.get('pass.length')
        pass_end_loc = pass_event.get('pass.end_location')
        pass_x = pass_event.get('x')
        pass_y = pass_event.get('y')
        
        if (pd.notna(pass_length) and pd.notna(pass_end_loc) and 
            pd.notna(pass_x) and pd.notna(pass_y) and isinstance(pass_end_loc, list) and len(pass_end_loc) > 0):
            if pass_length > 10 and pass_end_loc[0] > pass_x + 5 and (pass_y < 35 or pass_y > 45):
                progressive_passes.append(idx)
    
    return len(progressive_passes)


def compute_progressive_pass_distance_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average distance of progressive passes allowed."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.length' not in opponent_passes.columns or 'pass.end_location' not in opponent_passes.columns:
        return None
    
    progressive_lengths = []
    for _, pass_event in opponent_passes.iterrows():
        pass_length = pass_event.get('pass.length')
        pass_end_loc = pass_event.get('pass.end_location')
        pass_x = pass_event.get('x')
        
        if (pd.notna(pass_length) and pd.notna(pass_end_loc) and 
            pd.notna(pass_x) and isinstance(pass_end_loc, list) and len(pass_end_loc) > 0):
            if pass_length > 10 and pass_end_loc[0] > pass_x + 5:
                progressive_lengths.append(pass_length)
    
    if progressive_lengths:
        return np.mean(progressive_lengths)
    return None


def compute_progressive_pass_angle_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average angle of progressive passes (forward vs. sideways)."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.length' not in opponent_passes.columns or 'pass.end_location' not in opponent_passes.columns or 'pass.angle' not in opponent_passes.columns:
        return None
    
    progressive_angles = []
    for _, pass_event in opponent_passes.iterrows():
        pass_length = pass_event.get('pass.length')
        pass_end_loc = pass_event.get('pass.end_location')
        pass_x = pass_event.get('x')
        pass_angle = pass_event.get('pass.angle')
        
        if (pd.notna(pass_length) and pd.notna(pass_end_loc) and 
            pd.notna(pass_x) and pd.notna(pass_angle) and isinstance(pass_end_loc, list) and len(pass_end_loc) > 0):
            if pass_length > 10 and pass_end_loc[0] > pass_x + 5:
                progressive_angles.append(pass_angle)
    
    if progressive_angles:
        return np.mean(progressive_angles)
    return None


def compute_through_balls_allowed_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Through balls allowed in midfield."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.through_ball' not in opponent_passes.columns:
        return 0
    
    through_balls = opponent_passes[opponent_passes['pass.through_ball'] == True]
    return len(through_balls)


def compute_through_ball_prevention_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Percentage of through ball attempts intercepted."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.through_ball' not in opponent_passes.columns:
        return 0.0
    
    through_balls = opponent_passes[opponent_passes['pass.through_ball'] == True]
    
    if len(through_balls) > 0:
        outcome_col = 'pass.outcome.name'
        if outcome_col in through_balls.columns:
            intercepted_through_balls = through_balls[through_balls[outcome_col] == 'Incomplete']
            return len(intercepted_through_balls) / len(through_balls)
    return 0.0


def compute_through_balls_ending_final_third(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Through balls that reach final third."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.through_ball' not in opponent_passes.columns or 'pass.end_location' not in opponent_passes.columns:
        return 0
    
    through_balls = opponent_passes[opponent_passes['pass.through_ball'] == True]
    
    ending_final_third = 0
    for _, tb in through_balls.iterrows():
        end_loc = tb.get('pass.end_location')
        if pd.notna(end_loc) and isinstance(end_loc, list) and len(end_loc) > 0:
            if end_loc[0] > 80:
                ending_final_third += 1
    
    return ending_final_third


def compute_switches_allowed_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Long switches allowed through midfield."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.switch' not in opponent_passes.columns:
        return 0
    
    switches = opponent_passes[opponent_passes['pass.switch'] == True]
    return len(switches)


def compute_switch_prevention_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Percentage of switches intercepted."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.switch' not in opponent_passes.columns:
        return 0.0
    
    switches = opponent_passes[opponent_passes['pass.switch'] == True]
    
    if len(switches) > 0:
        outcome_col = 'pass.outcome.name'
        if outcome_col in switches.columns:
            intercepted_switches = switches[switches[outcome_col] == 'Incomplete']
            return len(intercepted_switches) / len(switches)
    return 0.0


def compute_switch_distance_allowed(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average distance of switches allowed."""
    opponent_passes = events[
        (events['type_name'] == 'Pass') &
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'pass.switch' not in opponent_passes.columns or 'pass.length' not in opponent_passes.columns:
        return None
    
    switches = opponent_passes[opponent_passes['pass.switch'] == True]
    
    if len(switches) > 0 and 'pass.length' in switches.columns:
        return switches['pass.length'].mean()
    return None


def compute_access_control_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all access control features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, possession_team.id,
        x, y, timestamp, possession, pass.length, pass.end_location,
        pass.through_ball, pass.switch, pass.angle
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
    
    features['progressive_passes_allowed_midfield'] = compute_progressive_passes_allowed_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['progressive_pass_prevention_rate'] = compute_progressive_pass_prevention_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['progressive_passes_central_lane'] = compute_progressive_passes_central_lane(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['progressive_passes_wide'] = compute_progressive_passes_wide(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['progressive_pass_distance_allowed'] = compute_progressive_pass_distance_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['progressive_pass_angle_allowed'] = compute_progressive_pass_angle_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['through_balls_allowed_midfield'] = compute_through_balls_allowed_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['through_ball_prevention_rate'] = compute_through_ball_prevention_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['through_balls_ending_final_third'] = compute_through_balls_ending_final_third(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['switches_allowed_midfield'] = compute_switches_allowed_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['switch_prevention_rate'] = compute_switch_prevention_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['switch_distance_allowed'] = compute_switch_distance_allowed(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
