"""
Contextual Features

Computes features related to match context and play patterns.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_score_differential(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Score differential (requires match state data - returns None)."""
    return None


def compute_match_minute(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean minute of midfield events."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'minute' in midfield_events.columns:
        return midfield_events['minute'].mean()
    return None


def compute_period(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[int]:
    """Most common period of midfield events."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'period' in midfield_events.columns and len(midfield_events) > 0:
        return midfield_events['period'].mode().iloc[0]
    return None


def compute_home_away(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[str]:
    """Home/away status (requires match metadata - returns None)."""
    return None


def compute_time_since_last_goal(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average time since last goal for midfield events."""
    goal_events = events[events['type_name'] == 'Shot'].copy()
    if 'shot.outcome.name' in goal_events.columns:
        goal_events = goal_events[goal_events['shot.outcome.name'] == 'Goal']
    elif 'outcome_name' in goal_events.columns:
        goal_events = goal_events[goal_events['outcome_name'] == 'Goal']
    else:
        goal_events = pd.DataFrame()
    
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if len(goal_events) > 0 and len(midfield_events) > 0:
        last_goal_time = goal_events.iloc[-1].get('timestamp')
        if pd.notna(last_goal_time):
            time_diffs = []
            for idx, event in midfield_events.iterrows():
                event_time = event.get('timestamp')
                if pd.notna(event_time) and event_time >= last_goal_time:
                    time_diff = (event_time - last_goal_time).total_seconds()
                    time_diffs.append(time_diff)
            
            if time_diffs:
                return np.mean(time_diffs)
    return None


def compute_play_pattern_regular(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[int]:
    """Number of regular play events."""
    if 'play_pattern.name' in events.columns or 'play_pattern_name' in events.columns:
        pattern_col = 'play_pattern.name' if 'play_pattern.name' in events.columns else 'play_pattern_name'
        regular_play = events[
            events.get(pattern_col, pd.Series()) == 'Regular Play'
        ]
        return len(regular_play)
    return None


def compute_play_pattern_set_piece(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[int]:
    """Number of set piece events."""
    if 'play_pattern.name' in events.columns or 'play_pattern_name' in events.columns:
        pattern_col = 'play_pattern.name' if 'play_pattern.name' in events.columns else 'play_pattern_name'
        set_pieces = events[
            events.get(pattern_col, pd.Series()).isin([
                'From Free Kick', 'From Corner', 'From Throw In', 'From Kick Off'
            ])
        ]
        return len(set_pieces)
    return None


def compute_play_pattern_transition(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of transition events (simplified - returns 0)."""
    return 0


def compute_play_pattern_counter_attack(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of counter attack events (simplified - returns 0)."""
    return 0


def compute_contextual_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all contextual features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, possession_team.id,
        x, y, timestamp, minute, period, play_pattern.name
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
    
    features['score_differential'] = compute_score_differential(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['match_minute'] = compute_match_minute(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['period'] = compute_period(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['home_away'] = compute_home_away(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['time_since_last_goal'] = compute_time_since_last_goal(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['play_pattern_regular'] = compute_play_pattern_regular(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['play_pattern_set_piece'] = compute_play_pattern_set_piece(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['play_pattern_transition'] = compute_play_pattern_transition(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['play_pattern_counter_attack'] = compute_play_pattern_counter_attack(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
