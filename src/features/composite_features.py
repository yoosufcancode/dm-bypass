"""
Advanced Composite Features

Computes composite metrics that combine multiple features into higher-level indicators.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


def normalize_feature(value: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """
    Normalize a feature value to 0-1 range.
    
    Parameters:
    -----------
    value : float
        Feature value to normalize
    min_val : float, optional
        Minimum value for normalization
    max_val : float, optional
        Maximum value for normalization
    
    Returns:
    --------
    float
        Normalized value (0-1)
    """
    if pd.isna(value) or value is None:
        return 0.0
    
    if min_val is None or max_val is None:
        return min(max(value / 100.0, 0.0), 1.0) if value > 0 else 0.0
    
    if max_val == min_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def compute_midfield_strength_index(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0,
    other_features: Optional[Dict[str, float]] = None
) -> float:
    """Composite midfield strength index (weighted combination of key metrics)."""
    if other_features is None:
        other_features = {}
    
    midfield_interceptions = events[
        (events['type_name'] == 'Interception') &
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].nunique()
    
    if opponent_possessions > 0:
        interception_rate = len(midfield_interceptions) / opponent_possessions
    else:
        interception_rate = 0.0
    
    midfield_recoveries = events[
        (events['type_name'] == 'Ball Recovery') &
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if len(events) > 0:
        match_duration_minutes = events['minute'].max() if 'minute' in events.columns else 90.0
    else:
        match_duration_minutes = 90.0
    
    if match_duration_minutes > 0:
        recovery_rate = len(midfield_recoveries) / match_duration_minutes
    else:
        recovery_rate = 0.0
    
    opponent_midfield_events = events[
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    pressure_events = opponent_midfield_events[
        opponent_midfield_events.get('under_pressure', pd.Series()).fillna(False) == True
    ]
    
    if len(opponent_midfield_events) > 0:
        pressure_intensity = len(pressure_events) / len(opponent_midfield_events) * 100
    else:
        pressure_intensity = 0.0
    
    bypass_attempts = other_features.get('bypass_attempts_total', 0)
    if opponent_possessions > 0:
        bypass_prevention_rate = 1 - (bypass_attempts / opponent_possessions)
    else:
        bypass_prevention_rate = 1.0
    
    defensive_events = events[
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna() &
        events['y'].notna()
    ]
    
    if len(defensive_events) > 0:
        width = defensive_events['y'].max() - defensive_events['y'].min()
        depth = defensive_events['x'].max() - defensive_events['x'].min()
        if depth > 0:
            compactness_index = width / depth
        else:
            compactness_index = 1.0
    else:
        compactness_index = 1.0
    
    norm_interception = normalize_feature(interception_rate, 0, 1)
    norm_recovery = normalize_feature(recovery_rate, 0, 5)
    norm_pressure = normalize_feature(pressure_intensity, 0, 100)
    norm_bypass = bypass_prevention_rate
    norm_compactness = 1 - normalize_feature(compactness_index, 0, 2)
    
    return (
        0.25 * norm_interception +
        0.25 * norm_recovery +
        0.20 * norm_pressure +
        0.20 * norm_bypass +
        0.10 * norm_compactness
    )


def compute_bypass_risk_score(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0,
    other_features: Optional[Dict[str, float]] = None
) -> float:
    """Risk score based on allowed dangerous actions."""
    if other_features is None:
        other_features = {}
    
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].nunique()
    
    progressive_passes_allowed = other_features.get('progressive_passes_allowed_midfield', 0)
    through_balls_allowed = other_features.get('through_balls_allowed_midfield', 0)
    carries_through = other_features.get('carries_through_midfield', 0)
    
    total_dangerous_actions = progressive_passes_allowed + through_balls_allowed + carries_through
    if opponent_possessions > 0:
        risk_score = total_dangerous_actions / opponent_possessions
    else:
        risk_score = 0.0
    
    return normalize_feature(risk_score, 0, 10)


def compute_bypass_risk_factors(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0,
    other_features: Optional[Dict[str, float]] = None
) -> List[str]:
    """List of risk factors identified."""
    if other_features is None:
        other_features = {}
    
    progressive_passes_allowed = other_features.get('progressive_passes_allowed_midfield', 0)
    through_balls_allowed = other_features.get('through_balls_allowed_midfield', 0)
    carries_through = other_features.get('carries_through_midfield', 0)
    
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].nunique()
    
    bypass_attempts = other_features.get('bypass_attempts_total', 0)
    if opponent_possessions > 0:
        bypass_prevention_rate = 1 - (bypass_attempts / opponent_possessions)
    else:
        bypass_prevention_rate = 1.0
    
    risk_factors = []
    if progressive_passes_allowed > 50:
        risk_factors.append('high_progressive_passes')
    if through_balls_allowed > 10:
        risk_factors.append('high_through_balls')
    if carries_through > 20:
        risk_factors.append('high_carries_through')
    if bypass_prevention_rate < 0.5:
        risk_factors.append('low_bypass_prevention')
    
    return risk_factors


def compute_defensive_action_efficiency(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Percentage of opponent events that result in successful defensive actions."""
    midfield_interceptions = events[
        (events['type_name'] == 'Interception') &
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    midfield_recoveries = events[
        (events['type_name'] == 'Ball Recovery') &
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    successful_defensive_actions = (
        len(midfield_interceptions) +
        len(midfield_recoveries) +
        len(events[
            (events.get('team.id', pd.Series()) == team_id) &
            (events['type_name'] == 'Duel') &
            (events.get('duel.outcome.name', pd.Series()).str.contains('Won', na=False)) &
            (events['x'] >= midfield_x_min) & 
            (events['x'] <= midfield_x_max) &
            events['x'].notna()
        ])
    )
    
    opponent_midfield_events = events[
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if len(opponent_midfield_events) > 0:
        return (successful_defensive_actions / len(opponent_midfield_events)) * 100
    return 0.0


def compute_pressure_to_interception_ratio(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Ratio of interceptions to pressure events."""
    midfield_interceptions = events[
        (events['type_name'] == 'Interception') &
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    opponent_midfield_events = events[
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    pressure_events = opponent_midfield_events[
        opponent_midfield_events.get('under_pressure', pd.Series()).fillna(False) == True
    ]
    
    if len(pressure_events) > 0:
        return len(midfield_interceptions) / len(pressure_events)
    return 0.0


def compute_recovery_quality_score(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average x-coordinate of recoveries (higher = better position)."""
    midfield_recoveries = events[
        (events['type_name'] == 'Ball Recovery') &
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if len(midfield_recoveries) > 0:
        return midfield_recoveries['x'].mean()
    return None


def compute_composite_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0,
    other_features: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute all composite features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe
    team_id : int
        Team ID for Barcelona (default: 217)
    midfield_x_min : float
        Minimum x-coordinate for midfield (default: 40.0)
    midfield_x_max : float
        Maximum x-coordinate for midfield (default: 80.0)
    other_features : Dict[str, float], optional
        Dictionary of other computed features to use in composite metrics
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of feature names and values
    """
    features = {}
    
    features['midfield_strength_index'] = compute_midfield_strength_index(
        events, team_id, midfield_x_min, midfield_x_max, other_features
    )
    
    features['bypass_risk_score'] = compute_bypass_risk_score(
        events, team_id, midfield_x_min, midfield_x_max, other_features
    )
    
    features['bypass_risk_factors'] = compute_bypass_risk_factors(
        events, team_id, midfield_x_min, midfield_x_max, other_features
    )
    
    features['defensive_action_efficiency'] = compute_defensive_action_efficiency(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['pressure_to_interception_ratio'] = compute_pressure_to_interception_ratio(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['recovery_quality_score'] = compute_recovery_quality_score(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
