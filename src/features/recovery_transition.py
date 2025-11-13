"""
Recovery and Transition Features

Computes features related to ball recoveries and transitions in midfield.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_midfield_transition_recoveries(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Recoveries during opponent transitions (within 3 seconds of possession change)."""
    midfield_recoveries = events[
        (events['type_name'] == 'Ball Recovery') &
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if len(midfield_recoveries) > 0:
        transition_recoveries = 0
        for idx, recovery in midfield_recoveries.iterrows():
            recovery_time = recovery.get('timestamp')
            if pd.notna(recovery_time):
                time_window = pd.Timedelta(seconds=3)
                possession_changes = events[
                    (events['timestamp'] >= recovery_time - time_window) &
                    (events['timestamp'] < recovery_time) &
                    (events.get('possession_team.id', pd.Series()) != team_id)
                ]
                if len(possession_changes) > 0:
                    transition_recoveries += 1
        return transition_recoveries
    return 0


def compute_midfield_counter_press_events(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Defensive actions within 5 seconds of losing possession."""
    barca_possessions = events[
        events.get('possession_team.id', pd.Series()) == team_id
    ]['possession'].unique()
    
    counter_press_events = 0
    for poss in barca_possessions:
        poss_events = events[events['possession'] == poss].sort_values('timestamp')
        
        lost_possession = poss_events[
            (poss_events.get('possession_team.id', pd.Series()) != team_id) |
            (poss_events['type_name'].isin(['Dispossessed', 'Miscontrol', 'Error']))
        ]
        
        if len(lost_possession) > 0:
            loss_time = lost_possession.iloc[0].get('timestamp')
            if pd.notna(loss_time):
                time_window = pd.Timedelta(seconds=5)
                defensive_actions = events[
                    (events.get('team.id', pd.Series()) == team_id) &
                    (events['timestamp'] >= loss_time) &
                    (events['timestamp'] <= loss_time + time_window) &
                    (events['type_name'].isin(['Interception', 'Duel', 'Ball Recovery', 'Pressure', 'Block']))
                ]
                counter_press_events += len(defensive_actions)
    
    return counter_press_events


def compute_midfield_transition_to_attack(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average time from recovery to first attacking action."""
    midfield_recoveries = events[
        (events['type_name'] == 'Ball Recovery') &
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if len(midfield_recoveries) > 0:
        recovery_to_attack_times = []
        for idx, recovery in midfield_recoveries.iterrows():
            recovery_time = recovery.get('timestamp')
            if pd.notna(recovery_time):
                attacking_actions = events[
                    (events.get('team.id', pd.Series()) == team_id) &
                    (events['timestamp'] > recovery_time) &
                    (events['type_name'].isin(['Pass', 'Shot', 'Carry', 'Dribble']))
                ]
                if len(attacking_actions) > 0:
                    time_diff = (attacking_actions.iloc[0].get('timestamp') - recovery_time).total_seconds()
                    recovery_to_attack_times.append(time_diff)
        
        if recovery_to_attack_times:
            return np.mean(recovery_to_attack_times)
    return None


def compute_midfield_recovery_location_quality(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean x-coordinate of recovery locations (higher = better)."""
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


def compute_possession_won_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of possessions won in midfield."""
    possessions_won_midfield = 0
    for poss in events['possession'].unique():
        poss_events = events[events['possession'] == poss].sort_values('timestamp')
        if len(poss_events) > 0:
            first_event = poss_events.iloc[0]
            if (first_event.get('x', 0) >= midfield_x_min and 
                first_event.get('x', 0) <= midfield_x_max and
                first_event.get('possession_team.id') == team_id):
                possessions_won_midfield += 1
    return possessions_won_midfield


def compute_possession_lost_midfield(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of possessions lost in midfield."""
    possessions_lost_midfield = len(events[
        (events.get('possession_team.id', pd.Series()) == team_id) &
        (events['type_name'].isin(['Dispossessed', 'Miscontrol', 'Error'])) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ])
    return possessions_lost_midfield


def compute_midfield_possession_win_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of midfield duels won."""
    midfield_duels = events[
        (events['type_name'] == 'Duel') &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    barca_won_duels = midfield_duels[
        (midfield_duels.get('team.id', pd.Series()) == team_id) &
        (midfield_duels.get('duel.outcome.name', pd.Series()).str.contains('Won', na=False))
    ]
    
    if len(midfield_duels) > 0:
        return len(barca_won_duels) / len(midfield_duels)
    return 0.0


def compute_midfield_turnover_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of opponent possessions ending due to team's defensive actions."""
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    turnovers_forced = 0
    for poss in opponent_possessions:
        poss_events = events[events['possession'] == poss]
        if len(poss_events) > 0:
            last_event = poss_events.iloc[-1]
            if (last_event.get('team.id') == team_id and
                last_event['type_name'] in ['Interception', 'Ball Recovery', 'Duel']):
                turnovers_forced += 1
    
    if len(opponent_possessions) > 0:
        return turnovers_forced / len(opponent_possessions)
    return 0.0


def compute_recovery_transition_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all recovery and transition features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, possession_team.id,
        x, y, timestamp, possession
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
    
    features['midfield_transition_recoveries'] = compute_midfield_transition_recoveries(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_counter_press_events'] = compute_midfield_counter_press_events(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_transition_to_attack'] = compute_midfield_transition_to_attack(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_recovery_location_quality'] = compute_midfield_recovery_location_quality(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['possession_won_midfield'] = compute_possession_won_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['possession_lost_midfield'] = compute_possession_lost_midfield(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_possession_win_rate'] = compute_midfield_possession_win_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_turnover_rate'] = compute_midfield_turnover_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
