"""
Pressure and Tempo Features

Computes features related to pressure intensity and tempo in midfield.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_midfield_pressure_events_total(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Total events where under_pressure = true for opponents in midfield."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    if 'under_pressure' in opponent_midfield_events.columns:
        opponent_midfield_under_pressure = opponent_midfield_events[
            opponent_midfield_events['under_pressure'].fillna(False) == True
        ]
        return len(opponent_midfield_under_pressure)
    return 0


def compute_midfield_pressure_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Pressure events per minute in midfield."""
    pressure_total = compute_midfield_pressure_events_total(events, team_id, midfield_x_min, midfield_x_max)
    if len(events) > 0 and 'minute' in events.columns and 'second' in events.columns:
        match_duration_minutes = events['minute'].max() + (events['second'].max() / 60.0)
        if match_duration_minutes > 0:
            return pressure_total / match_duration_minutes
    return 0.0


def compute_midfield_pressure_in_first_5s(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of pressure events within first 5 seconds of opponent possession."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    if 'under_pressure' not in opponent_midfield_events.columns:
        return 0
    
    opponent_midfield_under_pressure = opponent_midfield_events[
        opponent_midfield_events['under_pressure'].fillna(False) == True
    ]
    
    if 'possession' not in opponent_midfield_under_pressure.columns or 'timestamp' not in opponent_midfield_under_pressure.columns:
        return 0
    
    possession_start_times = events.groupby('possession')['timestamp'].min()
    pressure_in_first_5s = 0
    
    for poss_id in opponent_midfield_under_pressure['possession'].unique():
        poss_start = possession_start_times.get(poss_id)
        if poss_start is not None:
            poss_pressures = opponent_midfield_under_pressure[
                opponent_midfield_under_pressure['possession'] == poss_id
            ]
            for _, pressure_event in poss_pressures.iterrows():
                time_diff = (pressure_event['timestamp'] - poss_start).total_seconds()
                if time_diff <= 5:
                    pressure_in_first_5s += 1
    
    return pressure_in_first_5s


def compute_midfield_pressure_in_first_10s(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of pressure events within first 10 seconds."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    if 'under_pressure' not in opponent_midfield_events.columns:
        return 0
    
    opponent_midfield_under_pressure = opponent_midfield_events[
        opponent_midfield_events['under_pressure'].fillna(False) == True
    ]
    
    if 'possession' not in opponent_midfield_under_pressure.columns or 'timestamp' not in opponent_midfield_under_pressure.columns:
        return 0
    
    possession_start_times = events.groupby('possession')['timestamp'].min()
    pressure_in_first_10s = 0
    
    for poss_id in opponent_midfield_under_pressure['possession'].unique():
        poss_start = possession_start_times.get(poss_id)
        if poss_start is not None:
            poss_pressures = opponent_midfield_under_pressure[
                opponent_midfield_under_pressure['possession'] == poss_id
            ]
            for _, pressure_event in poss_pressures.iterrows():
                time_diff = (pressure_event['timestamp'] - poss_start).total_seconds()
                if time_diff <= 10:
                    pressure_in_first_10s += 1
    
    return pressure_in_first_10s


def compute_midfield_time_to_first_pressure(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average time from possession start to first pressure event."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    if 'under_pressure' not in opponent_midfield_events.columns:
        return None
    
    opponent_midfield_under_pressure = opponent_midfield_events[
        opponent_midfield_events['under_pressure'].fillna(False) == True
    ]
    
    if 'possession' not in opponent_midfield_under_pressure.columns or 'timestamp' not in opponent_midfield_under_pressure.columns:
        return None
    
    possession_start_times = events.groupby('possession')['timestamp'].min()
    time_to_first_pressure_list = []
    
    for poss_id in opponent_midfield_under_pressure['possession'].unique():
        poss_start = possession_start_times.get(poss_id)
        if poss_start is not None:
            first_pressure = opponent_midfield_under_pressure[
                opponent_midfield_under_pressure['possession'] == poss_id
            ].sort_values('timestamp')
            if len(first_pressure) > 0:
                time_diff = (first_pressure.iloc[0]['timestamp'] - poss_start).total_seconds()
                time_to_first_pressure_list.append(time_diff)
    
    if time_to_first_pressure_list:
        return np.mean(time_to_first_pressure_list)
    return None


def compute_midfield_pressure_zone_left(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Pressure events in left midfield zone (y < 26.67)."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    if 'under_pressure' not in opponent_midfield_events.columns:
        return 0
    
    opponent_midfield_under_pressure = opponent_midfield_events[
        opponent_midfield_events['under_pressure'].fillna(False) == True
    ]
    
    if 'y' in opponent_midfield_under_pressure.columns and len(opponent_midfield_under_pressure) > 0:
        return len(opponent_midfield_under_pressure[
            opponent_midfield_under_pressure['y'] < 26.67
        ])
    return 0


def compute_midfield_pressure_zone_center(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Pressure events in central midfield zone (26.67 ≤ y ≤ 53.33)."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    if 'under_pressure' not in opponent_midfield_events.columns:
        return 0
    
    opponent_midfield_under_pressure = opponent_midfield_events[
        opponent_midfield_events['under_pressure'].fillna(False) == True
    ]
    
    if 'y' in opponent_midfield_under_pressure.columns and len(opponent_midfield_under_pressure) > 0:
        return len(opponent_midfield_under_pressure[
            (opponent_midfield_under_pressure['y'] >= 26.67) & 
            (opponent_midfield_under_pressure['y'] <= 53.33)
        ])
    return 0


def compute_midfield_pressure_zone_right(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Pressure events in right midfield zone (y > 53.33)."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    if 'under_pressure' not in opponent_midfield_events.columns:
        return 0
    
    opponent_midfield_under_pressure = opponent_midfield_events[
        opponent_midfield_events['under_pressure'].fillna(False) == True
    ]
    
    if 'y' in opponent_midfield_under_pressure.columns and len(opponent_midfield_under_pressure) > 0:
        return len(opponent_midfield_under_pressure[
            opponent_midfield_under_pressure['y'] > 53.33
        ])
    return 0


def compute_midfield_pressure_on_passes(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Pressure applied during opponent passes."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    opponent_passes_midfield = opponent_midfield_events[
        opponent_midfield_events['type_name'] == 'Pass'
    ]
    if 'under_pressure' in opponent_passes_midfield.columns:
        return len(opponent_passes_midfield[
            opponent_passes_midfield['under_pressure'].fillna(False) == True
        ])
    return 0


def compute_midfield_pressure_on_receipts(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Pressure applied during ball receipts."""
    midfield_events = events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    opponent_midfield_events = midfield_events[
        (midfield_events.get('possession_team.id', pd.Series()) != team_id)
    ]
    opponent_receipts_midfield = opponent_midfield_events[
        opponent_midfield_events['type_name'] == 'Ball Receipt*'
    ]
    if 'under_pressure' in opponent_receipts_midfield.columns:
        return len(opponent_receipts_midfield[
            opponent_receipts_midfield['under_pressure'].fillna(False) == True
        ])
    return 0


def compute_midfield_reaction_time(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average time between opponent event and Barcelona defensive action."""
    opponent_midfield_events = events[
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    defensive_actions = events[
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna() &
        (events['type_name'].isin(['Interception', 'Ball Recovery', 'Duel', 'Block', 'Clearance']))
    ]
    
    if 'timestamp' not in opponent_midfield_events.columns or 'timestamp' not in defensive_actions.columns:
        return None
    
    reaction_times = []
    for _, opp_event in opponent_midfield_events.iterrows():
        opp_time = opp_event.get('timestamp')
        if pd.notna(opp_time):
            next_defensive = defensive_actions[
                defensive_actions['timestamp'] > opp_time
            ].sort_values('timestamp')
            if len(next_defensive) > 0:
                time_diff = (next_defensive.iloc[0]['timestamp'] - opp_time).total_seconds()
                reaction_times.append(time_diff)
    
    if reaction_times:
        return np.mean(reaction_times)
    return None


def compute_midfield_pressing_intensity(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Events per second in first 10 seconds of opponent possession."""
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    if 'timestamp' not in events.columns:
        return 0.0
    
    intensities = []
    for poss in opponent_possessions:
        poss_events = events[events['possession'] == poss].sort_values('timestamp')
        if len(poss_events) > 0:
            start_time = poss_events.iloc[0]['timestamp']
            events_in_10s = poss_events[
                (poss_events['timestamp'] - start_time).dt.total_seconds() <= 10
            ]
            if len(events_in_10s) > 0:
                duration = (events_in_10s.iloc[-1]['timestamp'] - start_time).total_seconds()
                if duration > 0:
                    intensities.append(len(events_in_10s) / duration)
    
    if intensities:
        return np.mean(intensities)
    return 0.0


def compute_midfield_immediate_pressure_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Percentage of opponent events pressured within 2 seconds."""
    opponent_midfield_events = events[
        (events.get('possession_team.id', pd.Series()) != team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ]
    
    if 'timestamp' not in opponent_midfield_events.columns or 'under_pressure' not in opponent_midfield_events.columns:
        return 0.0
    
    immediate_pressures = 0
    for _, opp_event in opponent_midfield_events.iterrows():
        if opp_event.get('under_pressure', False):
            opp_time = opp_event.get('timestamp')
            if pd.notna(opp_time):
                possession = opp_event.get('possession')
                if pd.notna(possession):
                    poss_events = events[events['possession'] == possession].sort_values('timestamp')
                    if len(poss_events) > 0:
                        poss_start = poss_events.iloc[0]['timestamp']
                        time_diff = (opp_time - poss_start).total_seconds()
                        if time_diff <= 2:
                            immediate_pressures += 1
    
    if len(opponent_midfield_events) > 0:
        return immediate_pressures / len(opponent_midfield_events)
    return 0.0


def compute_midfield_pressure_persistence(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Duration of sustained pressure sequences."""
    opponent_possessions = events[
        events.get('possession_team.id', pd.Series()) != team_id
    ]['possession'].unique()
    
    if 'timestamp' not in events.columns or 'under_pressure' not in events.columns:
        return None
    
    pressure_durations = []
    for poss in opponent_possessions:
        poss_events = events[events['possession'] == poss].sort_values('timestamp')
        pressure_events = poss_events[
            (poss_events.get('possession_team.id', pd.Series()) != team_id) &
            (poss_events['x'] >= midfield_x_min) & 
            (poss_events['x'] <= midfield_x_max) &
            poss_events['x'].notna() &
            (poss_events['under_pressure'].fillna(False) == True)
        ]
        
        if len(pressure_events) > 1:
            first_pressure = pressure_events.iloc[0]['timestamp']
            last_pressure = pressure_events.iloc[-1]['timestamp']
            if pd.notna(first_pressure) and pd.notna(last_pressure):
                duration = (last_pressure - first_pressure).total_seconds()
                pressure_durations.append(duration)
    
    if pressure_durations:
        return np.mean(pressure_durations)
    return None


def compute_pressure_tempo_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all pressure and tempo features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, possession_team.id,
        x, y, timestamp, possession, under_pressure
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
    
    features['midfield_pressure_events_total'] = compute_midfield_pressure_events_total(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_rate'] = compute_midfield_pressure_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_in_first_5s'] = compute_midfield_pressure_in_first_5s(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_in_first_10s'] = compute_midfield_pressure_in_first_10s(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_time_to_first_pressure'] = compute_midfield_time_to_first_pressure(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_zone_left'] = compute_midfield_pressure_zone_left(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_zone_center'] = compute_midfield_pressure_zone_center(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_zone_right'] = compute_midfield_pressure_zone_right(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_on_passes'] = compute_midfield_pressure_on_passes(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_on_receipts'] = compute_midfield_pressure_on_receipts(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_reaction_time'] = compute_midfield_reaction_time(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressing_intensity'] = compute_midfield_pressing_intensity(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_immediate_pressure_rate'] = compute_midfield_immediate_pressure_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_pressure_persistence'] = compute_midfield_pressure_persistence(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
