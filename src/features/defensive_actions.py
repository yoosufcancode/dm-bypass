"""
Defensive Actions in Midfield Features

Computes features related to interceptions, ball recoveries, and duels in midfield.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def _get_midfield_events(events, midfield_x_min, midfield_x_max):
    """Helper to get midfield events."""
    return events[
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ].copy()


def compute_midfield_interceptions_total(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Total interceptions by team in midfield."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_interceptions = midfield_events[
        (midfield_events['type_name'] == 'Interception') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    return len(midfield_interceptions)


def compute_midfield_interceptions_per_possession(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Interceptions per opponent possession."""
    total = compute_midfield_interceptions_total(events, team_id, midfield_x_min, midfield_x_max)
    opponent_possessions = events[events.get('possession_team.id', pd.Series()) != team_id]['possession'].nunique()
    if opponent_possessions > 0:
        return total / opponent_possessions
    return 0.0


def compute_midfield_interception_success_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of successful interceptions."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_interceptions = midfield_events[
        (midfield_events['type_name'] == 'Interception') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_interceptions) > 0:
        outcome_col = 'interception.outcome.name'
        if outcome_col in midfield_interceptions.columns:
            successful = midfield_interceptions[midfield_interceptions[outcome_col] == 'Success In Play']
            return len(successful) / len(midfield_interceptions)
    return 0.0


def compute_midfield_interceptions_central(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0,
    midfield_y_central_min: float = 30.0,
    midfield_y_central_max: float = 50.0
) -> int:
    """Interceptions in central zone (30 ≤ y ≤ 50)."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_interceptions = midfield_events[
        (midfield_events['type_name'] == 'Interception') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if 'y' in midfield_interceptions.columns and len(midfield_interceptions) > 0:
        return len(midfield_interceptions[
            (midfield_interceptions['y'] >= midfield_y_central_min) & 
            (midfield_interceptions['y'] <= midfield_y_central_max)
        ])
    return 0


def compute_midfield_interceptions_wide(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0,
    midfield_y_central_min: float = 30.0,
    midfield_y_central_max: float = 50.0
) -> int:
    """Interceptions in wide zones (y < 30 or y > 50)."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_interceptions = midfield_events[
        (midfield_events['type_name'] == 'Interception') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if 'y' in midfield_interceptions.columns and len(midfield_interceptions) > 0:
        return len(midfield_interceptions[
            (midfield_interceptions['y'] < midfield_y_central_min) | 
            (midfield_interceptions['y'] > midfield_y_central_max)
        ])
    return 0


def compute_midfield_interceptions_progressive(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of progressive passes intercepted."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_interceptions = midfield_events[
        (midfield_events['type_name'] == 'Interception') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    intercepted_pass_ids = set()
    if 'related_events' in midfield_interceptions.columns:
        for idx, interception in midfield_interceptions.iterrows():
            related_events = interception.get('related_events', [])
            if pd.notna(related_events) and related_events:
                related_passes = events[
                    (events.get('id', pd.Series()).isin(related_events)) &
                    (events['type_name'] == 'Pass')
                ]
                for _, pass_event in related_passes.iterrows():
                    pass_length = pass_event.get('pass.length')
                    pass_end_loc = pass_event.get('pass.end_location')
                    pass_x = pass_event.get('x')
                    
                    if (pd.notna(pass_length) and pd.notna(pass_end_loc) and 
                        pd.notna(pass_x) and isinstance(pass_end_loc, list) and len(pass_end_loc) > 0):
                        if pass_length > 10 and pass_end_loc[0] > pass_x + 5:
                            intercepted_pass_ids.add(pass_event.get('id'))
    
    return len(intercepted_pass_ids)


def compute_midfield_interception_time_to_event(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average time from possession start to first interception."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_interceptions = midfield_events[
        (midfield_events['type_name'] == 'Interception') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if 'possession' in midfield_interceptions.columns and 'timestamp' in midfield_interceptions.columns and len(midfield_interceptions) > 0:
        possession_start_times = events.groupby('possession')['timestamp'].min()
        interception_times = midfield_interceptions.groupby('possession')['timestamp'].min()
        time_diffs = []
        for poss_id in interception_times.index:
            if poss_id in possession_start_times.index:
                time_diff = (interception_times[poss_id] - possession_start_times[poss_id]).total_seconds()
                time_diffs.append(time_diff)
        if time_diffs:
            return np.mean(time_diffs)
    return None


def compute_midfield_recoveries_total(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Total ball recoveries by team in midfield."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_recoveries = midfield_events[
        (midfield_events['type_name'] == 'Ball Recovery') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    return len(midfield_recoveries)


def compute_midfield_recovery_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Recoveries per minute."""
    total = compute_midfield_recoveries_total(events, team_id, midfield_x_min, midfield_x_max)
    if len(events) > 0 and 'minute' in events.columns and 'second' in events.columns:
        match_duration_minutes = events['minute'].max() + (events['second'].max() / 60.0)
        if match_duration_minutes > 0:
            return total / match_duration_minutes
    return 0.0


def compute_midfield_recovery_success_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of successful recoveries (without recovery_failure)."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_recoveries = midfield_events[
        (midfield_events['type_name'] == 'Ball Recovery') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_recoveries) > 0:
        recovery_failure_col = 'ball_recovery.recovery_failure'
        if recovery_failure_col in midfield_recoveries.columns:
            successful = midfield_recoveries[~midfield_recoveries[recovery_failure_col].fillna(False)]
            return len(successful) / len(midfield_recoveries)
        return 1.0
    return 0.0


def compute_midfield_recoveries_after_pressure(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Recoveries that occurred after pressure events."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_recoveries = midfield_events[
        (midfield_events['type_name'] == 'Ball Recovery') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    recoveries_after_pressure = 0
    if 'possession' in midfield_recoveries.columns and 'timestamp' in midfield_recoveries.columns:
        for idx, recovery in midfield_recoveries.iterrows():
            recovery_time = recovery['timestamp']
            recovery_possession = recovery['possession']
            pressure_before = events[
                (events['possession'] == recovery_possession) &
                (events['type_name'] == 'Pressure') &
                (events['timestamp'] < recovery_time) &
                (events['x'] >= midfield_x_min) & (events['x'] <= midfield_x_max)
            ]
            if len(pressure_before) > 0:
                recoveries_after_pressure += 1
    return recoveries_after_pressure


def compute_midfield_recovery_locations_x(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean x-coordinate of recovery locations."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_recoveries = midfield_events[
        (midfield_events['type_name'] == 'Ball Recovery') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_recoveries) > 0 and 'x' in midfield_recoveries.columns:
        return midfield_recoveries['x'].mean()
    return None


def compute_midfield_recovery_locations_y(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean y-coordinate of recovery locations."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_recoveries = midfield_events[
        (midfield_events['type_name'] == 'Ball Recovery') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_recoveries) > 0 and 'y' in midfield_recoveries.columns:
        return midfield_recoveries['y'].mean()
    return None


def compute_midfield_recovery_time_to_event(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average time from possession start to recovery."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_recoveries = midfield_events[
        (midfield_events['type_name'] == 'Ball Recovery') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if 'possession' in midfield_recoveries.columns and 'timestamp' in midfield_recoveries.columns and len(midfield_recoveries) > 0:
        possession_start_times = events.groupby('possession')['timestamp'].min()
        recovery_times = midfield_recoveries.groupby('possession')['timestamp'].min()
        time_diffs = []
        for poss_id in recovery_times.index:
            if poss_id in possession_start_times.index:
                time_diff = (recovery_times[poss_id] - possession_start_times[poss_id]).total_seconds()
                time_diffs.append(time_diff)
        if time_diffs:
            return np.mean(time_diffs)
    return None


def compute_midfield_duels_total(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Total duels by team in midfield."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    return len(midfield_duels)


def compute_midfield_duel_win_rate(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of duels won."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_duels) > 0:
        outcome_col = 'duel.outcome.name'
        if outcome_col in midfield_duels.columns:
            won_duels = midfield_duels[midfield_duels[outcome_col].astype(str).str.contains('Won', na=False)]
            return len(won_duels) / len(midfield_duels)
    return 0.0


def compute_midfield_aerial_duels_won(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of aerial duels won."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_duels) > 0:
        type_col = 'duel.type.name'
        outcome_col = 'duel.outcome.name'
        if type_col in midfield_duels.columns:
            aerial_duels = midfield_duels[midfield_duels[type_col].astype(str).str.contains('Aerial', na=False)]
            if outcome_col in aerial_duels.columns:
                aerial_won = aerial_duels[aerial_duels[outcome_col].astype(str).str.contains('Won', na=False)]
                return len(aerial_won)
    return 0


def compute_midfield_ground_duels_won(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of ground duels won."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_duels) > 0:
        type_col = 'duel.type.name'
        outcome_col = 'duel.outcome.name'
        if type_col in midfield_duels.columns:
            ground_duels = midfield_duels[midfield_duels[type_col].astype(str).str.contains('Ground', na=False)]
            if outcome_col in ground_duels.columns:
                ground_won = ground_duels[ground_duels[outcome_col].astype(str).str.contains('Won', na=False)]
                return len(ground_won)
    return 0


def compute_midfield_duels_under_pressure(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of duels under pressure."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if 'under_pressure' in midfield_duels.columns and len(midfield_duels) > 0:
        return len(midfield_duels[midfield_duels['under_pressure'].fillna(False) == True])
    return 0


def compute_midfield_duel_locations_x_mean(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean x-coordinate of duel locations."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_duels) > 0 and 'x' in midfield_duels.columns:
        return midfield_duels['x'].mean()
    return None


def compute_midfield_duel_locations_x_std(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Standard deviation of x-coordinates of duel locations."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_duels) > 0 and 'x' in midfield_duels.columns:
        return midfield_duels['x'].std()
    return None


def compute_midfield_duel_locations_y_mean(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean y-coordinate of duel locations."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_duels) > 0 and 'y' in midfield_duels.columns:
        return midfield_duels['y'].mean()
    return None


def compute_midfield_duel_locations_y_std(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Standard deviation of y-coordinates of duel locations."""
    midfield_events = _get_midfield_events(events, midfield_x_min, midfield_x_max)
    midfield_duels = midfield_events[
        (midfield_events['type_name'] == 'Duel') &
        (midfield_events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(midfield_duels) > 0 and 'y' in midfield_duels.columns:
        return midfield_duels['y'].std()
    return None


def compute_defensive_action_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0,
    midfield_y_central_min: float = 30.0,
    midfield_y_central_max: float = 50.0
) -> Dict[str, float]:
    """
    Compute all defensive action features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, possession_team.id,
        x, y, timestamp, possession, interception.outcome.name,
        ball_recovery.recovery_failure, duel.type.name, duel.outcome.name,
        under_pressure, related_events, pass.length, pass.end_location
    team_id : int
        Team ID for Barcelona (default: 217)
    midfield_x_min : float
        Minimum x-coordinate for midfield (default: 40.0)
    midfield_x_max : float
        Maximum x-coordinate for midfield (default: 80.0)
    midfield_y_central_min : float
        Minimum y-coordinate for central midfield (default: 30.0)
    midfield_y_central_max : float
        Maximum y-coordinate for central midfield (default: 50.0)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of feature names and values
    """
    features = {}
    
    features['midfield_interceptions_total'] = compute_midfield_interceptions_total(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_interceptions_per_possession'] = compute_midfield_interceptions_per_possession(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_interception_success_rate'] = compute_midfield_interception_success_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_interceptions_central'] = compute_midfield_interceptions_central(
        events, team_id, midfield_x_min, midfield_x_max, midfield_y_central_min, midfield_y_central_max
    )
    
    features['midfield_interceptions_wide'] = compute_midfield_interceptions_wide(
        events, team_id, midfield_x_min, midfield_x_max, midfield_y_central_min, midfield_y_central_max
    )
    
    features['midfield_interceptions_progressive'] = compute_midfield_interceptions_progressive(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_interception_time_to_event'] = compute_midfield_interception_time_to_event(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_recoveries_total'] = compute_midfield_recoveries_total(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_recovery_rate'] = compute_midfield_recovery_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_recovery_success_rate'] = compute_midfield_recovery_success_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_recoveries_after_pressure'] = compute_midfield_recoveries_after_pressure(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_recovery_locations_x'] = compute_midfield_recovery_locations_x(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_recovery_locations_y'] = compute_midfield_recovery_locations_y(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_recovery_time_to_event'] = compute_midfield_recovery_time_to_event(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_duels_total'] = compute_midfield_duels_total(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_duel_win_rate'] = compute_midfield_duel_win_rate(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_aerial_duels_won'] = compute_midfield_aerial_duels_won(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_ground_duels_won'] = compute_midfield_ground_duels_won(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_duels_under_pressure'] = compute_midfield_duels_under_pressure(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_duel_locations_x_mean'] = compute_midfield_duel_locations_x_mean(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_duel_locations_x_std'] = compute_midfield_duel_locations_x_std(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_duel_locations_y_mean'] = compute_midfield_duel_locations_y_mean(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_duel_locations_y_std'] = compute_midfield_duel_locations_y_std(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
