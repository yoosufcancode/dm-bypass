"""
Player Position and Role Features

Computes features related to individual player performance and tactical formation.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max):
    """Helper to get Barcelona events in midfield."""
    return events[
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna()
    ].copy()


def compute_midfielder_interceptions_per_player(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict:
    """Dictionary of interceptions per player ID."""
    barca_midfield_events = _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if 'player.id' in barca_midfield_events.columns:
        midfield_players = barca_midfield_events[
            barca_midfield_events['player.id'].notna()
        ]['player.id'].unique()
        
        if len(midfield_players) > 0:
            interceptions = barca_midfield_events[
                barca_midfield_events['type_name'] == 'Interception'
            ]
            interceptions_per_player = interceptions.groupby('player.id').size()
            return interceptions_per_player.to_dict() if len(interceptions_per_player) > 0 else {}
    return {}


def compute_midfielder_recoveries_per_player(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict:
    """Dictionary of recoveries per player ID."""
    barca_midfield_events = _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if 'player.id' in barca_midfield_events.columns:
        midfield_players = barca_midfield_events[
            barca_midfield_events['player.id'].notna()
        ]['player.id'].unique()
        
        if len(midfield_players) > 0:
            recoveries = barca_midfield_events[
                barca_midfield_events['type_name'] == 'Ball Recovery'
            ]
            recoveries_per_player = recoveries.groupby('player.id').size()
            return recoveries_per_player.to_dict() if len(recoveries_per_player) > 0 else {}
    return {}


def compute_midfielder_pressures_per_player(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict:
    """Dictionary of pressures per player ID."""
    barca_midfield_events = _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if 'player.id' in barca_midfield_events.columns:
        midfield_players = barca_midfield_events[
            barca_midfield_events['player.id'].notna()
        ]['player.id'].unique()
        
        if len(midfield_players) > 0:
            pressures = barca_midfield_events[
                barca_midfield_events['type_name'] == 'Pressure'
            ]
            pressures_per_player = pressures.groupby('player.id').size()
            return pressures_per_player.to_dict() if len(pressures_per_player) > 0 else {}
    return {}


def compute_midfielder_duel_win_rate_per_player(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict:
    """Dictionary of duel win rate per player ID."""
    barca_midfield_events = _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if 'player.id' in barca_midfield_events.columns:
        midfield_players = barca_midfield_events[
            barca_midfield_events['player.id'].notna()
        ]['player.id'].unique()
        
        if len(midfield_players) > 0:
            duels = barca_midfield_events[
                barca_midfield_events['type_name'] == 'Duel'
            ]
            
            if len(duels) > 0:
                duel_win_rate = {}
                for player_id in midfield_players:
                    player_duels = duels[duels.get('player.id', pd.Series()) == player_id]
                    if len(player_duels) > 0:
                        won_duels = player_duels[
                            player_duels.get('duel.outcome.name', pd.Series()).str.contains('Won', na=False)
                        ]
                        duel_win_rate[player_id] = len(won_duels) / len(player_duels) if len(player_duels) > 0 else 0.0
                return duel_win_rate
    return {}


def compute_midfielder_zone_coverage(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict:
    """Dictionary of zone coverage stats per player ID."""
    barca_midfield_events = _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if 'player.id' in barca_midfield_events.columns:
        midfield_players = barca_midfield_events[
            barca_midfield_events['player.id'].notna()
        ]['player.id'].unique()
        
        if len(midfield_players) > 0:
            zone_coverage = {}
            for player_id in midfield_players:
                player_events = barca_midfield_events[
                    barca_midfield_events.get('player.id', pd.Series()) == player_id
                ]
                if len(player_events) > 0:
                    zone_coverage[player_id] = {
                        'x_mean': player_events['x'].mean(),
                        'x_std': player_events['x'].std(),
                        'y_mean': player_events['y'].mean(),
                        'y_std': player_events['y'].std()
                    }
            return zone_coverage
    return {}


def compute_formation_type(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Formation type from Starting XI events."""
    starting_xi = events[
        (events['type_name'] == 'Starting XI') &
        (events.get('team.id', pd.Series()) == team_id)
    ]
    
    if len(starting_xi) > 0:
        formation = starting_xi.iloc[0].get('tactics.formation')
        if pd.isna(formation):
            formation = starting_xi.iloc[0].get('tactics_formation')
        return formation if pd.notna(formation) else None
    return None


def compute_midfield_player_count(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of unique players with events in midfield."""
    barca_midfield_events = _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if 'player.id' in barca_midfield_events.columns and len(barca_midfield_events) > 0:
        midfield_players = barca_midfield_events[
            barca_midfield_events['player.id'].notna()
        ]['player.id'].unique()
        return len(midfield_players)
    return 0


def compute_midfield_width_utilization(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Standard deviation of y-coordinates (width utilization)."""
    barca_midfield_events = _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(barca_midfield_events) > 0 and 'y' in barca_midfield_events.columns:
        return barca_midfield_events['y'].std()
    return None


def compute_midfield_depth_utilization(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean x-coordinate (depth utilization)."""
    barca_midfield_events = _get_barca_midfield_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(barca_midfield_events) > 0 and 'x' in barca_midfield_events.columns:
        return barca_midfield_events['x'].mean()
    return None


def compute_player_tactical_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all player position and tactical features for midfield strength analysis.
    
    This function calls all individual feature functions and aggregates results.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe with columns: type_name, team.id, player.id, player.name,
        x, y, timestamp, position.name, tactics.formation
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
    
    features['midfielder_interceptions_per_player'] = compute_midfielder_interceptions_per_player(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfielder_recoveries_per_player'] = compute_midfielder_recoveries_per_player(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfielder_pressures_per_player'] = compute_midfielder_pressures_per_player(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfielder_duel_win_rate_per_player'] = compute_midfielder_duel_win_rate_per_player(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfielder_zone_coverage'] = compute_midfielder_zone_coverage(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['formation_type'] = compute_formation_type(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_player_count'] = compute_midfield_player_count(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_width_utilization'] = compute_midfield_width_utilization(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_depth_utilization'] = compute_midfield_depth_utilization(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
