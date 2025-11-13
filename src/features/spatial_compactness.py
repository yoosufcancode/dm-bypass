"""
Spatial and Compactness Features

Computes features related to defensive shape, compactness, and spatial distribution.
Each feature is implemented as a separate function for individual evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
try:
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max):
    """Helper to get defensive events in midfield."""
    return events[
        (events.get('team.id', pd.Series()) == team_id) &
        (events['x'] >= midfield_x_min) & 
        (events['x'] <= midfield_x_max) &
        events['x'].notna() &
        (events['type_name'].isin(['Interception', 'Ball Recovery', 'Duel', 'Block', 'Clearance']))
    ].copy()


def compute_midfield_defensive_width(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average defensive width (max_y - min_y) per possession."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'possession' in defensive_events.columns and 'y' in defensive_events.columns:
        width_stats = defensive_events.groupby('possession')['y'].agg(['max', 'min'])
        widths = (width_stats['max'] - width_stats['min'])
        return widths.mean()
    return None


def compute_midfield_defensive_depth(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Average defensive depth (max_x - min_x) per possession."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'possession' in defensive_events.columns and 'x' in defensive_events.columns:
        depth_stats = defensive_events.groupby('possession')['x'].agg(['max', 'min'])
        depths = (depth_stats['max'] - depth_stats['min'])
        return depths.mean()
    return None


def compute_midfield_compactness_index(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Compactness index (ratio of width to depth)."""
    width = compute_midfield_defensive_width(events, team_id, midfield_x_min, midfield_x_max)
    depth = compute_midfield_defensive_depth(events, team_id, midfield_x_min, midfield_x_max)
    
    if width is not None and depth is not None and depth > 0:
        return width / depth
    return None


def compute_midfield_central_concentration(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Proportion of defensive actions in central zone (35 ≤ y ≤ 45)."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'y' in defensive_events.columns:
        central_actions = defensive_events[
            (defensive_events['y'] >= 35) & (defensive_events['y'] <= 45)
        ]
        return len(central_actions) / len(defensive_events)
    return 0.0


def compute_midfield_player_density(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Average defensive events per possession (proxy for player density)."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if 'possession' in defensive_events.columns and len(defensive_events) > 0:
        events_per_possession = defensive_events.groupby('possession').size()
        return events_per_possession.mean()
    return 0.0


def compute_midfield_left_zone_coverage(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of defensive actions in left zone (y < 40)."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'y' in defensive_events.columns:
        return len(defensive_events[defensive_events['y'] < 40])
    return 0


def compute_midfield_central_zone_coverage(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of defensive actions in central zone (35 ≤ y ≤ 45)."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'y' in defensive_events.columns:
        return len(defensive_events[
            (defensive_events['y'] >= 35) & (defensive_events['y'] <= 45)
        ])
    return 0


def compute_midfield_right_zone_coverage(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of defensive actions in right zone (y > 40)."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'y' in defensive_events.columns:
        return len(defensive_events[defensive_events['y'] > 40])
    return 0


def compute_midfield_zone_balance(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Standard deviation of zone coverage (lower = more balanced)."""
    left = compute_midfield_left_zone_coverage(events, team_id, midfield_x_min, midfield_x_max)
    center = compute_midfield_central_zone_coverage(events, team_id, midfield_x_min, midfield_x_max)
    right = compute_midfield_right_zone_coverage(events, team_id, midfield_x_min, midfield_x_max)
    
    zone_counts = [left, center, right]
    return np.std(zone_counts)


def compute_midfield_coverage_gaps_count(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> int:
    """Number of grid zones with coverage below threshold."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'y' in defensive_events.columns:
        x_bins = [midfield_x_min, 53.33, 66.67, midfield_x_max]
        y_bins = [0, 26.67, 53.33, 80]
        coverage_grid = np.zeros((3, 3))
        
        for _, action in defensive_events.iterrows():
            x_idx = np.digitize(action['x'], x_bins) - 1
            y_idx = np.digitize(action['y'], y_bins) - 1
            if 0 <= x_idx < 3 and 0 <= y_idx < 3:
                coverage_grid[x_idx, y_idx] += 1
        
        mean_coverage = coverage_grid.mean()
        threshold = mean_coverage * 0.05 if mean_coverage > 0 else 0
        gaps = np.where(coverage_grid < threshold)
        return len(gaps[0])
    return 0


def compute_midfield_defensive_actions_x_mean(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean x-coordinate of defensive actions."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'x' in defensive_events.columns:
        return defensive_events['x'].mean()
    return None


def compute_midfield_defensive_actions_x_std(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Standard deviation of x-coordinates of defensive actions."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'x' in defensive_events.columns:
        return defensive_events['x'].std()
    return None


def compute_midfield_defensive_actions_y_mean(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Mean y-coordinate of defensive actions."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'y' in defensive_events.columns:
        return defensive_events['y'].mean()
    return None


def compute_midfield_defensive_actions_y_std(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Optional[float]:
    """Standard deviation of y-coordinates of defensive actions."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 0 and 'y' in defensive_events.columns:
        return defensive_events['y'].std()
    return None


def compute_midfield_defensive_actions_clustering(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> float:
    """Spatial clustering coefficient (inverse of mean nearest neighbor distance)."""
    defensive_events = _get_defensive_events(events, team_id, midfield_x_min, midfield_x_max)
    
    if len(defensive_events) > 1 and 'x' in defensive_events.columns and 'y' in defensive_events.columns:
        coords = defensive_events[['x', 'y']].values
        
        if SCIPY_AVAILABLE:
            distances = cdist(coords, coords)
            np.fill_diagonal(distances, np.inf)
            nearest_neighbor_distances = distances.min(axis=1)
            mean_nn_distance = nearest_neighbor_distances.mean()
        else:
            min_distances = []
            for i in range(len(coords)):
                distances = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
                distances[i] = np.inf
                min_distances.append(distances.min())
            mean_nn_distance = np.mean(min_distances)
        
        if mean_nn_distance > 0:
            return 1 / (mean_nn_distance + 1e-6)
    return 0.0


def compute_spatial_compactness_features(
    events: pd.DataFrame,
    team_id: int = 217,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> Dict[str, float]:
    """
    Compute all spatial and compactness features for midfield strength analysis.
    
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
    
    features['midfield_defensive_width'] = compute_midfield_defensive_width(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_defensive_depth'] = compute_midfield_defensive_depth(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_compactness_index'] = compute_midfield_compactness_index(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_central_concentration'] = compute_midfield_central_concentration(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_player_density'] = compute_midfield_player_density(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_left_zone_coverage'] = compute_midfield_left_zone_coverage(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_central_zone_coverage'] = compute_midfield_central_zone_coverage(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_right_zone_coverage'] = compute_midfield_right_zone_coverage(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_zone_balance'] = compute_midfield_zone_balance(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_coverage_gaps_count'] = compute_midfield_coverage_gaps_count(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_defensive_actions_x_mean'] = compute_midfield_defensive_actions_x_mean(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_defensive_actions_x_std'] = compute_midfield_defensive_actions_x_std(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_defensive_actions_y_mean'] = compute_midfield_defensive_actions_y_mean(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_defensive_actions_y_std'] = compute_midfield_defensive_actions_y_std(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    features['midfield_defensive_actions_clustering'] = compute_midfield_defensive_actions_clustering(
        events, team_id, midfield_x_min, midfield_x_max
    )
    
    return features
