"""
Feature engineering modules for midfield strength analysis.

Each module computes features for a specific category of midfield analysis.
"""

from .defensive_actions import compute_defensive_action_features
from .pressure_tempo import compute_pressure_tempo_features
from .access_control import compute_access_control_features
from .spatial_compactness import compute_spatial_compactness_features

try:
    from .passing_features import compute_passing_features
except ImportError:
    compute_passing_features = None

try:
    from .carrying_features import compute_carrying_features
except ImportError:
    compute_carrying_features = None

try:
    from .recovery_transition import compute_recovery_transition_features
except ImportError:
    compute_recovery_transition_features = None

try:
    from .temporal_features import compute_temporal_features
except ImportError:
    compute_temporal_features = None

try:
    from .zone_specific import compute_zone_specific_features
except ImportError:
    compute_zone_specific_features = None

try:
    from .player_tactical import compute_player_tactical_features
except ImportError:
    compute_player_tactical_features = None

try:
    from .composite_features import compute_composite_features
except ImportError:
    compute_composite_features = None

try:
    from .contextual_features import compute_contextual_features
except ImportError:
    compute_contextual_features = None

__all__ = [
    'compute_defensive_action_features',
    'compute_pressure_tempo_features',
    'compute_access_control_features',
    'compute_spatial_compactness_features',
]

