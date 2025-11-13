"""
Midfielder-level feature computation registry.

Each function returns a pandas Series indexed by player_id for the current match.
"""

from . import possession_tempo
from . import passing
from . import carrying
from . import defensive
from . import spatial
from . import pressure_resistance
from . import link_play
from . import shot_creation
from . import discipline
from . import set_pieces
from . import progression
from . import duels
from . import attacking_creation
from . import receiving

# Mapping from attribute name in features.csv to implementation function.
FEATURE_FUNCTIONS = {
    # Possession & Tempo
    "possessions_involved": possession_tempo.possessions_involved,
    "possession_time_seconds": possession_tempo.possession_time_seconds,
    "tempo_index": possession_tempo.tempo_index,
    "turnovers": possession_tempo.turnovers,
    # Passing Quality
    "passes_attempted": passing.passes_attempted,
    "pass_completion_rate": passing.pass_completion_rate,
    "progressive_passes": passing.progressive_passes,
    "final_third_entries_by_pass": passing.final_third_entries_by_pass,
    "key_passes": passing.key_passes,
    "under_pressure_pass_share": passing.under_pressure_pass_share,
    # Carrying & Dribbling
    "carries_attempted": carrying.carries_attempted,
    "progressive_carries": carrying.progressive_carries,
    "carry_distance_total": carrying.carry_distance_total,
    "successful_dribbles": carrying.successful_dribbles,
    "carries_leading_to_shot": carrying.carries_leading_to_shot,
    "carries_leading_to_key_pass": carrying.carries_leading_to_key_pass,
    "final_third_carries": carrying.final_third_carries,
    "penalty_area_carries": carrying.penalty_area_carries,
    "pressured_carry_success_rate": carrying.pressured_carry_success_rate,
    # Defensive Contribution
    "pressures_applied": defensive.pressures_applied,
    "ball_recoveries": defensive.ball_recoveries,
    "interceptions": defensive.interceptions,
    "tackles_won": defensive.tackles_won,
    "press_to_interception_chain": defensive.press_to_interception_chain,
    "counterpress_actions": defensive.counterpress_actions,
    "pressure_to_self_recovery": defensive.pressure_to_self_recovery,
    "blocked_passes": defensive.blocked_passes,
    "blocked_shots": defensive.blocked_shots,
    "clearance_followed_by_recovery": defensive.clearance_followed_by_recovery,
    "pressures_to_turnover_rate": defensive.pressures_to_turnover_rate,
    # Spatial Control
    "average_position_x": spatial.average_position_x,
    "average_position_y": spatial.average_position_y,
    "width_variance": spatial.width_variance,
    "zone_entries": spatial.zone_entries,
    # Pressure Resistance
    "pressured_touches": pressure_resistance.pressured_touches,
    "pressured_touch_retention_rate": pressure_resistance.pressured_touch_retention_rate,
    # Link Play
    "third_man_runs": link_play.third_man_runs,
    "wall_pass_events": link_play.wall_pass_events,
    # Shot Creation
    "shot_creating_actions": shot_creation.shot_creating_actions,
    "expected_threat_added": shot_creation.expected_threat_added,
    # Discipline
    "fouls_committed": discipline.fouls_committed,
    "fouls_suffered": discipline.fouls_suffered,
    "tactical_fouls": discipline.tactical_fouls,
    "advantage_fouls_won": discipline.advantage_fouls_won,
    # Set Pieces
    "set_piece_involvements": set_pieces.set_piece_involvements,
    "corner_delivery_accuracy": set_pieces.corner_delivery_accuracy,
    "set_piece_duels_won": set_pieces.set_piece_duels_won,
    "defensive_set_piece_clearances": set_pieces.defensive_set_piece_clearances,
    # Progression & Final Third
    "line_breaking_receipts": progression.line_breaking_receipts,
    "zone14_touches": progression.zone14_touches,
    "penalty_area_deliveries": progression.penalty_area_deliveries,
    "switches_completed": progression.switches_completed,
    "cross_accuracy": progression.cross_accuracy,
    # Duels & Aerial
    "aerial_duels_contested": duels.aerial_duels_contested,
    "aerial_duel_win_rate": duels.aerial_duel_win_rate,
    "fifty_fiftys_won": duels.fifty_fiftys_won,
    "sliding_tackles": duels.sliding_tackles,
    "sliding_tackle_success_rate": duels.sliding_tackle_success_rate,
    # Receiving & On-Ball Security
    "ball_receipts_total": receiving.ball_receipts_total,
    "central_lane_receipts": receiving.central_lane_receipts,
    "one_touch_passes": receiving.one_touch_passes,
    "weak_foot_pass_share": receiving.weak_foot_pass_share,
    "pressured_retention_rate": receiving.pressured_retention_rate,
    # Attacking Creation
    "secondary_shot_assists": attacking_creation.secondary_shot_assists,
    "expected_assists": attacking_creation.expected_assists,
    "xg_chain": attacking_creation.xg_chain,
}

__all__ = ["FEATURE_FUNCTIONS"]

