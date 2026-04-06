"""
Midfielder-level feature computation registry.

Each function returns a pandas Series indexed by player_id for the current match.

Features removed in the Wyscout transition (not available in Wyscout data):
  Pressure-based:  pressures_applied, counterpress_actions, press_to_interception_chain,
                   pressure_to_self_recovery, pressures_to_turnover_rate,
                   under_pressure_pass_share, pressured_carry_success_rate,
                   pressured_touches, pressured_touch_retention_rate, pressured_retention_rate,
                   transition_pressure_rate, press_on_deep_opp_possession
  xG-based:        expected_assists, xg_chain, expected_threat_added
  Ball Receipt:    ball_receipts_total, central_lane_receipts, one_touch_passes
  No equivalent:   blocked_passes, blocked_shots, fifty_fiftys_won, advantage_fouls_won
"""

from . import possession_tempo
from . import passing
from . import carrying
from . import defensive
from . import spatial
from . import link_play
from . import shot_creation
from . import discipline
from . import set_pieces
from . import progression
from . import duels
from . import attacking_creation
from . import receiving
from . import defensive_phase

FEATURE_FUNCTIONS = {
    # Possession & Tempo
    "possessions_involved":              possession_tempo.possessions_involved,
    "possession_time_seconds":           possession_tempo.possession_time_seconds,
    "tempo_index":                       possession_tempo.tempo_index,
    "turnovers":                         possession_tempo.turnovers,
    # Passing Quality
    "passes_attempted":                  passing.passes_attempted,
    "pass_completion_rate":              passing.pass_completion_rate,
    "progressive_passes":                passing.progressive_passes,
    "final_third_entries_by_pass":       passing.final_third_entries_by_pass,
    "key_passes":                        passing.key_passes,
    # Carrying & Dribbling
    "carries_attempted":                 carrying.carries_attempted,
    "progressive_carries":               carrying.progressive_carries,
    "carry_distance_total":              carrying.carry_distance_total,
    "successful_dribbles":               carrying.successful_dribbles,
    "carries_leading_to_shot":           carrying.carries_leading_to_shot,
    "carries_leading_to_key_pass":       carrying.carries_leading_to_key_pass,
    "final_third_carries":               carrying.final_third_carries,
    "penalty_area_carries":              carrying.penalty_area_carries,
    # Defensive Contribution
    "ball_recoveries":                   defensive.ball_recoveries,
    "interceptions":                     defensive.interceptions,
    "tackles_won":                       defensive.tackles_won,
    "clearance_followed_by_recovery":    defensive.clearance_followed_by_recovery,
    # Spatial Control
    "average_position_x":                spatial.average_position_x,
    "average_position_y":                spatial.average_position_y,
    "width_variance":                    spatial.width_variance,
    "zone_entries":                      spatial.zone_entries,
    # Shot Creation
    "shot_creating_actions":             shot_creation.shot_creating_actions,
    # Discipline
    "fouls_committed":                   discipline.fouls_committed,
    "tactical_fouls":                    discipline.tactical_fouls,
    # Set Pieces
    "set_piece_involvements":            set_pieces.set_piece_involvements,
    "corner_delivery_accuracy":          set_pieces.corner_delivery_accuracy,
    "set_piece_duels_won":               set_pieces.set_piece_duels_won,
    "defensive_set_piece_clearances":    set_pieces.defensive_set_piece_clearances,
    # Progression & Final Third
    "zone14_touches":                    progression.zone14_touches,
    "penalty_area_deliveries":           progression.penalty_area_deliveries,
    "switches_completed":                progression.switches_completed,
    "cross_accuracy":                    progression.cross_accuracy,
    # Duels & Aerial
    "aerial_duels_contested":            duels.aerial_duels_contested,
    "aerial_duel_win_rate":              duels.aerial_duel_win_rate,
    "sliding_tackles":                   duels.sliding_tackles,
    "sliding_tackle_success_rate":       duels.sliding_tackle_success_rate,
    # Receiving & On-Ball Security
    "weak_foot_pass_share":              receiving.weak_foot_pass_share,
    # Link Play
    "third_man_runs":                    link_play.third_man_runs,
    "wall_pass_events":                  link_play.wall_pass_events,
    # Attacking Creation
    "secondary_shot_assists":            attacking_creation.secondary_shot_assists,
    # Defensive Phase
    "defensive_midfield_actions":        defensive_phase.defensive_midfield_actions,
    "midfield_zone_coverage_x":          defensive_phase.midfield_zone_coverage_x,
    "midfield_presence_on_deep_opp":     defensive_phase.midfield_presence_on_deep_opp_possession,
    "bypass_channel_defensive_actions":  defensive_phase.bypass_channel_defensive_actions,
    "avg_defensive_x_on_deep_opp":       defensive_phase.avg_defensive_x_on_deep_opp,
    "defensive_shape_compactness":       defensive_phase.defensive_shape_compactness,
}

__all__ = ["FEATURE_FUNCTIONS"]
