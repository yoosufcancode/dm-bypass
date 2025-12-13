# Selected Features

This document lists the features selected for analysis based on the EDA review.

## Possession & Tempo

### possessions_involved
- **Category**: Possession & Tempo
- **Description**: Number of Barcelona possessions where player touches the ball
- **StatBomb Columns**: `type.name`, `player.id`, `possession`
- **Aggregation**: count unique possessions for player.id

### possession_time_seconds
- **Category**: Possession & Tempo
- **Description**: Total on-ball time inferred from sequential touches
- **StatBomb Columns**: `timestamp`, `player.id`, `possession`
- **Aggregation**: sum(diff timestamps within same possession for player.id, cap gaps at 10s)

### tempo_index
- **Category**: Possession & Tempo
- **Description**: Passes + carries per on-ball minute (higher = faster involvement)
- **StatBomb Columns**: `type.name`, `player.id`, `possession`
- **Aggregation**: ((passes + carries) / (possession_time_seconds / 60))

## Passing Quality

### passes_attempted
- **Category**: Passing Quality
- **Description**: Total passes
- **StatBomb Columns**: `type.name == 'Pass'`, `player.id`
- **Aggregation**: count rows

### pass_completion_rate
- **Category**: Passing Quality
- **Description**: Completion percentage
- **StatBomb Columns**: `pass.outcome.name`, `player.id`
- **Aggregation**: completed / attempted

### progressive_passes
- **Category**: Passing Quality
- **Description**: Forward passes gaining ≥10m
- **StatBomb Columns**: `pass.end_location`, `location`, `player.id`
- **Aggregation**: count where end_x - start_x >= 10

### final_third_entries_by_pass
- **Category**: Passing Quality
- **Description**: Passes ending in final third (x > 80)
- **StatBomb Columns**: `pass.end_location`, `player.id`
- **Aggregation**: count rows

### key_passes
- **Category**: Passing Quality
- **Description**: Passes leading to shots
- **StatBomb Columns**: `pass.shot_assist == True` or `pass.goal_assist == True`
- **Aggregation**: count rows

### under_pressure_pass_share
- **Category**: Passing Quality
- **Description**: Share of passes attempted while under pressure
- **StatBomb Columns**: `under_pressure`, `type.name == 'Pass'`, `player.id`
- **Aggregation**: passes_under_pressure / passes_attempted

## Carrying & Dribbling

### carries_attempted
- **Category**: Carrying & Dribbling
- **Description**: Total carries
- **StatBomb Columns**: `type.name == 'Carry'`, `player.id`
- **Aggregation**: count rows

### final_third_carries
- **Category**: Carrying & Dribbling
- **Description**: Carries entering final third (end_x > 80)
- **StatBomb Columns**: `type.name == 'Carry'`, `carry.end_location`
- **Aggregation**: count rows

### penalty_area_carries
- **Category**: Carrying & Dribbling
- **Description**: Carries ending inside penalty area
- **StatBomb Columns**: `type.name == 'Carry'`, `carry.end_location`
- **Aggregation**: count rows with end_x ≥ 102

### progressive_carries
- **Category**: Carrying & Dribbling
- **Description**: Carries adding ≥10m towards goal
- **StatBomb Columns**: `carry.end_location`, `location`, `player.id`
- **Aggregation**: count where end_x - start_x >= 10

### carry_distance_total
- **Category**: Carrying & Dribbling
- **Description**: Total distance carried
- **StatBomb Columns**: `location`, `carry.end_location`
- **Aggregation**: sum Euclidean distance per carry

### pressured_carry_success_rate
- **Category**: Carrying & Dribbling
- **Description**: Success of carries attempted under pressure
- **StatBomb Columns**: `type.name == 'Carry'`, `under_pressure`, `subsequent outcome`
- **Aggregation**: successful_pressured_carries / pressured_carries

## Defensive Contribution

### pressures_applied
- **Category**: Defensive Contribution
- **Description**: Opposition events pressured
- **StatBomb Columns**: `under_pressure == True AND possession_team.id != team_id`, `player.id`
- **Aggregation**: count rows where player is defensive team

### ball_recoveries
- **Category**: Defensive Contribution
- **Description**: Ball recoveries credited
- **StatBomb Columns**: `type.name == 'Ball Recovery'`, `player.id`
- **Aggregation**: count rows

### interceptions
- **Category**: Defensive Contribution
- **Description**: Interceptions credited
- **StatBomb Columns**: `type.name == 'Interception'`, `player.id`
- **Aggregation**: count rows

### tackles_won
- **Category**: Defensive Contribution
- **Description**: Tackles with successful outcome
- **StatBomb Columns**: `type.name == 'Duel'`, `duel.type.name == 'Tackle'` and `duel.outcome.name` contains 'Won', `player.id`
- **Aggregation**: count rows

### counterpress_actions
- **Category**: Defensive Contribution
- **Description**: Counterpress defensive actions
- **StatBomb Columns**: `counterpress == True`, `player.id`
- **Aggregation**: count rows

### press_to_interception_chain
- **Category**: Defensive Contribution
- **Description**: Pressure events by player that lead to team interception within 5s
- **StatBomb Columns**: `under_pressure`, `possession`, `timestamp`
- **Aggregation**: count possessions where player pressure precedes team interception

### pressure_to_self_recovery
- **Category**: Defensive Contribution
- **Description**: Player pressures leading to own recovery within 5s
- **StatBomb Columns**: `under_pressure`, `Ball Recovery`, `timestamp`
- **Aggregation**: count sequences

### clearance_followed_by_recovery
- **Category**: Defensive Contribution
- **Description**: Clearances by player with team recovery inside 5s
- **StatBomb Columns**: `type.name == 'Clearance'`, `same possession Ball Recovery`
- **Aggregation**: count sequences

### pressures_to_turnover_rate
- **Category**: Defensive Contribution
- **Description**: Pressure events forcing turnover within 3s
- **StatBomb Columns**: `under_pressure`, `possession change`
- **Aggregation**: forced_turnovers / pressures_applied

## Progression & Final Third

### line_breaking_receipts
- **Category**: Progression & Final Third
- **Description**: Receipts that break opponent lines (start_x < 40, receipt_x ≥ 40)
- **StatBomb Columns**: `type.name == 'Ball Receipt*'`, `location[0]`, `possession path`
- **Aggregation**: count rows meeting criteria

### zone14_touches
- **Category**: Progression & Final Third
- **Description**: Touches in zone 14 (78 ≤ x ≤ 102, 35 ≤ y ≤ 55)
- **StatBomb Columns**: `location`, `player.id`
- **Aggregation**: count rows in zone

### penalty_area_deliveries
- **Category**: Progression & Final Third
- **Description**: Passes ending inside penalty area
- **StatBomb Columns**: `type.name == 'Pass'`, `pass.end_location`
- **Aggregation**: count rows with end_x ≥ 102 and 18-yard y bounds

### switches_completed
- **Category**: Progression & Final Third
- **Description**: Successful switches executed by player
- **StatBomb Columns**: `type.name == 'Pass'`, `pass.switch == True`
- **Aggregation**: count completed switches

### cross_accuracy
- **Category**: Progression & Final Third
- **Description**: Cross completion rate
- **StatBomb Columns**: `type.name == 'Pass'`, `pass.cross == True`, `pass.outcome`
- **Aggregation**: completed_crosses / attempted_crosses

## Duels & Aerial

### aerial_duels_contested
- **Category**: Duels & Aerial
- **Description**: Aerial duels participated in
- **StatBomb Columns**: `type.name == 'Duel'`, `duel.type.name == 'Aerial Lost/Won'`
- **Aggregation**: count rows

### aerial_duel_win_rate
- **Category**: Duels & Aerial
- **Description**: Percentage of aerial duels won
- **StatBomb Columns**: `duel.type.name == 'Aerial Lost/Won'`, `duel.outcome.name`
- **Aggregation**: wins / contested

### fifty_fiftys_won
- **Category**: Duels & Aerial
- **Description**: 50/50 ground balls won
- **StatBomb Columns**: `type.name == '50/50'`, `outcome`
- **Aggregation**: count won rows

## Spatial Control

### average_position_x
- **Category**: Spatial Control
- **Description**: Mean x of player's events in open play
- **StatBomb Columns**: `location[0]`, `player.id`
- **Aggregation**: mean

### average_position_y
- **Category**: Spatial Control
- **Description**: Mean y location
- **StatBomb Columns**: `location[1]`, `player.id`
- **Aggregation**: mean

### width_variance
- **Category**: Spatial Control
- **Description**: Spread of lateral positioning
- **StatBomb Columns**: `location[1]`, `player.id`
- **Aggregation**: variance

### zone_entries
- **Category**: Spatial Control
- **Description**: Entries into central lane (35 ≤ y ≤ 45) by action type
- **StatBomb Columns**: `location[1]`, `type.name`, `player.id`
- **Aggregation**: count events with y between 35 and 45

## Receiving & On-Ball Security

### ball_receipts_total
- **Category**: Receiving & On-Ball Security
- **Description**: Ball receipts credited to player
- **StatBomb Columns**: `type.name == 'Ball Receipt*'`, `player.id`
- **Aggregation**: count rows

### central_lane_receipts
- **Category**: Receiving & On-Ball Security
- **Description**: Receipts in central lane (35 ≤ y ≤ 45)
- **StatBomb Columns**: `type.name == 'Ball Receipt*'`, `location[1]`
- **Aggregation**: count rows with 35 ≤ y ≤ 45

### one_touch_passes
- **Category**: Receiving & On-Ball Security
- **Description**: Ground passes where next team touch follows within 1s without carry
- **StatBomb Columns**: `type.name == 'Pass'`, `pass.height.name == 'Ground Pass'`, `possession`
- **Aggregation**: count qualifying sequences

### pressured_touches
- **Category**: Receiving & On-Ball Security
- **Description**: Player touches recorded while under pressure
- **StatBomb Columns**: `under_pressure`, `player.id`
- **Aggregation**: count rows

### pressured_retention_rate
- **Category**: Receiving & On-Ball Security
- **Description**: Share of pressured touches retained within 2 seconds
- **StatBomb Columns**: `under_pressure`, `subsequent event outcome`, `player.id`
- **Aggregation**: retained_pressured_touches / pressured_touches

### pressured_touch_retention_rate
- **Category**: Receiving & On-Ball Security
- **Description**: Share of pressured touches not lost within 2s
- **StatBomb Columns**: `under_pressure`, `type.name`, `next event outcome`
- **Aggregation**: retained / pressured_touches

## Link Play

### third_man_runs
- **Category**: Link Play
- **Description**: Carries or passes immediately following teammate pass from same possession to switch flank
- **StatBomb Columns**: `possession`, `pass.recipient.id`, `carry`
- **Aggregation**: count qualifying sequences

### wall_pass_events
- **Category**: Link Play
- **Description**: Give-and-go sequences where player initiates and receives within 3s
- **StatBomb Columns**: `possession`, `timestamp`, `pass`
- **Aggregation**: count sequences

## Attacking Creation

### shot_creating_actions
- **Category**: Attacking Creation
- **Description**: Two-event sequence ending in shot (pass/carry/foul won)
- **StatBomb Columns**: StatsBomb SCA manually derived
- **Aggregation**: count

### secondary_shot_assists
- **Category**: Attacking Creation
- **Description**: Pre-assist events (second to last action before shot)
- **StatBomb Columns**: `shot.key_pass_id chain`, `possession`
- **Aggregation**: count

### expected_assists
- **Category**: Attacking Creation
- **Description**: Sum of xA on passes
- **StatBomb Columns**: `pass.shot_assist == True`, `shot.statsbomb_xg`
- **Aggregation**: sum xA

### xg_chain
- **Category**: Attacking Creation
- **Description**: Player involvement in possessions ending in shot (xG chain)
- **StatBomb Columns**: `possession`, `shot.statsbomb_xg`, `player involvement`
- **Aggregation**: sum xGChain

## Discipline

### fouls_committed
- **Category**: Discipline
- **Description**: Fouls conceded by player
- **StatBomb Columns**: `type.name == 'Foul Won'` for opponent, `player.id`
- **Aggregation**: count rows

### fouls_suffered
- **Category**: Discipline
- **Description**: Fouls won by player
- **StatBomb Columns**: `type.name == 'Foul Won'`, `player.id`
- **Aggregation**: count rows

### advantage_fouls_won
- **Category**: Discipline
- **Description**: Fouls where advantage applied
- **StatBomb Columns**: `foul_won.advantage == True`
- **Aggregation**: count rows

## Set Pieces

### set_piece_involvements
- **Category**: Set Pieces
- **Description**: Restarts taken or contested (corners, free kicks)
- **StatBomb Columns**: `play_pattern.name`, `type.name`, `player.id`
- **Aggregation**: count events

### corner_delivery_accuracy
- **Category**: Set Pieces
- **Description**: Completed corners to own teammate
- **StatBomb Columns**: `type.name == 'Pass'`, `pass.type.name == 'Corner'`, `pass.outcome`
- **Aggregation**: completed / attempted corners

### set_piece_duels_won
- **Category**: Set Pieces
- **Description**: Aerial/ground duels during defensive set pieces
- **StatBomb Columns**: `play_pattern.name` in set pieces, `duel outcomes`
- **Aggregation**: count wins

### defensive_set_piece_clearances
- **Category**: Set Pieces
- **Description**: Clearances within 10s of opponent set piece
- **StatBomb Columns**: `play_pattern.name`, `type.name == 'Clearance'`
- **Aggregation**: count rows

---

## Summary

**Total Selected Features**: 55

- Possession & Tempo: 3
- Passing Quality: 6
- Carrying & Dribbling: 6
- Defensive Contribution: 9
- Progression & Final Third: 5
- Duels & Aerial: 3
- Spatial Control: 4
- Receiving & On-Ball Security: 6
- Link Play: 2
- Attacking Creation: 4
- Discipline: 3
- Set Pieces: 4

