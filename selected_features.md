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
- **Note**: ⚠️ Calculation error detected: Maximum value of 119 per match is humanly impossible. **Root Cause**: Code counts all "Carry" events without validation. May be counting duplicate events or including invalid carries. Need to verify data quality and add filtering logic.

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
- **Note**: ⚠️ Calculation error detected in current implementation. **Root Cause**: Uses `timestamp_seconds` which should be available from context, but may have issues with time window calculation or event matching logic. Needs verification.

### successful_dribbles
- **Category**: Carrying & Dribbling
- **Description**: 1v1 take-ons won
- **StatBomb Columns**: `type.name == 'Duel'`, `duel.type.name == 'Tackle'`, `duel.outcome.name` contains 'Won', `player.id`
- **Aggregation**: count rows
- **Note**: ⚠️ Calculation error detected: All values are zero per match, which is impossible. **Root Cause**: Code incorrectly searches for `type_name == "Take On"` instead of `type_name == "Duel"` with `duel.type.name == "Tackle"`. Logic mismatch between specification and implementation.

### carries_leading_to_shot
- **Category**: Carrying & Dribbling
- **Description**: Carries culminating in team shot
- **StatBomb Columns**: `carry.id` referenced by `shot.carry_id`
- **Aggregation**: count rows
- **Note**: ⚠️ Calculation error detected: All values are zero per match, which is impossible. **Root Cause**: Code checks for `carry.id` and `shot.carry_id` columns. While `shot.carry_id` is extracted in `clean_events()`, the StatsBomb data may not populate this field, or the IDs may not match between carry and shot events. The merge operation fails because no matching IDs are found.

### carries_leading_to_key_pass
- **Category**: Carrying & Dribbling
- **Description**: Carries ending in key pass/assist
- **StatBomb Columns**: `carry.id` referenced by `pass.carry_id`
- **Aggregation**: count rows
- **Note**: ⚠️ Calculation error detected: All values are zero per match, which is impossible. **Root Cause**: Code checks for `pass.carry_id` column which is extracted in `clean_events()`, but the StatsBomb data may not populate this field, or the IDs may not match between carry and pass events. The merge operation fails because no matching IDs are found.

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

### blocked_passes
- **Category**: Defensive Contribution
- **Description**: Pass blocks made
- **StatBomb Columns**: `type.name == 'Block'`, `block.block_type == 'Pass Block'`
- **Aggregation**: count rows
- **Note**: ⚠️ Calculation error detected: All values are zero per match, which is impossible. **Root Cause**: Code uses `get("block.block_type")` which extracts from `block.type.name` in `clean_events()`. The comparison to `"Pass Block"` likely fails because StatsBomb uses different terminology (e.g., "Pass" instead of "Pass Block"). Need to verify actual values in `block.type.name` column.

### blocked_shots
- **Category**: Defensive Contribution
- **Description**: Shots blocked by player
- **StatBomb Columns**: `type.name == 'Block'`, `block.block_type == 'Shot Block'`
- **Aggregation**: count rows
- **Note**: ⚠️ Calculation error detected: All values are zero per match, which is impossible. **Root Cause**: Code uses `get("block.block_type")` which extracts from `block.type.name` in `clean_events()`. The comparison to `"Shot Block"` likely fails because StatsBomb uses different terminology (e.g., "Shot" instead of "Shot Block"). Need to verify actual values in `block.type.name` column.

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
- **Note**: ⚠️ Calculation error detected in current implementation. **Root Cause**: Code uses `str.contains("Aerial", na=False)` which may match incorrectly, or `duel.outcome.name` may not contain "Won" string as expected. Need to verify actual values in StatsBomb data.

### sliding_tackles
- **Category**: Duels & Aerial
- **Description**: Sliding tackles attempted
- **StatBomb Columns**: `duel.tackle == 'Sliding Tackle'`
- **Aggregation**: count rows
- **Note**: ⚠️ Calculation error detected: All values are zero per match, which is impossible. **Root Cause**: Code uses `duels.get("duel.tackle")` which may return None if column doesn't exist. The `duel.tackle` field may not be populated in StatsBomb data, or the exact string value may differ from "Sliding Tackle". Need to verify actual values in the `duel.tackle` column.

### sliding_tackle_success_rate
- **Category**: Duels & Aerial
- **Description**: Success rate of sliding tackles
- **StatBomb Columns**: `duel.tackle == 'Sliding Tackle'`, `duel.outcome.name`
- **Aggregation**: wins / attempts
- **Note**: ⚠️ Calculation error detected: All values are zero/NaN per match because sliding_tackles is zero. Cannot calculate success rate without valid attempts. **Root Cause**: Inherits the issue from `sliding_tackles` - cannot calculate success rate when no sliding tackles are detected due to the same column/value matching problems.

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

### weak_foot_pass_share
- **Category**: Receiving & On-Ball Security
- **Description**: Percentage of passes played with non-dominant foot (needs roster metadata)
- **StatBomb Columns**: `type.name == 'Pass'`, `pass.body_part.name`
- **Aggregation**: passes w/ weak foot / passes_attempted
- **Note**: ⚠️ Calculation error detected: All values are NaN, which is impossible. **Root Cause**: Code intentionally returns NaN for all players (line 114 in receiving.py: `return ctx.players_series(default=np.nan)`). The function is not implemented - it requires roster metadata to determine dominant foot, which is not available in the event data. `pass.body_part.name` exists but cannot determine "weak foot" without knowing player's dominant foot from roster data.

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

### tactical_fouls
- **Category**: Discipline
- **Description**: Fouls labeled tactical or stopping attack
- **StatBomb Columns**: `foul_committed.type.name`, `foul_committed.card.name`
- **Aggregation**: count rows
- **Note**: ⚠️ Calculation error detected: All values are zero per match, which is impossible. **Root Cause**: Code checks for `foul_committed.type.name` values `["Tactical", "Professional Foul"]`, but StatsBomb may use different terminology or these specific type names may not exist in the dataset. Need to verify actual values in `foul_committed.type.name` column.

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

**Total Selected Features**: 64

- Possession & Tempo: 3
- Passing Quality: 6
- Carrying & Dribbling: 9
- Defensive Contribution: 11
- Progression & Final Third: 5
- Duels & Aerial: 5
- Spatial Control: 4
- Receiving & On-Ball Security: 7
- Link Play: 2
- Attacking Creation: 4
- Discipline: 4
- Set Pieces: 4

**Note**: 12 features have calculation errors detected in their current implementation (marked with ⚠️ in their descriptions). Root causes identified:

1. **carries_attempted**: May be counting duplicate/invalid events or missing validation
2. **successful_dribbles**: Logic error - code searches for "Take On" events instead of "Duel" events with tackle type
3. **carries_leading_to_shot**: Missing or unlinked `carry.id`/`shot.carry_id` columns in data
4. **carries_leading_to_key_pass**: Missing or unlinked `pass.carry_id` column in data
5. **blocked_passes**: Column value mismatch - checking for "Pass Block" but StatsBomb may use different terminology
6. **blocked_shots**: Column value mismatch - checking for "Shot Block" but StatsBomb may use different terminology
7. **tactical_fouls**: Value mismatch - checking for ["Tactical", "Professional Foul"] but StatsBomb may use different terminology
8. **sliding_tackles**: Column may not exist or value mismatch - "Sliding Tackle" string may not match StatsBomb data
9. **sliding_tackle_success_rate**: Inherits issue from sliding_tackles
10. **weak_foot_pass_share**: Not implemented - intentionally returns NaN (requires roster metadata for dominant foot)
11. **pressured_carry_success_rate**: Calculation error detected (needs investigation)
12. **aerial_duel_win_rate**: Calculation error detected (needs investigation)

