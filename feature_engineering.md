# Feature Engineering for Midfield Strength Analysis

## Overview

This document outlines all features that can be engineered from the StatsBomb event data to analyze midfield strength and bypass prevention for **Barcelona (team ID: 217)**. The focus is on identifying features that capture the quality and effectiveness of the midfield in preventing opponents from bypassing through passing or carrying.

## Pitch Zones

- **Defensive Third**: x = 0-40
- **Midfield Third**: x = 40-80
- **Final Third**: x = 80-120

## Feature Categories

---

## 1. Defensive Actions in Midfield

### 1.1 Interception Features
**Purpose**: Measure ability to intercept opponent passes in midfield

- **`midfield_interceptions_total`**: Total number of interceptions by Barcelona in midfield (40 ≤ x ≤ 80)
- **`midfield_interceptions_per_possession`**: Average interceptions per opponent possession
- **`midfield_interception_success_rate`**: Percentage of successful interceptions (outcome = "Success In Play")
- **`midfield_interceptions_central`**: Interceptions in central midfield (40 ≤ x ≤ 80, 30 ≤ y ≤ 50)
- **`midfield_interceptions_wide`**: Interceptions in wide midfield areas
- **`midfield_interceptions_progressive`**: Interceptions of progressive passes (passes moving forward >10m by x axis)
- **`midfield_interception_time_to_event`**: Average time from possession start to first interception

**Computation**:
```python
# Filter interceptions in midfield by Barcelona
midfield_interceptions = events[
    (events['type_name'] == 'Interception') &
    (events['team.id'] == 217) &  # Barcelona
    (events['x'] >= 40) & (events['x'] <= 80)
]
```

### 1.2 Ball Recovery Features
**Purpose**: Measure ability to recover possession in midfield

- **`midfield_recoveries_total`**: Total ball recoveries in midfield zone
- **`midfield_recovery_rate`**: Recoveries per minute in midfield
- **`midfield_recovery_success_rate`**: Percentage without `recovery_failure = true`
- **`midfield_recoveries_after_pressure`**: Recoveries following pressure events
- **`midfield_recovery_locations_x`**: Mean x-coordinate of recovery locations
- **`midfield_recovery_locations_y`**: Mean y-coordinate of recovery locations
- **`midfield_recovery_time_to_event`**: Time from possession start to recovery

**Computation**:
```python
midfield_recoveries = events[
    (events['type_name'] == 'Ball Recovery') &
    (events['team.id'] == 217) &
    (events['x'] >= 40) & (events['x'] <= 80)
]
```

### 1.3 Duel Features
**Purpose**: Measure success in one-on-one contests in midfield

- **`midfield_duels_total`**: Total duels in midfield
- **`midfield_duel_win_rate`**: Percentage of duels won (outcome contains "Won")
- **`midfield_aerial_duels_won`**: Aerial duels won in midfield
- **`midfield_ground_duels_won`**: Ground duels won in midfield
- **`midfield_duels_under_pressure`**: Duels won while under pressure
- **`midfield_duel_locations`**: Spatial distribution of duels

**Computation**:
```python
midfield_duels = events[
    (events['type_name'] == 'Duel') &
    (events['team.id'] == 217) &
    (events['x'] >= 40) & (events['x'] <= 80)
]
```

---

## 2. Pressure and Tempo Features

### 2.1 Pressure Intensity Features
**Purpose**: Measure defensive intensity and pressing in midfield

- **`midfield_pressure_events_total`**: Total events where `under_pressure = true` for opponents in midfield
- **`midfield_pressure_rate`**: Pressure events per minute in midfield
- **`midfield_pressure_in_first_5s`**: Number of pressure events within first 5 seconds of opponent possession
- **`midfield_pressure_in_first_10s`**: Number of pressure events within first 10 seconds
- **`midfield_time_to_first_pressure`**: Average time from possession start to first pressure event
- **`midfield_pressure_zones`**: Pressure distribution across midfield zones (left/center/right)
- **`midfield_pressure_on_passes`**: Pressure applied during opponent passes
- **`midfield_pressure_on_receipts`**: Pressure applied during ball receipts

**Computation**:
```python
# Opponent events under pressure in midfield
opponent_midfield_under_pressure = events[
    (events['possession_team.id'] != 217) &  # Opponent possession
    (events['team.id'] == 217) &  # Barcelona applying pressure
    (events['x'] >= 40) & (events['x'] <= 80) &
    (events['under_pressure'] == True)
]
```

### 2.2 Tempo Features
**Purpose**: Measure speed of defensive response

- **`midfield_reaction_time`**: Average time between opponent event and Barcelona defensive action
- **`midfield_pressing_intensity`**: Events per second in first 10 seconds of opponent possession
- **`midfield_immediate_pressure_rate`**: Percentage of opponent events pressured within 2 seconds
- **`midfield_pressure_persistence`**: Duration of sustained pressure sequences

---

## 3. Access Control Features

### 3.1 Progressive Pass Prevention
**Purpose**: Measure ability to prevent forward passes through midfield

- **`progressive_passes_allowed_midfield`**: Number of progressive passes allowed through midfield
  - Progressive: pass length > 10m AND end_x > start_x + 5m
- **`progressive_pass_prevention_rate`**: Percentage of progressive passes intercepted/blocked
- **`progressive_passes_central_lane`**: Progressive passes through central lane (35 ≤ y ≤ 45)
- **`progressive_passes_wide`**: Progressive passes through wide areas
- **`progressive_pass_distance_allowed`**: Average distance of progressive passes allowed
- **`progressive_pass_angle_allowed`**: Average angle of progressive passes (forward vs. sideways)

**Computation**:
```python
# Opponent progressive passes through midfield
opponent_passes = events[
    (events['type_name'] == 'Pass') &
    (events['possession_team.id'] != 217) &
    (events['x'] >= 40) & (events['x'] <= 80)
]

progressive = opponent_passes[
    (opponent_passes['pass.length'] > 10) &
    (opponent_passes['pass.end_location[0]'] > opponent_passes['x'] + 5)
]
```

### 3.2 Through Ball Prevention
**Purpose**: Measure ability to prevent dangerous through balls

- **`through_balls_allowed_midfield`**: Through balls allowed in midfield
- **`through_ball_prevention_rate`**: Percentage of through ball attempts intercepted
- **`through_balls_ending_final_third`**: Through balls that reach final third

**Computation**:
```python
through_balls = opponent_passes[opponent_passes['pass.through_ball'] == True]
```

### 3.3 Switch Prevention
**Purpose**: Measure ability to prevent long switches across midfield

- **`switches_allowed_midfield`**: Long switches allowed through midfield
- **`switch_prevention_rate`**: Percentage of switches intercepted
- **`switch_distance_allowed`**: Average distance of switches allowed

**Computation**:
```python
switches = opponent_passes[opponent_passes['pass.switch'] == True]
```

---

## 4. Spatial and Compactness Features

### 4.1 Compactness Proxies
**Purpose**: Measure defensive compactness in midfield

- **`midfield_defensive_width`**: Width of defensive shape in midfield (max_y - min_y of defensive events)
- **`midfield_defensive_depth`**: Depth of defensive shape (max_x - min_x)
- **`midfield_player_density`**: Average number of Barcelona players in midfield zone per event
- **`midfield_compactness_index`**: Ratio of width to depth (lower = more compact)
- **`midfield_central_concentration`**: Percentage of defensive actions in central zone (35 ≤ y ≤ 45)

**Computation**:
```python
# For each opponent possession, calculate defensive shape
defensive_events = events[
    (events['team.id'] == 217) &
    (events['x'] >= 40) & (events['x'] <= 80) &
    (events['type_name'].isin(['Interception', 'Ball Recovery', 'Duel', 'Tackle']))
]

width = defensive_events.groupby('possession')['y'].agg(['max', 'min'])
midfield_defensive_width = (width['max'] - width['min']).mean()
```

### 4.2 Zone Coverage Features
**Purpose**: Measure coverage of different midfield zones

- **`midfield_left_zone_coverage`**: Defensive actions in left midfield (y < 40)
- **`midfield_central_zone_coverage`**: Defensive actions in central midfield (35 ≤ y ≤ 45)
- **`midfield_right_zone_coverage`**: Defensive actions in right midfield (y > 40)
- **`midfield_zone_balance`**: Standard deviation of coverage across zones (lower = more balanced)
- **`midfield_coverage_gaps`**: Identification of zones with minimal defensive presence

### 4.3 Spatial Distribution Features
**Purpose**: Understand spatial patterns of defensive actions

- **`midfield_defensive_actions_x_mean`**: Mean x-coordinate of defensive actions
- **`midfield_defensive_actions_y_mean`**: Mean y-coordinate of defensive actions
- **`midfield_defensive_actions_x_std`**: Standard deviation of x-coordinates
- **`midfield_defensive_actions_y_std`**: Standard deviation of y-coordinates
- **`midfield_defensive_actions_clustering`**: Spatial clustering coefficient

---

## 5. Passing Through Midfield Features

### 5.1 Pass Prevention Features
**Purpose**: Measure ability to prevent opponent passes through midfield

- **`passes_allowed_midfield_total`**: Total opponent passes in midfield
- **`passes_intercepted_midfield`**: Passes intercepted in midfield
- **`pass_completion_rate_allowed_midfield`**: Opponent pass completion rate in midfield
- **`passes_allowed_forward`**: Forward passes allowed (end_x > start_x)
- **`passes_allowed_backward`**: Backward passes allowed
- **`passes_allowed_lateral`**: Lateral passes allowed (|end_y - start_y| > |end_x - start_x|)
- **`average_pass_length_allowed`**: Average length of passes allowed
- **`long_passes_allowed_midfield`**: Passes > 20m allowed

**Computation**:
```python
opponent_passes_midfield = events[
    (events['type_name'] == 'Pass') &
    (events['possession_team.id'] != 217) &
    (events['x'] >= 40) & (events['x'] <= 80)
]

passes_allowed = len(opponent_passes_midfield)
passes_intercepted = len(opponent_passes_midfield[
    opponent_passes_midfield['pass.outcome.name'] == 'Incomplete'
])
```

### 5.2 Pass Sequence Features
**Purpose**: Measure ability to break opponent passing sequences

- **`consecutive_passes_allowed_midfield`**: Average consecutive passes allowed before intervention
- **`pass_sequence_length_allowed`**: Average length of pass sequences in midfield
- **`pass_chain_break_rate`**: Percentage of pass sequences broken by Barcelona
- **`passes_before_interception`**: Average number of passes before interception

### 5.3 Pass Direction Features
**Purpose**: Understand which directions opponents can pass through midfield

- **`passes_allowed_left_to_right`**: Passes moving from left to right
- **`passes_allowed_right_to_left`**: Passes moving from right to left
- **`passes_allowed_center_to_wide`**: Passes from center to wide areas
- **`passes_allowed_wide_to_center`**: Passes from wide to center
- **`passes_allowed_defensive_to_final`**: Passes from defensive third through midfield to final third

---

## 6. Carrying Through Midfield Features

### 6.1 Carry Prevention Features
**Purpose**: Measure ability to prevent opponent carries through midfield

- **`carries_allowed_midfield_total`**: Total opponent carries in midfield
- **`carries_interrupted_midfield`**: Carries interrupted by Barcelona actions
- **`carry_distance_allowed`**: Average distance of carries allowed
- **`carries_entering_midfield`**: Carries that enter midfield from defensive third
- **`carries_exiting_midfield`**: Carries that exit midfield to final third
- **`carries_through_midfield`**: Carries that traverse entire midfield zone
- **`carry_progression_allowed`**: Average forward progression of carries (end_x - start_x)

**Computation**:
```python
opponent_carries = events[
    (events['type_name'] == 'Carry') &
    (events['possession_team.id'] != 217) &
    (events['x'] >= 40) & (events['x'] <= 80)
]

carry_distance = opponent_carries.apply(
    lambda row: ((row['carry.end_location[0]'] - row['x'])**2 + 
                 (row['carry.end_location[1]'] - row['y'])**2)**0.5,
    axis=1
)
```

### 6.2 Carry Direction Features
**Purpose**: Understand carry patterns through midfield

- **`carries_forward_allowed`**: Forward carries (end_x > start_x)
- **`carries_lateral_allowed`**: Lateral carries
- **`carries_central_lane`**: Carries through central lane
- **`carries_wide_areas`**: Carries through wide areas

---

## 7. Recovery and Transition Features

### 7.1 Transition Features
**Purpose**: Measure effectiveness in transition moments

- **`midfield_transition_recoveries`**: Recoveries during opponent transitions
- **`midfield_counter_press_events`**: Defensive actions within 5 seconds of losing possession
- **`midfield_transition_to_attack`**: Time from recovery to first attacking action
- **`midfield_recovery_location_quality`**: Quality of recovery locations (higher x = better)

### 7.2 Possession Transition Features
**Purpose**: Measure control of possession transitions in midfield

- **`possession_won_midfield`**: Possessions won in midfield
- **`possession_lost_midfield`**: Possessions lost in midfield
- **`midfield_possession_win_rate`**: Percentage of midfield duels/contests won
- **`midfield_turnover_rate`**: Rate of forcing opponent turnovers in midfield

---

## 8. Temporal Features

### 8.1 Time-Based Features
**Purpose**: Measure timing of defensive actions

- **`midfield_action_timing_mean`**: Mean time of defensive actions within possession
- **`midfield_action_timing_std`**: Standard deviation of action timing
- **`midfield_early_interventions`**: Actions within first 3 seconds of possession
- **`midfield_late_interventions`**: Actions after 10 seconds of possession
- **`midfield_sustained_pressure_duration`**: Average duration of pressure sequences

### 8.2 Possession Duration Features
**Purpose**: Measure ability to limit opponent possession time

- **`opponent_possession_duration_midfield`**: Average duration of opponent possessions in midfield
- **`possession_termination_rate`**: Rate at which opponent possessions are terminated in midfield
- **`time_to_possession_end`**: Average time from midfield entry to possession end

---

## 9. Zone-Specific Features

### 9.1 Defensive Third to Midfield Transition
**Purpose**: Measure prevention of transitions from defensive to midfield

- **`defensive_to_midfield_passes_allowed`**: Passes from defensive third (x < 40) entering midfield
- **`defensive_to_midfield_carries_allowed`**: Carries from defensive third entering midfield
- **`defensive_to_midfield_prevention_rate`**: Percentage of transitions prevented
- **`defensive_to_midfield_interception_rate`**: Interceptions at midfield entry point

### 9.2 Midfield to Final Third Transition
**Purpose**: Measure prevention of transitions from midfield to final third

- **`midfield_to_final_passes_allowed`**: Passes from midfield (40 ≤ x ≤ 80) entering final third (x > 80)
- **`midfield_to_final_carries_allowed`**: Carries from midfield entering final third
- **`midfield_to_final_prevention_rate`**: Percentage of transitions prevented
- **`midfield_to_final_interception_rate`**: Interceptions at midfield exit point
- **`bypass_attempts_total`**: Total attempts to bypass midfield (reach final third from defensive third)
- **`bypass_prevention_rate`**: Overall bypass prevention rate

**Computation**:
```python
# Identify bypass attempts: possession starting in defensive third, reaching final third
bypass_attempts = []
for poss_id, poss_events in opponent_possessions.groupby('possession'):
    start_x = poss_events['x'].iloc[0]
    max_x = poss_events['x'].max()
    if start_x < 40 and max_x > 80:
        bypass_attempts.append(poss_id)

bypass_prevention_rate = 1 - (len(bypass_attempts) / total_opponent_possessions)
```

---

## 10. Player Position and Role Features

### 10.1 Midfielder-Specific Features
**Purpose**: Analyze performance of midfield players

- **`midfielder_interceptions_per_player`**: Interceptions by each midfielder
- **`midfielder_recoveries_per_player`**: Recoveries by each midfielder
- **`midfielder_pressures_per_player`**: Pressure events applied by each midfielder
- **`midfielder_duel_win_rate_per_player`**: Duel win rate by each midfielder
- **`midfielder_zone_coverage`**: Spatial coverage by each midfielder

**Computation**:
```python
# Filter midfielders from Starting XI
midfielders = lineup[
    (lineup['position.name'].str.contains('Midfield')) |
    (lineup['position.name'].str.contains('Center'))
]

# Aggregate events by midfielder
midfielder_stats = events[
    (events['team.id'] == 217) &
    (events['x'] >= 40) & (events['x'] <= 80)
].groupby('player.id').agg({
    'type_name': 'count',
    'interception': lambda x: (x == 'Interception').sum(),
    # ... other aggregations
})
```

### 10.2 Formation and Tactical Features
**Purpose**: Understand tactical setup impact

- **`formation_type`**: Formation used (from Starting XI)
- **`midfield_player_count`**: Number of players in midfield positions
- **`midfield_width_utilization`**: How wide midfielders are positioned
- **`midfield_depth_utilization`**: How deep/advanced midfielders are positioned

---

## 11. Advanced Composite Features

### 11.1 Midfield Strength Index
**Purpose**: Composite metric combining multiple features

- **`midfield_strength_index`**: Weighted combination of:
  - Interception rate
  - Recovery rate
  - Pressure intensity
  - Bypass prevention rate
  - Compactness score

**Formula**:
```
midfield_strength_index = (
    0.25 * normalized_interception_rate +
    0.25 * normalized_recovery_rate +
    0.20 * normalized_pressure_intensity +
    0.20 * bypass_prevention_rate +
    0.10 * (1 - normalized_compactness_index)
)
```

### 11.2 Bypass Risk Score
**Purpose**: Predict likelihood of bypass based on current state

- **`bypass_risk_score`**: Probability of bypass given current features
- **`bypass_risk_factors`**: Key factors contributing to high risk
  - Low pressure intensity
  - Wide defensive shape
  - Low interception rate
  - High progressive pass completion rate

### 11.3 Defensive Efficiency Metrics
**Purpose**: Measure efficiency of defensive actions

- **`defensive_action_efficiency`**: Successful defensive actions per 100 opponent events
- **`pressure_to_interception_ratio`**: Interceptions per pressure event
- **`recovery_quality_score`**: Average x-coordinate of recoveries (higher = better)

---

## 12. Contextual Features

### 12.1 Match Context Features
**Purpose**: Account for match situation

- **`score_differential`**: Barcelona goals - Opponent goals at time of possession
- **`match_minute`**: Minute of match
- **`period`**: First half (1) or second half (2)
- **`home_away`**: Home or away match
- **`time_since_last_goal`**: Time since last goal scored

### 12.2 Play Pattern Features
**Purpose**: Account for how play was initiated

- **`play_pattern_regular`**: Regular play possessions
- **`play_pattern_set_piece`**: Set piece possessions (free kick, corner, etc.)
- **`play_pattern_transition`**: Transition possessions
- **`play_pattern_counter_attack`**: Counter attack possessions

---

## 13. Feature Aggregation Levels

Features can be computed at multiple aggregation levels:

1. **Match Level**: Overall statistics for the entire match
2. **Period Level**: Statistics for first half vs. second half
3. **Possession Level**: Features for each opponent possession
4. **Time Window Level**: Features within specific time windows (e.g., first 10 seconds)
5. **Zone Level**: Features for specific zones within midfield
6. **Player Level**: Individual player contributions

---

## 14. Feature Engineering Implementation Notes

### 14.1 Data Requirements
- All features require filtering by:
  - Team ID: 217 (Barcelona)
  - Zone: x between 40 and 80 (midfield)
  - Event types: Pass, Carry, Interception, Ball Recovery, Duel, etc.

### 14.2 Normalization Considerations
- Many features should be normalized by:
  - Total opponent possessions
  - Match duration
  - Number of events
  - Pitch area

### 14.3 Missing Data Handling
- Some events may not have location data
- Some events may not have player information
- Use appropriate imputation or exclusion strategies

### 14.4 Feature Selection Priority
For initial modeling, prioritize:
1. **Core Defensive Actions**: Interceptions, Recoveries, Duels
2. **Pressure Features**: Time to first pressure, pressure intensity
3. **Access Control**: Progressive pass prevention, through ball prevention
4. **Spatial Features**: Compactness, zone coverage
5. **Temporal Features**: Reaction time, possession duration

---

## 15. Example Feature Calculation Workflow

```python
def compute_midfield_features(events_df, possessions_df, team_id=217):
    """
    Compute all midfield strength features for Barcelona.
    
    Parameters:
    -----------
    events_df : DataFrame
        Events data with columns: type_name, team.id, possession_team.id, 
                                  x, y, timestamp, player.id, etc.
    possessions_df : DataFrame
        Possession data with columns: possession, team_id, start_time, etc.
    team_id : int
        Team ID for Barcelona (217)
    
    Returns:
    --------
    features_df : DataFrame
        Feature matrix with one row per possession or match
    """
    
    # Filter Barcelona defensive events in midfield
    midfield_events = events_df[
        (events_df['x'] >= 40) & (events_df['x'] <= 80)
    ]
    
    barca_defensive = midfield_events[
        (midfield_events['team.id'] == team_id) &
        (midfield_events['type_name'].isin(['Interception', 'Ball Recovery', 'Duel']))
    ]
    
    opponent_events = midfield_events[
        (midfield_events['possession_team.id'] != team_id)
    ]
    
    # Compute features
    features = {}
    
    # 1. Defensive Actions
    features['midfield_interceptions_total'] = len(
        barca_defensive[barca_defensive['type_name'] == 'Interception']
    )
    
    # 2. Pressure Features
    features['midfield_pressure_events_total'] = len(
        opponent_events[opponent_events['under_pressure'] == True]
    )
    
    # 3. Access Control
    opponent_passes = opponent_events[opponent_events['type_name'] == 'Pass']
    features['progressive_passes_allowed'] = len(
        opponent_passes[
            (opponent_passes['pass.length'] > 10) &
            (opponent_passes['pass.end_location[0]'] > opponent_passes['x'] + 5)
        ]
    )
    
    # ... continue for all features
    
    return pd.DataFrame([features])
```

---

## 16. Feature Validation and Quality Checks

### 16.1 Data Quality Checks
- Verify all required fields are present
- Check for null values in critical fields
- Validate coordinate ranges (x: 0-120, y: 0-80)
- Verify timestamp consistency

### 16.2 Feature Quality Checks
- Check for feature correlations (multicollinearity)
- Identify features with low variance
- Validate feature distributions
- Check for outliers and anomalies

### 16.3 Feature Importance
- Use domain knowledge to prioritize features
- Validate with correlation to bypass label
- Use feature importance from models (SHAP values)
- Consider feature interpretability

---

## Summary

This document outlines **100+ features** that can be engineered from the StatsBomb event data to analyze midfield strength and bypass prevention. The features are organized into 16 categories covering:

- Defensive actions (interceptions, recoveries, duels)
- Pressure and tempo
- Access control (progressive passes, through balls)
- Spatial features (compactness, coverage)
- Passing and carrying through midfield
- Recovery and transitions
- Temporal patterns
- Zone-specific analysis
- Player and tactical features
- Composite metrics

These features provide a comprehensive framework for analyzing Barcelona's midfield strength and identifying factors that contribute to effective bypass prevention.

