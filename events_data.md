# Events Data Analysis - Match 15946

## Overview

The file `15946.json` contains detailed event-level data from a football match between **Barcelona** (team ID: 217) and **Deportivo Alavés** (team ID: 206). The data follows the StatsBomb open data format and includes comprehensive tracking of all events that occurred during the match.

## Data Structure

The file contains a JSON array of event objects, with each event representing a specific action or occurrence during the match. The file contains approximately **3,762 events** covering the entire match duration.

## Common Fields (Present in Most Events)

All events share a common set of base fields:

- **`id`**: Unique identifier (UUID) for the event
- **`index`**: Sequential index number of the event in the match
- **`period`**: Match period (1 = first half, 2 = second half)
- **`timestamp`**: Time within the period (format: "HH:MM:SS.mmm")
- **`minute`**: Match minute (0-45 for first half, 45-90+ for second half)
- **`second`**: Second within the minute
- **`type`**: Event type object containing:
  - `id`: Numeric event type ID
  - `name`: Event type name (e.g., "Pass", "Shot", "Duel")
- **`possession`**: Possession number (sequential counter)
- **`possession_team`**: Team in possession object:
  - `id`: Team ID
  - `name`: Team name
- **`play_pattern`**: How play was initiated:
  - `id`: Pattern ID
  - `name`: Pattern name (e.g., "Regular Play", "From Kick Off", "From Throw In", "From Free Kick")
- **`team`**: Team performing the event:
  - `id`: Team ID
  - `name`: Team name
- **`duration`**: Duration of the event in seconds (0.0 for instantaneous events)
- **`related_events`**: Array of event IDs that are related to this event

## Event-Specific Fields

Many events include additional fields specific to their type:

### Player and Position Information
- **`player`**: Player object (when applicable):
  - `id`: Player ID
  - `name`: Player full name
- **`position`**: Player position object:
  - `id`: Position ID
  - `name`: Position name (e.g., "Goalkeeper", "Right Back", "Center Forward")

### Location Data
- **`location`**: Array `[x, y]` coordinates on the pitch (pitch dimensions: 0-120 for x, 0-80 for y)
- **`under_pressure`**: Boolean indicating if the player was under pressure (when applicable)

## Event Types Observed

### 1. **Starting XI** (Type ID: 35)
- Contains team lineup information
- Includes `tactics` object with:
  - `formation`: Formation number (e.g., 442)
  - `lineup`: Array of player objects with:
    - `player`: Player information
    - `position`: Starting position
    - `jersey_number`: Player's jersey number

### 2. **Half Start** (Type ID: 18)
- Marks the start of a half
- May have `related_events` linking to corresponding events

### 3. **Half End** (Type ID: 34)
- Marks the end of a half
- May have `related_events` linking to corresponding events

### 4. **Pass** (Type ID: 30)
- Most common event type
- Additional `pass` object contains:
  - `recipient`: Player receiving the pass
  - `length`: Pass distance
  - `angle`: Pass angle in radians
  - `height`: Pass height object (e.g., "Ground Pass", "High Pass")
  - `end_location`: `[x, y]` coordinates where pass ended
  - `body_part`: Body part used (e.g., "Right Foot", "Left Foot")
  - `type`: Pass type (e.g., "Recovery", "Kick Off")
  - `outcome`: Pass outcome (e.g., "Complete", "Incomplete")
  - `switch`: Boolean for long switches
  - `aerial_won`: Boolean if aerial duel was won
  - `through_ball`: Boolean for through balls
  - `assisted_shot_id`: ID of shot if pass assisted a shot

### 5. **Ball Receipt*** (Type ID: 42)
- Player receiving the ball
- `ball_receipt` object contains:
  - `outcome`: Receipt outcome (e.g., "Complete", "Incomplete")

### 6. **Carry** (Type ID: 43)
- Player carrying/running with the ball
- `carry` object contains:
  - `end_location`: `[x, y]` where carry ended

### 7. **Duel** (Type ID: 4)
- One-on-one contest between players
- `duel` object contains:
  - `type`: Duel type (e.g., "Aerial Lost", "Aerial Won", "Ground Lost", "Ground Won")

### 8. **Ball Recovery** (Type ID: 2)
- Team recovering possession
- `ball_recovery` object may contain:
  - `recovery_failure`: Boolean indicating if recovery failed

### 9. **Interception** (Type ID: 10)
- Player intercepting a pass
- `interception` object contains:
  - `outcome`: Interception outcome (e.g., "Success In Play", "Success Out")

### 10. **Shot** (Type ID: 16)
- Shooting attempts
- `shot` object contains:
  - `statsbomb_xg`: Expected goals value (0.0-1.0)
  - `end_location`: `[x, y, z]` coordinates where shot ended (z = height)
  - `key_pass_id`: ID of pass that created the shot
  - `outcome`: Shot outcome (e.g., "Goal", "Off T", "Saved", "Blocked", "Post")
  - `first_time`: Boolean for first-time shots
  - `technique`: Shot technique (e.g., "Half Volley", "Normal", "Volley")
  - `body_part`: Body part used
  - `type`: Shot type (e.g., "Open Play", "Free Kick", "Penalty")
  - `freeze_frame`: Array of player positions at moment of shot:
    - `location`: `[x, y]` coordinates
    - `player`: Player information
    - `position`: Player position
    - `teammate`: Boolean indicating if player is teammate of shooter

### 11. **Goal Keeper** (Type ID: 23)
- Goalkeeper-specific actions
- `goalkeeper` object contains:
  - `technique`: Technique used (e.g., "Standing", "Diving")
  - `position`: Position type (e.g., "Set")
  - `type`: Action type (e.g., "Goal Conceded", "Save", "Shot Saved")
  - `outcome`: Outcome (e.g., "No Touch", "Saved", "Claimed")

### 12. **Substitution** (Type ID: 19)
- Player substitutions
- `substitution` object contains:
  - `outcome`: Substitution type (e.g., "Tactical", "Injury")
  - `replacement`: Player object for the replacement player

## Play Patterns

Events are categorized by how play was initiated:
- **Regular Play** (ID: 1): Normal open play
- **From Kick Off** (ID: 9): Following a kick-off
- **From Throw In** (ID: 4): Following a throw-in
- **From Free Kick** (ID: 3): Following a free kick
- **From Corner** (ID: 5): Following a corner kick
- **From Goal Kick** (ID: 6): Following a goal kick

## Key Observations

1. **Temporal Coverage**: Events span the entire match with timestamps from 00:00:00.000 to approximately 00:47:31.061 (92 minutes, 31 seconds), indicating the match went into stoppage time.

2. **Event Relationships**: Events are linked through `related_events` arrays, allowing reconstruction of sequences (e.g., Pass → Ball Receipt → Carry → Shot).

3. **Spatial Data**: Most events include `location` coordinates, enabling spatial analysis of player movements and event locations.

4. **Possession Tracking**: Each event is assigned to a possession number, allowing reconstruction of possession sequences.

5. **Team Information**: Both teams (Barcelona and Deportivo Alavés) are represented with their respective team IDs and names throughout.

6. **Player Identification**: Players are identified by unique IDs and full names, with position information available for most events.

7. **Rich Context**: Events include contextual information such as pressure situations, body parts used, techniques, and outcomes, providing comprehensive detail for analysis.

## Use Cases

This data structure supports various analytical applications:
- **Possession Analysis**: Track possession sequences and transitions
- **Passing Networks**: Analyze passing patterns and player connections
- **Shot Analysis**: Evaluate shot quality, locations, and expected goals
- **Defensive Actions**: Study interceptions, recoveries, and duels
- **Tactical Analysis**: Examine formations, play patterns, and spatial distributions
- **Player Performance**: Individual player event tracking and statistics
- **Match Flow**: Temporal analysis of match events and momentum

## Data Quality Notes

- All events have unique IDs and sequential indices
- Timestamps are precise to milliseconds
- Location data appears complete for events that involve spatial positioning
- Related events are properly linked through ID references
- Team and player information is consistently structured throughout

