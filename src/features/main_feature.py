"""
Main feature engineering pipeline for midfielder-level features.

Loads event data from JSON files, computes all features for each midfielder
in each match, and saves results to CSV.
"""

import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Set, Union
from datetime import datetime
import sys
import warnings

# Suppress FutureWarnings about fillna downcasting
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting object dtype arrays.*')

# Adjust path for local imports when running as a script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.midfield import FEATURE_FUNCTIONS
from src.features.midfield.context import MidfieldFeatureContext, get_midfielder_ids, get_position_code


def load_config(config_path: Path = Path("config/config.yaml")) -> Dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to config.yaml file.

    Returns
    -------
    Dict
        Configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_team_id_from_name(team_name: str) -> Optional[int]:
    """
    Get team ID from team name.

    Parameters
    ----------
    team_name : str
        Team name.

    Returns
    -------
    Optional[int]
        Team ID if found, None otherwise.
    """
    team_name_to_id = {
        "Barcelona": 217,
        "Real Madrid": 220,
        "Atlético Madrid": 212,
        "Liverpool": 24,
        "Manchester City": 36,
        "Arsenal": 1,
        "Chelsea": 33,
        "Tottenham Hotspur": 38,
    }

    return team_name_to_id.get(team_name)


def get_midfielder_positions(raw_events: pd.DataFrame, team_id: int, midfielder_ids: Set[int]) -> Dict[int, Optional[int]]:
    """
    Extract midfielder position codes from Starting XI lineup data.

    Parameters
    ----------
    raw_events : pd.DataFrame
        Raw events DataFrame from JSON.
    team_id : int
        Team ID to filter for.
    midfielder_ids : Set[int]
        Set of midfielder player IDs.

    Returns
    -------
    Dict[int, Optional[int]]
        Dictionary mapping player_id to position_code (0-7) or None if not found.
    """
    position_map: Dict[int, Optional[int]] = {}

    # Starting XI entries hold the lineups
    xi_rows = raw_events[
        (raw_events.get("team.id") == team_id)
        & (raw_events.get("type.name") == "Starting XI")
    ]
    
    for _, row in xi_rows.iterrows():
        lineup = row.get("tactics.lineup")
        if isinstance(lineup, list):
            for entry in lineup:
                player = entry.get("player", {})
                player_id = player.get("id")
                if player_id is None or player_id not in midfielder_ids:
                    continue
                
                pos_name = entry.get("position", {}).get("name", "")
                if pos_name:
                    position_code = get_position_code(pos_name)
                    if position_code is not None:
                        position_map[player_id] = position_code

    # Check substitutions for replacements
    sub_rows = raw_events[
        (raw_events.get("team.id") == team_id)
        & (raw_events.get("type.name") == "Substitution")
    ]
    
    for _, row in sub_rows.iterrows():
        repl = row.get("substitution.replacement")
        if isinstance(repl, dict):
            player_id = repl.get("id")
            if player_id is None or player_id not in midfielder_ids:
                continue
            
            pos_name = (repl.get("position") or {}).get("name", "")
            if pos_name:
                position_code = get_position_code(pos_name)
                if position_code is not None:
                    position_map[player_id] = position_code

    return position_map


def load_single_json(json_path: Path) -> pd.DataFrame:
    """
    Load a single event JSON file and return as DataFrame.

    Parameters
    ----------
    json_path : Path
        Path to JSON file.

    Returns
    -------
    pd.DataFrame
        DataFrame with flattened event data.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.json_normalize(data, sep=".")
    df["match_id"] = int(json_path.stem) if json_path.stem.isdigit() else json_path.stem
    return df


def coalesce_outcome(df: pd.DataFrame) -> pd.Series:
    """
    Create a single 'outcome_name' by taking the first non-null among common outcome fields.

    Parameters
    ----------
    df : pd.DataFrame
        Raw events DataFrame.

    Returns
    -------
    pd.Series
        Series with outcome names.
    """
    candidates = [c for c in df.columns if c.endswith(".outcome.name")] + \
                 [c for c in df.columns if c.endswith(".outcome")]
    if not candidates:
        return pd.Series([None] * len(df), index=df.index)

    sub = df.reindex(columns=candidates)
    for c in sub.columns:
        sub[c] = sub[c].apply(lambda v: v if (pd.isna(v) or isinstance(v, str)) else str(v))
    sub = sub.astype(str).replace('nan', pd.NA)
    result = sub.bfill(axis=1).iloc[:, 0]
    return result.replace(pd.NA, None) if hasattr(result, 'replace') else result


def _safe_get_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Safely get a boolean series, handling missing columns and non-boolean types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to extract from.
    column : str
        Column name.

    Returns
    -------
    pd.Series
        Boolean series.
    """
    if column in df.columns:
        series = df[column].copy()
        # Convert to bool, replacing NaN with False
        # Use where to avoid FutureWarning about fillna downcasting
        series = series.where(pd.notna(series), False)
        return series.astype(bool)
    return pd.Series(False, index=df.index)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize event DataFrame columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw events DataFrame from JSON.

    Returns
    -------
    pd.DataFrame
        Cleaned events DataFrame with standardized columns.
    """
    out = pd.DataFrame({
        "match_id": df["match_id"],
        "team_name": df.get("team.name"),
        "team_id": df.get("team.id"),
        "player_name": df.get("player.name"),
        "player_id": df.get("player.id"),
        "type_name": df.get("type.name"),
        "timestamp": df.get("timestamp"),
        "minute": df.get("minute"),
        "second": df.get("second"),
        "possession_team_name": df.get("possession_team.name"),
        "possession_team_id": df.get("possession_team.id"),
        "possession": df.get("possession"),
        "period": df.get("period"),
        "duration": df.get("duration"),
        "under_pressure": _safe_get_bool_series(df, "under_pressure"),
        "id": df.get("id"),  # Event ID for linking
    })

    # Extract location coordinates
    loc = df.get("location")
    if loc is not None:
        out["x"] = loc.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else np.nan)
        out["y"] = loc.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else np.nan)
    else:
        out["x"] = np.nan
        out["y"] = np.nan

    # Pass-related features
    out["pass.length"] = df.get("pass.length")
    out["pass.end_location"] = df.get("pass.end_location")
    out["pass.through_ball"] = _safe_get_bool_series(df, "pass.through_ball")
    out["pass.switch"] = _safe_get_bool_series(df, "pass.switch")
    out["pass.angle"] = df.get("pass.angle")
    out["pass.progressive"] = _safe_get_bool_series(df, "pass.progressive")
    out["pass.shot_assist"] = _safe_get_bool_series(df, "pass.shot_assist")
    out["pass.goal_assist"] = _safe_get_bool_series(df, "pass.goal_assist")
    out["pass.type.name"] = df.get("pass.type.name")
    out["pass.cross"] = _safe_get_bool_series(df, "pass.cross")
    out["pass.carry_id"] = df.get("pass.carry_id")
    out["pass.outcome.name"] = df.get("pass.outcome.name")

    # Standardized names for easier access
    out["pass_length"] = out["pass.length"]
    out["pass_end_location"] = out["pass.end_location"]
    out["pass_through_ball"] = out["pass.through_ball"]
    out["pass_switch"] = out["pass.switch"]
    out["pass_angle"] = out["pass.angle"]
    out["pass_progressive"] = out["pass.progressive"]
    out["pass_shot_assist"] = out["pass.shot_assist"]
    out["pass_goal_assist"] = out["pass.goal_assist"]
    out["pass_type_name"] = out["pass.type.name"]
    out["pass_cross"] = out["pass.cross"]
    out["pass_carry_id"] = out["pass.carry_id"]
    out["pass_outcome_name"] = out["pass.outcome.name"]

    # Duel-related features
    out["duel.type.name"] = df.get("duel.type.name")
    out["duel.outcome.name"] = df.get("duel.outcome.name")
    out["duel.tackle"] = df.get("duel.tackle")
    out["duel_type"] = out["duel.type.name"]
    out["duel_outcome"] = out["duel.outcome.name"]
    out["duel_tackle"] = out["duel.tackle"]

    # Counterpress
    out["counterpress"] = _safe_get_bool_series(df, "counterpress")

    # Block-related features
    out["block.deflection"] = _safe_get_bool_series(df, "block.deflection")
    out["block.block_type"] = df.get("block.type.name")
    out["block_deflection"] = out["block.deflection"]
    out["block_block_type"] = out["block.block_type"]

    # Carry-related features
    out["carry.end_location"] = df.get("carry.end_location")
    out["carry.id"] = df.get("carry.id")
    out["carry_end_location"] = out["carry.end_location"]
    out["carry_id"] = out["carry.id"]

    # Tactical/Formation
    out["tactics.formation"] = df.get("tactics.formation")
    out["tactics.lineup"] = df.get("tactics.lineup")
    out["tactics_formation"] = out["tactics.formation"]
    out["tactics_lineup"] = out["tactics.lineup"]

    # Play Pattern
    out["play_pattern.name"] = df.get("play_pattern.name")
    out["play_pattern_name"] = out["play_pattern.name"]

    # Take On
    out["take_on.outcome.name"] = df.get("take_on.outcome.name")
    out["take_on_outcome_name"] = out["take_on.outcome.name"]

    # Foul-related
    out["foul_committed.type.name"] = df.get("foul_committed.type.name")
    out["foul_committed.card.name"] = df.get("foul_committed.card.name")
    out["foul_won.advantage"] = _safe_get_bool_series(df, "foul_won.advantage")
    out["foul_committed_type_name"] = out["foul_committed.type.name"]
    out["foul_committed_card_name"] = out["foul_committed.card.name"]
    out["foul_won_advantage"] = out["foul_won.advantage"]

    # 50/50
    out["50_50.outcome.name"] = df.get("50_50.outcome.name")
    out["50_50_outcome_name"] = out["50_50.outcome.name"]

    # Shot xG for expected assists/xg chain
    out["shot.statsbomb_xg"] = df.get("shot.statsbomb_xg")
    out["shot.key_pass_id"] = df.get("shot.key_pass_id")
    out["shot.carry_id"] = df.get("shot.carry_id")
    out["shot_statsbomb_xg"] = out["shot.statsbomb_xg"]
    out["shot_key_pass_id"] = out["shot.key_pass_id"]
    out["shot_carry_id"] = out["shot.carry_id"]

    # Additional columns for compatibility
    out["team.id"] = out["team_id"]
    out["possession_team.id"] = out["possession_team_id"]
    out["outcome_name"] = coalesce_outcome(df)

    # Convert timestamp to timedelta
    out["timestamp"] = pd.to_timedelta(out["timestamp"])

    return out


def compute_all_features(
    config_path: Optional[Path] = None,
    team_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Main function to compute all midfielder-level features for team and season specified in config.yaml.

    Parameters
    ----------
    config_path : Path, optional
        Path to config.yaml file (default: config/config.yaml)
    team_id : int, optional
        Team ID to use. If None, will be inferred from team name in config.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per midfielder per match and one column per feature
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    config = load_config(config_path)

    dataset_config = config.get("Dataset", {})
    team_name = dataset_config.get("team_name")
    season = dataset_config.get("season")

    if not team_name:
        raise ValueError("team_name not found in config.yaml")
    if not season:
        raise ValueError("season not found in config.yaml")

    print("=" * 80)
    print("Feature Engineering Pipeline")
    print("=" * 80)
    print(f"Team: {team_name}")
    print(f"Season: {season}")
    print()

    if team_id is None:
        team_id = get_team_id_from_name(team_name)
        if team_id is None:
            raise ValueError(f"Could not find team ID for '{team_name}'. Please provide team_id parameter.")

    print(f"Team ID: {team_id}")
    print()

    base_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "events"
    team_normalized = team_name.replace(" ", "_").replace("/", "_").replace("-", "_")
    season_normalized = season.replace("/", "_")
    dir_name = f"{team_normalized}_{season_normalized}"
    team_season_dir = base_dir / dir_name

    if not team_season_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {team_season_dir}\n"
            f"Run scripts/create_team_season_directory.py first to create the directory and download event files."
        )

    json_files = sorted(team_season_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {team_season_dir}")

    print(f"Processing {len(json_files)} matches...")
    print()
    print("Computing midfielder features per match...")
    print("-" * 80)

    all_midfielder_features = []

    for match_idx, json_file in enumerate(json_files, 1):
        match_id: Union[int, str] = int(json_file.stem) if json_file.stem.isdigit() else json_file.stem
        print(f"Match {match_idx}/{len(json_files)}: {match_id}")

        try:
            raw_df = load_single_json(json_file)
            events = clean_events(raw_df)

            midfielder_ids = get_midfielder_ids(raw_df, team_id)
            if not midfielder_ids:
                print(f"   No midfielders identified for match {match_id}. Skipping.")
                continue

            # Get position codes for all midfielders
            position_map = get_midfielder_positions(raw_df, team_id, midfielder_ids)

            ctx = MidfieldFeatureContext(
                raw_events=raw_df,
                events=events,
                team_id=team_id,
                midfielder_ids=midfielder_ids,
                match_id=match_id,
            )

            match_midfielder_data = []
            for player_id in sorted(midfielder_ids):
                # Get player name
                player_name = None
                player_events = events[events["player_id"] == player_id]
                if not player_events.empty:
                    player_name = player_events["player_name"].iloc[0]

                # Get position code (0-7) or None if not found
                midfielder_type = position_map.get(player_id)

                player_features = {
                    "player_id": player_id,
                    "player_name": player_name,
                    "midfielder_type": midfielder_type,
                    "match_id": match_id,
                    "team_id": team_id,
                    "team_name": team_name,
                    "season": season,
                    "computed_at": datetime.now().isoformat()
                }

                # Compute all features
                for feature_name, feature_func in FEATURE_FUNCTIONS.items():
                    try:
                        # Call the feature function, which returns a Series indexed by player_id
                        feature_series = feature_func(ctx)
                        # Get the value for the current player
                        value = feature_series.get(player_id)
                        player_features[feature_name] = value
                    except Exception as e:
                        print(f"   Warning: failed computing {feature_name} for player {player_id}: {e}")
                        player_features[feature_name] = np.nan

                match_midfielder_data.append(player_features)

            if match_midfielder_data:
                all_midfielder_features.extend(match_midfielder_data)
                print(f"   Computed features for {len(match_midfielder_data)} midfielders")

        except Exception as e:
            print(f"   Error processing match {match_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print()
    print("=" * 80)
    print("Feature Engineering Complete")
    print("=" * 80)
    print(f"Processed {len(json_files)} match entries")
    print()

    if not all_midfielder_features:
        raise ValueError("No midfielder features were successfully processed")

    features_df = pd.DataFrame(all_midfielder_features)

    # Reorder columns: metadata first, then features
    metadata_cols = ["player_id", "player_name", "midfielder_type", "match_id", "team_id", "team_name", "season", "computed_at"]
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]
    features_df = features_df[metadata_cols + feature_cols]

    # Round all numeric columns to 2 decimal places (except metadata columns)
    for col in feature_cols:
        if col in features_df.columns and features_df[col].dtype in [np.float64, np.float32]:
            features_df[col] = features_df[col].round(2)

    output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{team_normalized}_{season_normalized}_features.csv"
    output_path = output_dir / output_filename

    features_df.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")
    print(f"Shape: {features_df.shape} (rows=midfielder_match_events, columns=features)")
    print()

    return features_df


if __name__ == "__main__":
    features = compute_all_features()
    print("Feature computation complete.")

