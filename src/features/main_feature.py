"""
Main feature engineering pipeline.

Loads event data based on team and season from config.yaml and computes
all feature categories for midfield strength analysis.
"""

import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from access_control import compute_access_control_features
from defensive_actions import compute_defensive_action_features
from pressure_tempo import compute_pressure_tempo_features
from spatial_compactness import compute_spatial_compactness_features
from passing_features import compute_passing_features
from carrying_features import compute_carrying_features
from recovery_transition import compute_recovery_transition_features
from temporal_features import compute_temporal_features
from zone_specific import compute_zone_specific_features
from player_tactical import compute_player_tactical_features
from composite_features import compute_composite_features
from contextual_features import compute_contextual_features


def load_config(config_path: Path = Path("config/config.yaml")) -> Dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_team_id_from_name(team_name: str) -> Optional[int]:
    """Get team ID from team name."""
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


def load_single_json(json_path: Path) -> pd.DataFrame:
    """Load a single event JSON file and return as DataFrame."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.json_normalize(data, sep=".")
    df["match_id"] = int(json_path.stem) if json_path.stem.isdigit() else json_path.stem
    return df


def coalesce_outcome(df: pd.DataFrame) -> pd.Series:
    """Create a single 'outcome_name' by taking the first non-null among common outcome fields."""
    candidates = [c for c in df.columns if c.endswith(".outcome.name")] + \
                 [c for c in df.columns if c.endswith(".outcome")]
    if not candidates:
        return pd.Series([None] * len(df), index=df.index)
    
    sub = df.reindex(columns=candidates)
    for c in sub.columns:
        sub[c] = sub[c].apply(lambda v: v if (pd.isna(v) or isinstance(v, str)) else str(v))
    return sub.bfill(axis=1).iloc[:, 0]


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize event DataFrame columns."""
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
        "under_pressure": df.get("under_pressure", False)
    })
    
    loc = df.get("location")
    if loc is not None:
        out["x"] = loc.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else None)
        out["y"] = loc.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else None)
    else:
        out["x"] = None
        out["y"] = None
    
    out["pass.length"] = df.get("pass.length")
    out["pass.end_location"] = df.get("pass.end_location")
    out["pass.through_ball"] = df.get("pass.through_ball", False)
    out["pass.switch"] = df.get("pass.switch", False)
    out["pass.angle"] = df.get("pass.angle")
    out["pass.progressive"] = df.get("pass.progressive", False)
    out["pass_length"] = out["pass.length"]
    out["pass_end_location"] = out["pass.end_location"]
    out["pass_through_ball"] = out["pass.through_ball"]
    out["pass_switch"] = out["pass.switch"]
    out["pass_angle"] = out["pass.angle"]
    out["pass_progressive"] = out["pass.progressive"]
    
    out["interception.outcome.name"] = df.get("interception.outcome.name")
    out["interception_outcome"] = out["interception.outcome.name"]
    
    out["ball_recovery.recovery_failure"] = df.get("ball_recovery.recovery_failure", False)
    out["ball_recovery_recovery_failure"] = out["ball_recovery.recovery_failure"]
    
    out["duel.type.name"] = df.get("duel.type.name")
    out["duel.outcome.name"] = df.get("duel.outcome.name")
    out["duel_type"] = out["duel.type.name"]
    out["duel_outcome"] = out["duel.outcome.name"]
    
    out["counterpress"] = df.get("counterpress", False)
    
    out["block.deflection"] = df.get("block.deflection", False)
    out["block_deflection"] = out["block.deflection"]
    
    out["clearance.aerial_won"] = df.get("clearance.aerial_won", False)
    out["clearance.body_part.name"] = df.get("clearance.body_part.name")
    out["clearance_aerial_won"] = out["clearance.aerial_won"]
    out["clearance_body_part"] = out["clearance.body_part.name"]
    
    out["related_events"] = df.get("related_events")
    
    out["carry.end_location"] = df.get("carry.end_location")
    out["carry_end_location"] = out["carry.end_location"]
    
    out["tactics.formation"] = df.get("tactics.formation")
    out["tactics_formation"] = out["tactics.formation"]
    
    out["play_pattern.name"] = df.get("play_pattern.name")
    out["play_pattern_name"] = out["play_pattern.name"]
    
    out["team.id"] = out["team_id"]
    out["possession_team.id"] = out["possession_team_id"]
    out["outcome_name"] = coalesce_outcome(df)
    out["timestamp"] = pd.to_timedelta(out["timestamp"])
    
    return out


def get_feature_modules() -> List[Tuple[str, callable, Dict]]:
    """
    Get list of feature modules with their compute functions and parameters.
    
    Returns:
    --------
    List of tuples: (module_name, compute_function, extra_kwargs)
    """
    return [
        ("access_control", compute_access_control_features, {}),
        ("defensive_actions", compute_defensive_action_features, {}),
        ("pressure_tempo", compute_pressure_tempo_features, {}),
        ("spatial_compactness", compute_spatial_compactness_features, {}),
        ("passing_features", compute_passing_features, {}),
        ("carrying_features", compute_carrying_features, {}),
        ("recovery_transition", compute_recovery_transition_features, {}),
        ("temporal_features", compute_temporal_features, {}),
        ("zone_specific", compute_zone_specific_features, {}),
        ("player_tactical", compute_player_tactical_features, {}),
        ("composite_features", compute_composite_features, {"other_features": None}),
        ("contextual_features", compute_contextual_features, {}),
    ]


def compute_features_for_match(
    events: pd.DataFrame,
    team_id: int,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0,
    match_features: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Compute all features for a single match.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Cleaned events DataFrame for the match
    team_id : int
        Team ID
    midfield_x_min : float
        Minimum x-coordinate for midfield
    midfield_x_max : float
        Maximum x-coordinate for midfield
    match_features : Dict, optional
        Existing match features dict (for composite features that need other features)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of all computed features
    """
    if match_features is None:
        match_features = {}
    
    feature_modules = get_feature_modules()
    
    for module_name, compute_func, extra_kwargs in feature_modules:
        try:
            kwargs = {
                "events": events,
                "team_id": team_id,
                "midfield_x_min": midfield_x_min,
                "midfield_x_max": midfield_x_max,
                **extra_kwargs
            }
            
            if module_name == "composite_features":
                kwargs["other_features"] = match_features
            
            features = compute_func(**kwargs)
            match_features.update(features)
            
        except Exception as e:
            print(f"   Warning: {module_name} features failed: {e}")
    
    return match_features


def compute_all_features(
    config_path: Optional[Path] = None,
    team_id: Optional[int] = None,
    midfield_x_min: float = 40.0,
    midfield_x_max: float = 80.0
) -> pd.DataFrame:
    """
    Main function to compute all features for team and season specified in config.yaml.
    
    Parameters:
    -----------
    config_path : Path, optional
        Path to config.yaml file (default: config/config.yaml)
    team_id : int, optional
        Team ID to use. If None, will be inferred from team name in config.
    midfield_x_min : float
        Minimum x-coordinate for midfield (default: 40.0)
    midfield_x_max : float
        Maximum x-coordinate for midfield (default: 80.0)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per match and one column per feature
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
    print("Computing features per match...")
    print("-" * 80)
    
    all_match_features = []
    
    for match_idx, json_file in enumerate(json_files, 1):
        match_id = int(json_file.stem) if json_file.stem.isdigit() else json_file.stem
        print(f"Match {match_idx}/{len(json_files)}: {match_id}")
        
        try:
            df = load_single_json(json_file)
            events = clean_events(df)
            
            match_features = {
                "match_id": match_id,
                "team_name": team_name,
                "season": season,
                "team_id": team_id,
                "computed_at": datetime.now().isoformat()
            }
            
            match_features = compute_features_for_match(
                events=events,
                team_id=team_id,
                midfield_x_min=midfield_x_min,
                midfield_x_max=midfield_x_max,
                match_features=match_features
            )
            
            all_match_features.append(match_features)
            print(f"   Computed {len(match_features) - 5} features for match {match_id}")
            
        except Exception as e:
            print(f"   Error processing match {match_id}: {e}")
            continue
    
    print()
    print("=" * 80)
    print("Feature Engineering Complete")
    print("=" * 80)
    print(f"Processed {len(all_match_features)} matches")
    print()
    
    if not all_match_features:
        raise ValueError("No matches were successfully processed")
    
    features_df = pd.DataFrame(all_match_features)
    
    output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{team_normalized}_{season_normalized}_features.csv"
    output_path = output_dir / output_filename
    
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to: {output_path}")
    print(f"Shape: {features_df.shape} (rows=matches, columns=features)")
    print()
    
    return features_df


if __name__ == "__main__":
    features = compute_all_features()
    
    print("Feature Summary:")
    print("-" * 80)
    categories = [
        "access_control", "defensive", "pressure", "spatial", "passing",
        "carrying", "recovery", "temporal", "zone", "player", "composite", "contextual"
    ]
    for category in categories:
        category_features = [c for c in features.columns if category in c.lower()]
        if category_features:
            print(f"{category.capitalize()}: {len(category_features)} features")
