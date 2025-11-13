"""
Create a directory for team-season event data and move relevant event files.

This script reads the team name and season from config.yaml and creates
a directory in data/raw/events with the naming convention: team_season,
then moves/copies the relevant event files into that directory.

Usage:
    python scripts/create_team_season_directory.py
"""

import yaml
from pathlib import Path
import sys
import json
import shutil
import requests
import time


def load_config(config_path: Path = Path("config/config.yaml")) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : Path
        Path to config.yaml file
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_team_id_from_name(team_name: str, competitions_file: Path = Path("data/raw/competitions.json")) -> int:
    """
    Get team ID from team name by searching through match files.
    
    Parameters:
    -----------
    team_name : str
        Team name to search for
    competitions_file : Path
        Path to competitions.json
    
    Returns:
    --------
    int or None
        Team ID if found, None otherwise
    """
    # Common team IDs (can be expanded)
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
    
    # Try direct lookup first
    if team_name in team_name_to_id:
        return team_name_to_id[team_name]
    
    # If not found, search through matches (this is slower but more comprehensive)
    matches_dir = Path("data/raw/matches")
    if matches_dir.exists():
        for comp_dir in matches_dir.iterdir():
            if comp_dir.is_dir():
                for season_file in comp_dir.glob("*.json"):
                    try:
                        with open(season_file, 'r') as f:
                            matches = json.load(f)
                        
                        for match in matches:
                            home_team = match.get('home_team', {})
                            away_team = match.get('away_team', {})
                            
                            if isinstance(home_team, dict):
                                if home_team.get('home_team_name') == team_name:
                                    return home_team.get('home_team_id')
                            
                            if isinstance(away_team, dict):
                                if away_team.get('away_team_name') == team_name:
                                    return away_team.get('away_team_id')
                    except:
                        pass
    
    return None


def find_team_matches_for_season(team_id: int, season: str, base_dir: Path = Path("data/raw")) -> list:
    """
    Find all match IDs for a team in a given season.
    
    Parameters:
    -----------
    team_id : int
        Team ID
    season : str
        Season name (e.g., "2017/2018")
    base_dir : Path
        Base directory for data
    
    Returns:
    --------
    list
        List of match IDs
    """
    matches_dir = base_dir / "matches"
    match_ids = []
    
    if not matches_dir.exists():
        return match_ids
    
    # Load competitions to find which competitions have this season
    competitions_file = base_dir / "competitions.json"
    if not competitions_file.exists():
        return match_ids
    
    with open(competitions_file, 'r') as f:
        competitions = json.load(f)
    
    # Find competitions for this season
    for comp in competitions:
        if comp.get('season_name') == season:
            comp_id = comp.get('competition_id')
            season_id = comp.get('season_id')
            
            matches_file = matches_dir / str(comp_id) / f"{season_id}.json"
            
            if matches_file.exists():
                try:
                    with open(matches_file, 'r') as f:
                        matches = json.load(f)
                    
                    for match in matches:
                        home_team = match.get('home_team', {})
                        away_team = match.get('away_team', {})
                        
                        home_team_id = home_team.get('home_team_id') if isinstance(home_team, dict) else None
                        away_team_id = away_team.get('away_team_id') if isinstance(away_team, dict) else None
                        
                        if home_team_id == team_id or away_team_id == team_id:
                            match_id = match.get('match_id')
                            if match_id:
                                match_ids.append(match_id)
                except Exception as e:
                    print(f"Error reading {matches_file}: {e}")
    
    return match_ids


def create_team_season_directory(team_name: str, season: str, base_dir: Path = Path("data/raw/events")) -> Path:
    """
    Create a directory for team-season event data.
    
    Parameters:
    -----------
    team_name : str
        Team name (e.g., "Barcelona")
    season : str
        Season name (e.g., "2017/2018")
    base_dir : Path
        Base directory for events (default: data/raw/events)
    
    Returns:
    --------
    Path
        Path to created directory
    """
    # Normalize team name: remove spaces, special characters, convert to title case
    team_normalized = team_name.replace(" ", "_").replace("/", "_").replace("-", "_")
    
    # Normalize season: replace "/" with "_"
    season_normalized = season.replace("/", "_")
    
    # Create directory name: team_season
    dir_name = f"{team_normalized}_{season_normalized}"
    
    # Create full path
    team_season_dir = base_dir / dir_name
    
    # Create directory if it doesn't exist
    team_season_dir.mkdir(parents=True, exist_ok=True)
    
    return team_season_dir


def download_event_file(match_id: int, target_dir: Path) -> bool:
    """
    Download event file from StatsBomb repository.
    
    Parameters:
    -----------
    match_id : int
        Match ID
    target_dir : Path
        Target directory to save the file
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
    target_file = target_dir / f"{match_id}.json"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Verify it's valid JSON
        with open(target_file, 'r') as f:
            json.load(f)
        
        # Small delay to avoid rate limiting
        time.sleep(0.3)
        
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return False
        print(f"HTTP Error {e.response.status_code} for match {match_id}")
        return False
    except Exception as e:
        print(f"Error downloading match {match_id}: {e}")
        return False


def move_event_files(match_ids: list, source_dir: Path, target_dir: Path, copy: bool = False, download_missing: bool = True) -> dict:
    """
    Move or copy event files to target directory.
    Optionally download missing files from StatsBomb.
    
    Parameters:
    -----------
    match_ids : list
        List of match IDs
    source_dir : Path
        Source directory containing event files
    target_dir : Path
        Target directory to move/copy files to
    copy : bool
        If True, copy files instead of moving (default: False)
    download_missing : bool
        If True, download missing files from StatsBomb (default: True)
    
    Returns:
    --------
    dict
        Statistics about the operation
    """
    stats = {
        'moved': 0,
        'copied': 0,
        'downloaded': 0,
        'not_found': 0,
        'errors': 0
    }
    
    for match_id in match_ids:
        source_file = source_dir / f"{match_id}.json"
        target_file = target_dir / f"{match_id}.json"
        
        # Skip if already in target directory
        if target_file.exists():
            print(f"  ✓ Match {match_id}: Already in target directory")
            continue
        
        if source_file.exists():
            try:
                if copy:
                    shutil.copy2(source_file, target_file)
                    stats['copied'] += 1
                    print(f"  ✓ Match {match_id}: Copied")
                else:
                    shutil.move(str(source_file), str(target_file))
                    stats['moved'] += 1
                    print(f"  ✓ Match {match_id}: Moved")
            except Exception as e:
                print(f"Error processing match {match_id}: {e}")
                stats['errors'] += 1
        elif download_missing:
            # Try to download from StatsBomb
            print(f"  Downloading match {match_id}...")
            if download_event_file(match_id, target_dir):
                stats['downloaded'] += 1
                print(f"    ✓ Downloaded match {match_id}")
            else:
                stats['not_found'] += 1
        else:
            stats['not_found'] += 1
    
    return stats


def main():
    """Main function."""
    try:
        # Load config
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        config = load_config(config_path)
        
        # Extract team name and season
        dataset_config = config.get("Dataset", {})
        team_name = dataset_config.get("team_name")
        season = dataset_config.get("season")
        
        if not team_name:
            raise ValueError("team_name not found in config.yaml")
        if not season:
            raise ValueError("season not found in config.yaml")
        
        print(f"Configuration loaded:")
        print(f"  Team: {team_name}")
        print(f"  Season: {season}")
        print()
        
        # Get team ID
        print("Finding team ID...")
        base_dir = Path(__file__).parent.parent / "data" / "raw"
        team_id = get_team_id_from_name(team_name, base_dir / "competitions.json")
        
        if not team_id:
            print(f"Could not find team ID for '{team_name}'")
            print("   Creating directory but cannot find matches without team ID")
            team_season_dir = create_team_season_directory(team_name, season, base_dir / "events")
            print(f"Created directory: {team_season_dir}")
            return 0
        
        print(f"  Found team ID: {team_id}")
        print()
        
        # Find matches for this team and season
        print(f"Finding matches for {team_name} in {season}...")
        match_ids = find_team_matches_for_season(team_id, season, base_dir)
        
        if not match_ids:
            print(f"No matches found for {team_name} in {season}")
            team_season_dir = create_team_season_directory(team_name, season, base_dir / "events")
            print(f"Created directory: {team_season_dir}")
            return 0
        
        print(f"  Found {len(match_ids)} matches")
        print()
        
        # Create directory
        events_base_dir = base_dir / "events"
        team_season_dir = create_team_season_directory(team_name, season, events_base_dir)
        
        print(f" Created directory: {team_season_dir}")
        print()
        
        # Move/copy/download event files
        print("Processing event files for team-season directory...")
        source_dir = events_base_dir
        stats = move_event_files(match_ids, source_dir, team_season_dir, copy=False, download_missing=True)
        
        print()
        print("=" * 80)
        print("Summary:")
        print("=" * 80)
        print(f"  Directory: {team_season_dir}")
        print(f"  Matches found: {len(match_ids)}")
        print(f"  Files moved: {stats['moved']}")
        print(f"  Files copied: {stats['copied']}")
        print(f"  Files downloaded: {stats['downloaded']}")
        print(f"  Files already present: {len(match_ids) - stats['moved'] - stats['copied'] - stats['downloaded'] - stats['not_found']}")
        print(f"  Files not found: {stats['not_found']}")
        print(f"  Errors: {stats['errors']}")
        print()
        
        total_processed = stats['moved'] + stats['copied'] + stats['downloaded']
        files_present = len([m for m in match_ids if (team_season_dir / f"{m}.json").exists()])
        
        if files_present > 0:
            print(f"  Total event files in directory: {files_present} out of {len(match_ids)} matches")
            print(f"   Directory: {team_season_dir}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

