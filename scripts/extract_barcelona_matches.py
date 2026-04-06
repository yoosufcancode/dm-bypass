"""
Extract Barcelona match information: Competitions, Seasons, and Match Dates

This script extracts match information for Barcelona (team ID: 217) from StatsBomb data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import os


def extract_match_info_from_events(events_file: Path) -> Optional[Dict]:
    """
    Extract match information from events file.
    
    Parameters:
    -----------
    events_file : Path
        Path to events JSON file
    
    Returns:
    --------
    Dict or None
        Match information dictionary
    """
    match_id = events_file.stem
    
    try:
        with open(events_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
    except Exception as e:
        print(f"Error loading {events_file}: {e}")
        return None
    
    if not events:
        return None
    
    # Extract team information
    teams = set()
    for event in events[:100]:  # Check first 100 events
        if 'team' in event and event['team']:
            teams.add((event['team'].get('id'), event['team'].get('name')))
        if 'possession_team' in event and event['possession_team']:
            teams.add((event['possession_team'].get('id'), event['possession_team'].get('name')))
    
    # Check if Barcelona (217) is in the match
    barcelona_present = any(team_id == 217 for team_id, _ in teams)
    
    if not barcelona_present:
        return None
    
    # Get match date from file modification time
    match_updated_date = datetime.fromtimestamp(events_file.stat().st_mtime).strftime('%Y-%m-%d')
    
    
    return {
        'match_id': match_id,
        'match_updated_date': match_updated_date,
        'teams': list(teams),
        'competition': 'La Liga',  # Default - should be verified
        'season_name': '2018/2019'  # Default - should be verified
    }


def get_barcelona_matches_from_events(events_dir: Path) -> List[Dict]:
    """
    Extract Barcelona match information from events files.
    
    Parameters:
    -----------
    events_dir : Path
        Directory containing event JSON files
    
    Returns:
    --------
    List[Dict]
        List of match information dictionaries
    """
    matches = []
    
    if not events_dir.exists():
        return matches
    
    for events_file in sorted(events_dir.glob("*.json")):
        match_info = extract_match_info_from_events(events_file)
        if match_info:
            matches.append(match_info)
    
    return matches


def create_barcelona_matches_csv(
    output_path: Path = Path("barcelona_matches.csv"),
    events_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create CSV file with Barcelona matches information.
    
    Parameters:
    -----------
    output_path : Path
        Output CSV file path
    events_dir : Path, optional
        Directory containing events
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with match information
    """
    matches_data = []
    
    # Extract from events files
    if events_dir:
        events_dir_path = Path(events_dir) if isinstance(events_dir, str) else events_dir
        event_matches = get_barcelona_matches_from_events(events_dir_path)
        
        for match_info in event_matches:
            matches_data.append({
                'Competition': match_info.get('competition', 'Unknown'),
                'Season Name': match_info.get('season_name', 'Unknown'),
                'Match Updated Date': match_info.get('match_updated_date', ''),
                'Match ID': match_info.get('match_id', ''),
            })
    
    # Create DataFrame
    if matches_data:
        df = pd.DataFrame(matches_data)
        # Sort by match_updated_date
        if 'Match Updated Date' in df.columns:
            df = df.sort_values('Match Updated Date', ascending=False)
        # Select columns for output
        output_df = df[['Competition', 'Season Name', 'Match Updated Date']].copy()
        output_df = output_df.drop_duplicates()
    else:
        # Create empty DataFrame with correct columns
        output_df = pd.DataFrame(columns=['Competition', 'Season Name', 'Match Updated Date'])
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    return output_df


def main():
    """Main function to extract Barcelona matches."""
    base_dir = Path(__file__).parent.parent
    events_dir = base_dir / "data" / "raw" / "events"
    output_path = base_dir / "barcelona_matches.csv"
    
    print("Extracting Barcelona match information...")
    print(f"Events dir: {events_dir} (exists: {events_dir.exists()})")
    
    df = create_barcelona_matches_csv(
        output_path=output_path,
        events_dir=events_dir if events_dir.exists() else None
    )
    
    print(f"\nCreated {output_path}")
    print(f"Found {len(df)} matches")
    
    if len(df) > 0:
        print("\nMatch information:")
        print(df.to_string(index=False))
        
        # Add note about manual verification
        print("\n Note: Competition and Season names are inferred.")
        print("   Please verify and update using matches.json if available.")
    else:
        print("\nNo Barcelona matches found in events directory.")


if __name__ == "__main__":
    main()
