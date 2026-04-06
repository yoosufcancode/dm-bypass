"""
Download StatsBomb open data for Barcelona matches.

This script helps download competitions.json and matches data from StatsBomb open data repository.
"""

import json
import requests
from pathlib import Path
from typing import Dict, List
import pandas as pd


def download_competitions_json(output_dir: Path = Path("data/raw")) -> Path:
    """
    Download competitions.json from StatsBomb open data repository.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save the file
    
    Returns:
    --------
    Path
        Path to downloaded file
    """
    url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
    output_path = output_dir / "competitions.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading competitions.json from {url}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✅ Downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"⚠️  Could not download competitions.json: {e}")
        print("   You may need to download it manually from:")
        print("   https://github.com/statsbomb/open-data/tree/master/data")
        return None


def get_barcelona_competitions(competitions_path: Path) -> pd.DataFrame:
    """
    Get all competitions where Barcelona (team ID 217) has matches.
    
    Parameters:
    -----------
    competitions_path : Path
        Path to competitions.json
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with competition and season information
    """
    if not competitions_path.exists():
        return pd.DataFrame()
    
    with open(competitions_path, 'r', encoding='utf-8') as f:
        competitions = json.load(f)
    
    return pd.DataFrame(competitions)


def download_matches_for_competition(competition_id: int, season_id: int, 
                                     output_dir: Path = Path("data/raw/matches")) -> Path:
    """
    Download matches.json for a specific competition and season.
    
    Parameters:
    -----------
    competition_id : int
        Competition ID
    season_id : int
        Season ID
    output_dir : Path
        Directory to save matches
    
    Returns:
    --------
    Path
        Path to downloaded file
    """
    url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{competition_id}/{season_id}.json"
    output_path = output_dir / str(competition_id) / f"{season_id}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return output_path
    except Exception as e:
        print(f"⚠️  Could not download matches for competition {competition_id}, season {season_id}: {e}")
        return None


def extract_barcelona_matches_from_matches_file(matches_path: Path, 
                                                 competition_name: str,
                                                 season_name: str,
                                                 barcelona_team_id: int = 217) -> List[Dict]:
    """
    Extract Barcelona matches from a matches.json file.
    
    Parameters:
    -----------
    matches_path : Path
        Path to matches.json file
    competition_name : str
        Competition name
    season_name : str
        Season name
    barcelona_team_id : int
        Barcelona team ID (default: 217)
    
    Returns:
    --------
    List[Dict]
        List of match dictionaries
    """
    matches = []
    
    if not matches_path.exists():
        return matches
    
    try:
        with open(matches_path, 'r', encoding='utf-8') as f:
            matches_data = json.load(f)
        
        for match in matches_data:
            home_team = match.get('home_team', {})
            away_team = match.get('away_team', {})
            
            home_team_id = home_team.get('home_team_id') if isinstance(home_team, dict) else None
            away_team_id = away_team.get('away_team_id') if isinstance(away_team, dict) else None
            
            if home_team_id == barcelona_team_id or away_team_id == barcelona_team_id:
                matches.append({
                    'Competition': competition_name,
                    'Season Name': season_name,
                    'Match Updated Date': match.get('last_updated', match.get('match_date', '')),
                    'Match ID': match.get('match_id', ''),
                    'Match Date': match.get('match_date', ''),
                    'Home Team': home_team.get('home_team_name', '') if isinstance(home_team, dict) else '',
                    'Away Team': away_team.get('away_team_name', '') if isinstance(away_team, dict) else '',
                    'Competition ID': match.get('competition', {}).get('competition_id', '') if isinstance(match.get('competition'), dict) else '',
                    'Season ID': match.get('season', {}).get('season_id', '') if isinstance(match.get('season'), dict) else '',
                })
    except Exception as e:
        print(f"Error processing {matches_path}: {e}")
    
    return matches


def get_all_barcelona_matches(output_csv: Path = Path("barcelona_matches.csv"),
                               download_data: bool = True) -> pd.DataFrame:
    """
    Get all Barcelona matches across all competitions and seasons.
    
    Parameters:
    -----------
    output_csv : Path
        Output CSV file path
    download_data : bool
        Whether to download data from StatsBomb repository
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all Barcelona matches
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw"
    
    # Download competitions.json if needed
    competitions_path = data_dir / "competitions.json"
    if download_data and not competitions_path.exists():
        download_competitions_json(data_dir)
    
    if not competitions_path.exists():
        print("⚠️  competitions.json not found. Cannot extract match information.")
        print("   Please download it from: https://github.com/statsbomb/open-data/tree/master/data")
        return pd.DataFrame()
    
    # Load competitions
    competitions_df = get_barcelona_competitions(competitions_path)
    
    if competitions_df.empty:
        print("⚠️  No competitions found in competitions.json")
        return pd.DataFrame()
    
    print(f"Found {len(competitions_df)} competition-season combinations")
    
    all_matches = []
    matches_dir = data_dir / "matches"
    
    # Process each competition and season
    for idx, comp in competitions_df.iterrows():
        competition_id = comp.get('competition_id')
        season_id = comp.get('season_id')
        competition_name = comp.get('competition_name', 'Unknown')
        season_name = comp.get('season_name', 'Unknown')
        
        print(f"\nProcessing: {competition_name} - {season_name} (Comp ID: {competition_id}, Season ID: {season_id})")
        
        # Check if matches file exists locally
        matches_file = matches_dir / str(competition_id) / f"{season_id}.json"
        
        # Download if needed
        if download_data and not matches_file.exists():
            download_matches_for_competition(competition_id, season_id, matches_dir)
        
        # Extract Barcelona matches
        if matches_file.exists():
            matches = extract_barcelona_matches_from_matches_file(
                matches_file, competition_name, season_name
            )
            all_matches.extend(matches)
            print(f"  Found {len(matches)} Barcelona matches")
        else:
            print(f"  ⚠️  Matches file not found: {matches_file}")
    
    # Create DataFrame
    if all_matches:
        df = pd.DataFrame(all_matches)
        # Sort by match updated date
        if 'Match Updated Date' in df.columns:
            df = df.sort_values('Match Updated Date', ascending=False)
        # Remove duplicates
        df = df.drop_duplicates(subset=['Match ID'], keep='first')
        
        # Select and order columns for output
        output_columns = ['Competition', 'Season Name', 'Match Updated Date']
        if all(col in df.columns for col in output_columns):
            output_df = df[output_columns].copy()
        else:
            output_df = df
        
        # Save to CSV
        output_path = base_dir / output_csv
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Created {output_path}")
        print(f"Total Barcelona matches found: {len(output_df)}")
        print(f"\nCompetitions and Seasons:")
        summary = output_df.groupby(['Competition', 'Season Name']).size().reset_index(name='Match Count')
        print(summary.to_string(index=False))
        
        return output_df
    else:
        print("\n⚠️  No Barcelona matches found")
        return pd.DataFrame()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract all Barcelona matches across competitions and seasons')
    parser.add_argument('--no-download', action='store_true', 
                       help='Do not download data from StatsBomb repository')
    parser.add_argument('--output', type=str, default='barcelona_matches.csv',
                       help='Output CSV filename')
    
    args = parser.parse_args()
    
    get_all_barcelona_matches(
        output_csv=Path(args.output),
        download_data=not args.no_download
    )


if __name__ == "__main__":
    main()

