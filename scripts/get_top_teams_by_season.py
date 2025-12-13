"""
Get top 10 teams with most match data in StatsBomb for a particular season.

Usage:
    python scripts/get_top_teams_by_season.py --season "2015/2016"
    python scripts/get_top_teams_by_season.py --season "2020/2021"
"""

import json
import pandas as pd
from pathlib import Path
import requests
from collections import defaultdict
import argparse


def get_top_teams_by_season(season_name: str, top_n: int = 10, output_file: str = None):
    """
    Get top N teams with most matches in a given season.
    
    Parameters:
    -----------
    season_name : str
        Season name (e.g., "2015/2016", "2020/2021")
    top_n : int
        Number of top teams to return (default: 10)
    output_file : str
        Optional CSV file path to save results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with top teams and their match counts
    """
    # Load competitions.json
    competitions_file = Path("data/raw/competitions.json")
    if not competitions_file.exists():
        print("⚠️  competitions.json not found. Downloading...")
        url = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            competitions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(competitions_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
        except Exception as e:
            print(f"❌ Could not download competitions.json: {e}")
            return pd.DataFrame()
    
    with open(competitions_file, 'r') as f:
        competitions = json.load(f)
    
    df = pd.DataFrame(competitions)
    
    # Filter for target season
    df_season = df[df['season_name'] == season_name].copy()
    
    if len(df_season) == 0:
        print(f"❌ No competitions found for season: {season_name}")
        return pd.DataFrame()
    
    print(f"Analyzing {season_name} season...")
    print(f"Found {len(df_season)} competitions")
    print("=" * 100)
    
    team_match_counts = defaultdict(int)
    team_competitions = defaultdict(set)
    total_matches_processed = 0
    
    for idx, row in df_season.iterrows():
        comp_id = row['competition_id']
        season_id = row['season_id']
        comp_name = row['competition_name']
        
        matches_file = Path(f"data/raw/matches/{comp_id}/{season_id}.json")
        
        # Download if not exists
        if not matches_file.exists():
            url = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{comp_id}/{season_id}.json"
            try:
                print(f"  Downloading {comp_name}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                matches_file.parent.mkdir(parents=True, exist_ok=True)
                with open(matches_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            except Exception as e:
                print(f"  ⚠️  Could not download {comp_name}: {e}")
                continue
        
        # Count matches per team
        if matches_file.exists():
            try:
                with open(matches_file, 'r') as f:
                    matches = json.load(f)
                
                total_matches_processed += len(matches)
                
                for match in matches:
                    home_team = match.get('home_team', {})
                    away_team = match.get('away_team', {})
                    
                    if isinstance(home_team, dict):
                        team_id = home_team.get('home_team_id')
                        team_name = home_team.get('home_team_name')
                        if team_id and team_name:
                            team_match_counts[(team_id, team_name)] += 1
                            team_competitions[(team_id, team_name)].add(comp_name)
                    
                    if isinstance(away_team, dict):
                        team_id = away_team.get('away_team_id')
                        team_name = away_team.get('away_team_name')
                        if team_id and team_name:
                            team_match_counts[(team_id, team_name)] += 1
                            team_competitions[(team_id, team_name)].add(comp_name)
            except Exception as e:
                print(f"  ⚠️  Error processing {comp_name}: {e}")
    
    if not team_match_counts:
        print("❌ No matches found for this season")
        return pd.DataFrame()
    
    # Create results
    results = []
    for (team_id, team_name), match_count in team_match_counts.items():
        competitions_list = ', '.join(sorted(team_competitions[(team_id, team_name)]))
        results.append({
            'Team ID': team_id,
            'Team Name': team_name,
            'Total Matches': match_count,
            'Competitions': competitions_list,
            'Number of Competitions': len(team_competitions[(team_id, team_name)])
        })
    
    # Sort by match count
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Total Matches', ascending=False)
    
    # Get top N
    top_teams = results_df.head(top_n)
    
    print()
    print(f"Total matches processed: {total_matches_processed}")
    print(f"Total unique teams: {len(team_match_counts)}")
    print()
    print(f"Top {top_n} Teams by Match Count in {season_name}:")
    print("=" * 100)
    print(top_teams.to_string(index=False))
    
    # Save to CSV if requested
    if output_file:
        output_path = Path(output_file)
        top_teams.to_csv(output_path, index=False)
        print()
        print(f"✅ Saved to {output_path}")
    
    return top_teams


def main():
    parser = argparse.ArgumentParser(description='Get top teams by match count for a season')
    parser.add_argument('--season', type=str, required=True,
                       help='Season name (e.g., "2015/2016", "2020/2021")')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top teams to return (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    get_top_teams_by_season(args.season, args.top, args.output)


if __name__ == "__main__":
    main()

