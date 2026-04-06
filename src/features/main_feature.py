"""
Main feature engineering pipeline — Wyscout edition.

Loads Wyscout Open Dataset event files, computes all midfielder-level
features for every team in every match of a competition, and saves results.

Usage:
    python -m src.features.main_feature                          # La Liga (default)
    python -m src.features.main_feature --leagues Spain England  # multiple leagues
"""

import warnings
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.midfield import FEATURE_FUNCTIONS
from src.features.midfield.context import MidfieldFeatureContext
from src.features.midfield.independent_var import calculate_bypasses_per_match
from src.features.opponent_context import compute_opponent_context
from src.ingest.load_wyscout import (
    load_wyscout_events,
    load_wyscout_matches,
    load_wyscout_players,
    load_wyscout_teams,
    get_midfielder_ids_wyscout,
    get_all_team_ids_in_match,
    load_and_clean_match,
)

# Default data location
WYSCOUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "wyscout"
OUTPUT_DIR  = Path(__file__).parent.parent.parent / "data" / "processed"

AVAILABLE_LEAGUES = ["Spain", "England", "France", "Germany", "Italy"]


def compute_features_for_competition(
    league: str = "Spain",
    wyscout_dir: Path = WYSCOUT_DIR,
    max_matches: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute midfielder features for all teams in a full competition.

    Parameters
    ----------
    league : str
        League name matching the Wyscout file suffix (e.g. "Spain").
    wyscout_dir : Path
        Root directory of the Wyscout data files.
    max_matches : int, optional
        Limit processing to first N matches (useful for testing).

    Returns
    -------
    pd.DataFrame
        One row per (player, match, team, period).
    """
    print("=" * 80)
    print(f"Wyscout Feature Engineering Pipeline — {league}")
    print("=" * 80)

    # ── Load raw data ─────────────────────────────────────────────────────────
    print(f"Loading events for {league}...")
    raw_events_all = load_wyscout_events(wyscout_dir, league)
    print(f"  {len(raw_events_all):,} events loaded")

    print("Loading match metadata...")
    matches_df = load_wyscout_matches(wyscout_dir, league)
    print(f"  {len(matches_df)} matches")

    print("Loading player metadata...")
    players_dict = load_wyscout_players(wyscout_dir)

    teams_dict = load_wyscout_teams(wyscout_dir)

    # ── Iterate over matches ──────────────────────────────────────────────────
    all_rows = []
    match_list = matches_df.iterrows()
    n_matches = len(matches_df)
    if max_matches:
        import itertools
        match_list = itertools.islice(match_list, max_matches)
        n_matches = min(max_matches, n_matches)

    print(f"\nProcessing {n_matches} matches...")
    print("-" * 80)

    for match_idx, (_, match_row) in enumerate(match_list, 1):
        match_id = int(match_row.get("wyId") or match_row.get("game_id") or match_row.get("matchId") or 0)

        match_events_raw = raw_events_all[raw_events_all["matchId"] == match_id]
        if match_events_raw.empty:
            continue

        print(f"Match {match_idx}/{n_matches}: id={match_id}")

        try:
            events = load_and_clean_match(match_events_raw, match_row, players_dict, teams_dict)
        except Exception as e:
            print(f"  Error cleaning match {match_id}: {e}")
            continue

        team_ids = get_all_team_ids_in_match(match_row)
        if not team_ids:
            # Fallback: infer from events
            team_ids = [int(t) for t in events["team_id"].dropna().unique()]

        for team_id in team_ids:
            team_name_val = (teams_dict.get(team_id) or {}).get("name", f"Team_{team_id}")

            midfielder_ids = get_midfielder_ids_wyscout(match_row, team_id, players_dict)
            if not midfielder_ids:
                continue

            for period in [1, 2]:
                period_events = events[events["period"] == period]
                if period_events.empty:
                    continue

                ctx = MidfieldFeatureContext(
                    raw_events=period_events,  # Wyscout: clean events serve as both raw and clean
                    events=period_events,
                    team_id=team_id,
                    midfielder_ids=midfielder_ids,
                    match_id=f"{match_id}_P{period}",
                )

                try:
                    bypasses_count = calculate_bypasses_per_match(ctx)
                except Exception as e:
                    bypasses_count = 0

                try:
                    opp_ctx = compute_opponent_context(period_events, team_name_val, period)
                except Exception as e:
                    opp_ctx = {}

                for player_id in sorted(midfielder_ids):
                    player_events = period_events[period_events["player_id"] == player_id]
                    if player_events.empty:
                        continue  # Player didn't play this period

                    player_name_val = player_events["player_name"].iloc[0] if not player_events.empty else None

                    player_features = {
                        "player_id":            player_id,
                        "player_name":          player_name_val,
                        "match_id":             f"{match_id}_P{period}",
                        "team_id":              team_id,
                        "team_name":            team_name_val,
                        "league":               league,
                        "computed_at":          datetime.now().isoformat(),
                        "bypasses_per_halftime": bypasses_count,
                        **opp_ctx,
                    }

                    for feat_name, feat_func in FEATURE_FUNCTIONS.items():
                        try:
                            series = feat_func(ctx)
                            player_features[feat_name] = series.get(player_id)
                        except Exception:
                            player_features[feat_name] = np.nan

                    all_rows.append(player_features)

            print(f"  Teams processed: {len(team_ids)}")

    print()
    print("=" * 80)
    print("Feature Engineering Complete")
    print("=" * 80)

    if not all_rows:
        raise ValueError("No features were computed. Check that Wyscout data is present.")

    df = pd.DataFrame(all_rows)

    # Round numeric columns
    meta_cols = {"player_id", "player_name", "match_id", "team_id", "team_name",
                 "league", "computed_at", "bypasses_per_halftime"}
    for col in df.columns:
        if col not in meta_cols and df[col].dtype in (np.float64, np.float32):
            df[col] = df[col].round(4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"wyscout_{league}_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows × {len(df.columns)} cols → {out_path}")

    return df


def compute_features_multi_league(
    leagues: List[str] = None,
    wyscout_dir: Path = WYSCOUT_DIR,
    max_matches_per_league: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute and concatenate features across multiple leagues.

    Parameters
    ----------
    leagues : list of str
        League names to include. Defaults to all 5 major leagues.
    """
    if leagues is None:
        leagues = AVAILABLE_LEAGUES

    all_dfs = []
    for league in leagues:
        print(f"\n{'='*80}")
        print(f"Processing league: {league}")
        print(f"{'='*80}")
        try:
            df = compute_features_for_competition(
                league=league,
                wyscout_dir=wyscout_dir,
                max_matches=max_matches_per_league,
            )
            all_dfs.append(df)
        except FileNotFoundError as e:
            print(f"Skipping {league}: {e}")
        except Exception as e:
            import traceback
            print(f"Error processing {league}: {e}")
            traceback.print_exc()

    if not all_dfs:
        raise ValueError("No leagues were processed successfully.")

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = OUTPUT_DIR / "wyscout_all_leagues_features.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nAll leagues combined: {len(combined):,} rows → {out_path}")
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Wyscout midfielder features")
    parser.add_argument(
        "--leagues", nargs="+", default=["Spain"],
        choices=AVAILABLE_LEAGUES,
        help="Leagues to process (default: Spain)"
    )
    parser.add_argument(
        "--max-matches", type=int, default=None,
        help="Limit to N matches per league (for testing)"
    )
    parser.add_argument(
        "--all-leagues", action="store_true",
        help="Process all 5 major leagues"
    )
    args = parser.parse_args()

    leagues = AVAILABLE_LEAGUES if args.all_leagues else args.leagues

    if len(leagues) == 1:
        compute_features_for_competition(
            league=leagues[0],
            max_matches=args.max_matches,
        )
    else:
        compute_features_multi_league(
            leagues=leagues,
            max_matches_per_league=args.max_matches,
        )
