"""
Process Barcelona midfielder features across all available La Liga seasons.

Usage:
    python scripts/process_barcelona_all_seasons.py
    python scripts/process_barcelona_all_seasons.py --skip-download
"""

import sys
import argparse
import traceback
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.create_team_season_directory import (
    find_team_matches_for_season,
    create_team_season_directory,
    move_event_files,
)
from scripts.process_all_teams import (
    load_selected_features,
    save_player_selected_features,
    save_player_midfield_summary,
    write_temp_config,
)
from src.features.main_feature import compute_all_features

TEAM_NAME = "Barcelona"
TEAM_ID   = 217

# All La Liga seasons with full match data (skip 1973/74 — only 1 match)
SEASONS = [
    "2004/2005", "2005/2006", "2006/2007", "2007/2008", "2008/2009",
    "2009/2010", "2010/2011", "2011/2012", "2012/2013", "2013/2014",
    "2014/2015", "2015/2016", "2016/2017", "2017/2018", "2018/2019",
    "2019/2020", "2020/2021",
]


def process_all_seasons(skip_download: bool = False) -> None:
    base_dir    = ROOT / "data" / "raw"
    events_base = base_dir / "events"
    output_dir  = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_features = load_selected_features(output_dir)
    print(f"Selected features ({len(selected_features)}): {selected_features}\n")

    season_dfs = []

    for season in SEASONS:
        season_slug = season.replace("/", "_")
        feat_path   = output_dir / f"Barcelona_{season_slug}_features.csv"

        print("=" * 70)
        print(f"Season: {season}")
        print("=" * 70)

        # Already processed?
        if feat_path.exists():
            print(f"  Cached — loading from disk.")
            df = pd.read_csv(feat_path)
            print(f"  Rows: {df.shape[0]},  matches: {df['match_id'].nunique()}")
            season_dfs.append(df)
            continue

        # Find match IDs
        match_ids = find_team_matches_for_season(TEAM_ID, season, base_dir)
        if not match_ids:
            print(f"  No matches found — skipping")
            continue
        print(f"  Matches: {len(match_ids)}")

        # Create directory + download events
        team_dir = create_team_season_directory(TEAM_NAME, season, events_base)

        if not skip_download:
            stats = move_event_files(
                match_ids,
                source_dir=events_base,
                target_dir=team_dir,
                copy=True,
                download_missing=True,
            )
            present = len([m for m in match_ids if (team_dir / f"{m}.json").exists()])
            print(f"  Event files: {present}/{len(match_ids)} (dl={stats['downloaded']}, cp={stats['copied']})")
        else:
            present = len(list(team_dir.glob("*.json")))
            print(f"  Skip download — {present} files present")

        # Compute features
        tmp_cfg = write_temp_config(TEAM_NAME, season)
        try:
            df = compute_all_features(config_path=tmp_cfg, team_id=TEAM_ID)
        except Exception as e:
            print(f"  ERROR computing features: {e}")
            traceback.print_exc()
            tmp_cfg.unlink(missing_ok=True)
            continue
        finally:
            tmp_cfg.unlink(missing_ok=True)

        df.to_csv(feat_path, index=False)
        print(f"  Saved: {feat_path}  ({df.shape[0]} rows)")
        season_dfs.append(df)

    if not season_dfs:
        print("No data processed.")
        return

    # Combine all seasons
    print("\n" + "=" * 70)
    print(f"Combining {len(season_dfs)} seasons")
    print("=" * 70)

    combined = pd.concat(season_dfs, ignore_index=True)
    out_path = output_dir / "Barcelona_allseasons_features.csv"
    combined.to_csv(out_path, index=False)
    print(f"  Combined: {out_path}")
    print(f"  Rows: {len(combined)},  matches: {combined['match_id'].nunique()}")
    print(f"  Seasons: {sorted(combined['season'].unique())}")

    # Player selected features
    sel_path = output_dir / "Barcelona_allseasons_player_selected_features.csv"
    save_player_selected_features(combined, selected_features, sel_path)
    print(f"  Player selected: {sel_path}")

    # Player midfield summary
    summary_path = output_dir / "Barcelona_allseasons_player_midfield_summary.csv"
    filtered = pd.read_csv(sel_path)
    save_player_midfield_summary(filtered, summary_path)
    print(f"  Summary: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    process_all_seasons(skip_download=args.skip_download)
