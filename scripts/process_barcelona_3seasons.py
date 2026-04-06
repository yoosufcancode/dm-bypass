"""
Download event files and compute midfielder features for Barcelona
across 3 seasons: 2012/2013, 2013/2014, 2014/2015.

Usage:
    python scripts/process_barcelona_3seasons.py
    python scripts/process_barcelona_3seasons.py --skip-download
"""

import sys
import argparse
import tempfile
import traceback
from pathlib import Path
import pandas as pd
import yaml

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
    POSITION_LABELS,
)
from src.features.main_feature import compute_all_features

TEAM_NAME = "Barcelona"
TEAM_ID   = 217
SEASONS   = ["2012/2013", "2013/2014", "2014/2015"]


def process_barcelona(skip_download: bool = False) -> None:
    base_dir    = ROOT / "data" / "raw"
    events_base = base_dir / "events"
    output_dir  = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_features = load_selected_features(output_dir)
    print(f"Loaded {len(selected_features)} selected features: {selected_features}\n")

    season_dfs = []

    for season in SEASONS:
        season_slug = season.replace("/", "_")
        feat_path   = output_dir / f"Barcelona_{season_slug}_features.csv"

        print("=" * 70)
        print(f"Season: {season}")
        print("=" * 70)

        # ── Already processed? ────────────────────────────────────────────
        if feat_path.exists():
            print(f"  Features CSV already exists — loading from disk.")
            df = pd.read_csv(feat_path)
            print(f"  Rows: {df.shape[0]},  unique matches: {df['match_id'].nunique()}")
            season_dfs.append(df)
            continue

        # ── Find match IDs ────────────────────────────────────────────────
        match_ids = find_team_matches_for_season(TEAM_ID, season, base_dir)
        if not match_ids:
            print(f"  ⚠️  No matches found — skipping")
            continue
        print(f"  Matches found: {len(match_ids)}")

        # ── Create directory + download events ────────────────────────────
        team_dir = create_team_season_directory(TEAM_NAME, season, events_base)
        print(f"  Directory: {team_dir}")

        if not skip_download:
            stats = move_event_files(
                match_ids,
                source_dir=events_base,
                target_dir=team_dir,
                copy=True,
                download_missing=True,
            )
            present = len([m for m in match_ids if (team_dir / f"{m}.json").exists()])
            print(f"  Event files present: {present}/{len(match_ids)}"
                  f"  (downloaded={stats['downloaded']}, copied={stats['copied']})")
        else:
            present = len(list(team_dir.glob("*.json")))
            print(f"  Skipping download — {present} files already present")

        # ── Compute features ──────────────────────────────────────────────
        tmp_cfg = write_temp_config(TEAM_NAME, season)
        try:
            df = compute_all_features(config_path=tmp_cfg, team_id=TEAM_ID)
        finally:
            tmp_cfg.unlink(missing_ok=True)

        df.to_csv(feat_path, index=False)
        print(f"  ✓ Features saved: {feat_path}  ({df.shape[0]} rows)")
        season_dfs.append(df)

    if not season_dfs:
        print("No data processed.")
        return

    # ── Combine all 3 seasons ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Combining 3 seasons")
    print("=" * 70)

    combined = pd.concat(season_dfs, ignore_index=True)
    combined_feat_path = output_dir / "Barcelona_3seasons_features.csv"
    combined.to_csv(combined_feat_path, index=False)
    print(f"  ✓ Combined features: {combined_feat_path}")
    print(f"    Rows: {len(combined)},  seasons: {sorted(combined['season'].unique())}")

    # ── Player selected features (3 seasons) ─────────────────────────────
    sel_path = output_dir / "Barcelona_3seasons_player_selected_features.csv"
    save_player_selected_features(combined, selected_features, sel_path)
    print(f"  ✓ Player selected features: {sel_path}")

    # ── Player midfield summary (3 seasons) ──────────────────────────────
    summary_path = output_dir / "Barcelona_3seasons_player_midfield_summary.csv"
    save_player_midfield_summary(
        save_player_selected_features(combined, selected_features, sel_path),
        summary_path,
    )
    print(f"  ✓ Player midfield summary: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    process_barcelona(skip_download=args.skip_download)
