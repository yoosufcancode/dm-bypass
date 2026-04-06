"""
Download event files and compute midfielder features for ALL teams in a given season.

Reuses the existing building blocks:
  - create_team_season_directory.py  →  download events per team
  - src/features/main_feature.py     →  compute midfielder features

Usage:
    python scripts/process_all_teams.py
    python scripts/process_all_teams.py --season "2014/2015"
    python scripts/process_all_teams.py --season "2014/2015" --skip-download
"""

import json
import yaml
import argparse
import sys
import tempfile
import traceback
import requests
import time
from pathlib import Path
import pandas as pd

# ── project root on sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.create_team_season_directory import (
    find_team_matches_for_season,
    create_team_season_directory,
    move_event_files,
)
from src.features.main_feature import compute_all_features


# ── helpers ────────────────────────────────────────────────────────────────

STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _download_json(url: str, dest: Path) -> bool:
    """Download a JSON file from StatsBomb open-data. Returns True on success."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        dest.write_text(response.text, encoding="utf-8")
        time.sleep(0.2)
        return True
    except Exception as e:
        print(f"    ⚠️  Could not download {url}: {e}")
        return False


def get_all_teams_for_season(season_name: str, base_dir: Path) -> list[dict]:
    """
    Return a list of dicts {team_id, team_name, competition} for every team
    that appears in ANY competition/league for the given season.

    Missing match index files are automatically downloaded from StatsBomb
    open-data so that all leagues — not just those already cached locally —
    are included.
    """
    competitions_file = base_dir / "competitions.json"
    if not competitions_file.exists():
        print("  Downloading competitions.json ...")
        _download_json(f"{STATSBOMB_BASE}/competitions.json", competitions_file)

    with open(competitions_file) as f:
        competitions = json.load(f)

    # Filter to the requested season across ALL competitions/leagues
    season_comps = [c for c in competitions if c.get("season_name") == season_name]
    if not season_comps:
        raise ValueError(f"No competitions found for season '{season_name}'")

    print(f"  Season '{season_name}' found in {len(season_comps)} competition(s):")
    for c in season_comps:
        print(f"    • {c['competition_name']} (comp_id={c['competition_id']}, season_id={c['season_id']})")
    print()

    teams: dict[int, dict] = {}  # team_id → {team_name, competitions}

    for comp in season_comps:
        comp_id = comp["competition_id"]
        season_id = comp["season_id"]
        comp_name = comp["competition_name"]
        matches_file = base_dir / "matches" / str(comp_id) / f"{season_id}.json"

        # Download match index if not cached locally
        if not matches_file.exists():
            url = f"{STATSBOMB_BASE}/matches/{comp_id}/{season_id}.json"
            print(f"  Downloading match index for {comp_name} ...")
            if not _download_json(url, matches_file):
                print(f"  ⚠️  Skipping {comp_name} — could not fetch match index")
                continue

        with open(matches_file) as f:
            matches = json.load(f)

        for match in matches:
            for side, id_key, name_key in [
                ("home_team", "home_team_id", "home_team_name"),
                ("away_team", "away_team_id", "away_team_name"),
            ]:
                t = match.get(side, {})
                if isinstance(t, dict):
                    tid = t.get(id_key)
                    tname = t.get(name_key)
                    if tid and tname:
                        if tid not in teams:
                            teams[tid] = {"team_id": tid, "team_name": tname, "competitions": set()}
                        teams[tid]["competitions"].add(comp_name)

    # Convert sets to sorted strings for display
    result = []
    for tid, info in sorted(teams.items()):
        result.append({
            "team_id": info["team_id"],
            "team_name": info["team_name"],
            "competitions": ", ".join(sorted(info["competitions"])),
        })
    return result


POSITION_LABELS = {
    0: "Defensive Midfield",
    1: "Center Midfield",
    2: "Attacking Midfield",
    3: "Wing Midfield",
    4: "Wing Back",
    5: "Midfield (generic)",
}

PLAYER_METADATA_COLS = [
    "player_id", "player_name", "midfielder_type",
    "match_id", "team_id", "team_name", "season", "bypasses_per_halftime",
]


def load_selected_features(processed_dir: Path) -> list[str]:
    """
    Load the 10 features selected during Barcelona feature selection.
    Reads from Barcelona_2014_2015_selected_features_list.txt.
    """
    feat_file = processed_dir / "Barcelona_2014_2015_selected_features_list.txt"
    if not feat_file.exists():
        raise FileNotFoundError(
            f"Feature list not found at {feat_file}. "
            "Run feature_selection.ipynb for Barcelona first."
        )
    features = [line.strip() for line in feat_file.read_text().splitlines() if line.strip()]
    return features


def save_player_selected_features(
    features_df: pd.DataFrame,
    selected_features: list[str],
    out_path: Path,
) -> pd.DataFrame:
    """
    Extract the selected features at individual player level and save as CSV.
    Mirrors the faa5efc5 cell in feature_selection.ipynb.
    """
    available = [f for f in selected_features if f in features_df.columns]
    missing = [f for f in selected_features if f not in features_df.columns]
    if missing:
        print(f"    ⚠️  Features absent in this team's data (skipped): {missing}")

    cols = [c for c in PLAYER_METADATA_COLS if c in features_df.columns] + available
    player_df = features_df[cols].copy()
    player_df.to_csv(out_path, index=False)
    return player_df


def save_player_midfield_summary(
    player_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Build a per-player midfield appearance summary and save as CSV.
    Mirrors the ff326d1d cell in feature_selection.ipynb.
    """
    player_df = player_df.copy()
    player_df["base_match_id"] = player_df["match_id"].str.replace(
        r"_P\d+$", "", regex=True
    )

    summary = (
        player_df.groupby(["player_id", "player_name", "midfielder_type"])
        .agg(match_halves=("match_id", "count"), matches_played=("base_match_id", "nunique"))
        .reset_index()
        .sort_values("matches_played", ascending=False)
        .reset_index(drop=True)
    )
    summary["position"] = summary["midfielder_type"].map(POSITION_LABELS).fillna("Unknown")
    summary[["player_id", "player_name", "position", "matches_played", "match_halves"]].to_csv(
        out_path, index=False
    )


def write_temp_config(team_name: str, season: str) -> Path:
    """Write a temporary config.yaml for the given team/season and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="cfg_"
    )
    yaml.dump({"Dataset": {"team_name": team_name, "season": season}}, tmp)
    tmp.close()
    return Path(tmp.name)


# ── main pipeline ──────────────────────────────────────────────────────────

def process_all_teams(season: str, skip_download: bool = False) -> None:
    base_dir = ROOT / "data" / "raw"
    events_base = base_dir / "events"
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Load the 10 features selected from Barcelona's feature selection notebook
    print("=" * 80)
    selected_features = load_selected_features(output_dir)
    print(f"Loaded {len(selected_features)} selected features from Barcelona analysis:")
    for f in selected_features:
        print(f"  • {f}")
    print()

    # 1. Discover all teams in this season
    print("=" * 80)
    print(f"Finding all teams in season: {season}")
    print("=" * 80)
    teams = get_all_teams_for_season(season, base_dir)
    print(f"Found {len(teams)} teams across all leagues\n")
    for t in teams:
        print(f"  {t['team_id']:>6}  {t['team_name']:<35}  [{t['competitions']}]")

    # 2. Process each team
    successful, failed = [], []

    for idx, team in enumerate(teams, 1):
        team_name = team["team_name"]
        team_id = team["team_id"]
        print()
        print("=" * 80)
        print(f"[{idx}/{len(teams)}]  {team_name}  (id={team_id})  [{team['competitions']}]")
        print("=" * 80)

        try:
            # ── Step A: find match IDs ───────────────────────────────────
            match_ids = find_team_matches_for_season(team_id, season, base_dir)
            if not match_ids:
                print(f"  ⚠️  No matches found for {team_name} in {season} — skipping")
                failed.append({"team": team_name, "reason": "no matches found"})
                continue
            print(f"  Matches found: {len(match_ids)}")

            # ── Step B: create directory + download events ───────────────
            team_dir = create_team_season_directory(team_name, season, events_base)
            print(f"  Directory: {team_dir}")

            if not skip_download:
                stats = move_event_files(
                    match_ids,
                    source_dir=events_base,
                    target_dir=team_dir,
                    copy=True,           # copy so originals stay in place
                    download_missing=True,
                )
                already = len([m for m in match_ids if (team_dir / f"{m}.json").exists()])
                print(f"  Event files present: {already}/{len(match_ids)}"
                      f"  (downloaded={stats['downloaded']}, copied={stats['copied']})")
            else:
                present = len(list(team_dir.glob("*.json")))
                print(f"  Skipping download — {present} files already in directory")

            # ── Step C: compute features ─────────────────────────────────
            tmp_config = write_temp_config(team_name, season)
            try:
                features_df = compute_all_features(
                    config_path=tmp_config,
                    team_id=team_id,
                )
            finally:
                tmp_config.unlink(missing_ok=True)

            # ── Step D: save full per-team features CSV ──────────────────
            team_slug = team_name.replace(" ", "_").replace("/", "_").replace("-", "_")
            season_slug = season.replace("/", "_")
            out_path = output_dir / f"{team_slug}_{season_slug}_features.csv"
            features_df.to_csv(out_path, index=False)
            print(f"  ✓ Features saved: {out_path}  ({features_df.shape[0]} rows)")

            # ── Step E: player-level selected features CSV ───────────────
            player_feat_path = output_dir / f"{team_slug}_{season_slug}_player_selected_features.csv"
            player_df = save_player_selected_features(features_df, selected_features, player_feat_path)
            print(f"  ✓ Player selected features: {player_feat_path}"
                  f"  ({player_df['player_id'].nunique()} players, "
                  f"{player_df['match_id'].nunique()} match-halves)")

            # ── Step F: player midfield summary CSV ──────────────────────
            summary_path = output_dir / f"{team_slug}_{season_slug}_player_midfield_summary.csv"
            save_player_midfield_summary(player_df, summary_path)
            print(f"  ✓ Player midfield summary: {summary_path}")

            successful.append(team_name)

        except Exception as e:
            print(f"  ✗ ERROR processing {team_name}: {e}")
            traceback.print_exc()
            failed.append({"team": team_name, "reason": str(e)})

    # 3. Combine all midfield summaries into one master table
    season_slug = season.replace("/", "_")
    summary_files = sorted(output_dir.glob(f"*_{season_slug}_player_midfield_summary.csv"))
    if summary_files:
        combined = pd.concat(
            [pd.read_csv(f).assign(team_name=f.name.replace(f"_{season_slug}_player_midfield_summary.csv", "").replace("_", " "))
             for f in summary_files],
            ignore_index=True,
        )
        combined = combined[["team_name", "player_id", "player_name", "position", "matches_played", "match_halves"]]
        combined = combined.sort_values(["team_name", "matches_played"], ascending=[True, False]).reset_index(drop=True)
        combined_path = output_dir / f"all_teams_{season_slug}_player_midfield_summary.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  ✓ Combined midfield summary: {combined_path}"
              f"  ({combined['team_name'].nunique()} teams, {len(combined)} player-team rows)")
        for f in summary_files:
            f.unlink()
        print(f"  ✓ Deleted {len(summary_files)} individual summary files")

    # 4. Final summary
    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Successful : {len(successful)}")
    print(f"  Failed     : {len(failed)}")
    if failed:
        print("\n  Failed teams:")
        for f in failed:
            print(f"    - {f['team']}: {f['reason']}")
    print(f"\n  Per-team CSVs saved to: {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download events + compute features for all teams in a season."
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2014/2015",
        help='Season name (default: "2014/2015")',
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading event files (use what is already on disk)",
    )
    args = parser.parse_args()

    process_all_teams(season=args.season, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
