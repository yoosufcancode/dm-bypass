"""Stage 1: Data ingestion and feature engineering via Wyscout open dataset."""
import sys
from pathlib import Path
from typing import Callable

ALL_LEAGUES = ["Spain", "England", "France", "Germany", "Italy"]


def run_ingestion(
    leagues: list,
    skip_download: bool,
    progress_cb: Callable[[int, str], None],
) -> dict:
    """
    For each league, call compute_features_for_competition from
    src.features.main_feature. Skips leagues whose CSV already exists when
    skip_download is True.

    Returns dict matching IngestResult schema:
      {
        "features_paths": {league: path_str, ...},
        "row_counts":     {league: n_rows, ...},
        "leagues_processed": [...],
      }
    """
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.features.main_feature import compute_features_for_competition

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    leagues_processed: list[str] = []

    total = len(leagues)
    for i, league in enumerate(leagues):
        base_pct = int(i / total * 90)
        progress_cb(base_pct, f"Processing league: {league}")

        csv_path = processed_dir / f"wyscout_{league}_features.csv"

        if skip_download and csv_path.exists():
            progress_cb(base_pct + int(90 / total * 0.9), f"Skipping {league} — CSV already exists")
            import pandas as pd
            try:
                n = len(pd.read_csv(csv_path))
            except Exception:
                n = 0
            features_paths[league] = str(csv_path)
            row_counts[league] = n
            leagues_processed.append(league)
            continue

        # If skip_download is False or CSV doesn't exist, run feature engineering
        try:
            progress_cb(base_pct + 2, f"Computing features for {league}...")
            df = compute_features_for_competition(league=league)
            if df is not None and not df.empty:
                out_path = csv_path
                df.to_csv(out_path, index=False)
                features_paths[league] = str(out_path)
                row_counts[league] = len(df)
                leagues_processed.append(league)
                progress_cb(base_pct + int(90 / total * 0.95), f"Done {league}: {len(df)} rows")
            else:
                progress_cb(base_pct + int(90 / total * 0.95), f"Warning: no data returned for {league}")
        except Exception as exc:
            progress_cb(base_pct + int(90 / total * 0.95), f"Error processing {league}: {exc}")

    progress_cb(100, "Ingestion complete")

    return {
        "features_paths": features_paths,
        "row_counts": row_counts,
        "leagues_processed": leagues_processed,
    }
