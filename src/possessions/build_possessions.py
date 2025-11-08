"""
build_possessions.py — reconstruct possessions using possession_team_name
"""
from pathlib import Path
import pandas as pd

TEAM_COL = "possession_team_name"   # from ingest/clean_columns()

def build_possessions(events: pd.DataFrame) -> pd.DataFrame:
    # Ensure the expected columns exist
    needed = ["match_id", "period", "timestamp", TEAM_COL, "x", "y", "type_name"]
    missing = [c for c in needed if c not in events.columns]
    if missing:
        raise ValueError(f"Missing columns in events: {missing}")

    df = events.sort_values(["match_id", "period", "timestamp"]).copy()

    # Start a new possession when team changes OR period changes
    df["possession_change"] = (
        (df[TEAM_COL] != df[TEAM_COL].shift(1)) |
        (df["period"] != df["period"].shift(1))
    )
    df["poss_id"] = df["possession_change"].cumsum()

    # Summarize one row per possession
    poss = (
        df.groupby(["match_id", "poss_id", TEAM_COL], as_index=False)
          .agg(
              start_time=("timestamp", "min"),
              end_time=("timestamp", "max"),
              start_x=("x", "first"),
              start_y=("y", "first"),
              end_x=("x", "last"),
              end_y=("y", "last"),
              n_events=("type_name", "count"),
          )
          .rename(columns={TEAM_COL: "team_name"})
    )
    return poss


def main():
    src = Path("data/processed/events.parquet")
    dst = Path("data/processed/possessions.parquet")

    print(f"Reading {src} ...")
    events = pd.read_parquet(src)

    # quick sanity: show available columns once
    print("Events columns:", list(events.columns))

    print("Reconstructing possessions ...")
    possessions = build_possessions(events)

    dst.parent.mkdir(parents=True, exist_ok=True)
    possessions.to_parquet(dst, index=False)
    print(f"✅ saved {len(possessions):,} possessions → {dst}")


if __name__ == "__main__":
    main()
