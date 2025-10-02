from pathlib import Path
import pandas as pd

def build_possessions(events: pd.DataFrame) -> pd.DataFrame:
    """Return a table of opponent possessions with start/end timestamps and team ids."""
    ev = events.sort_values(["match_id","period","timestamp"]).copy()
    # possession id: increment when possession_team changes or a restart happens
    ev["poss_id"] = ( (ev["possession_team"].shift(1) != ev["possession_team"]) |
                      (ev["type"].isin(["Kick Off","Throw-in","Free Kick","Goal Kick","Corner","Foul Won"])) ).cumsum()
    grp = ev.groupby(["match_id","poss_id","possession_team"], as_index=False).agg(
        start_time = ("timestamp","min"),
        end_time   = ("timestamp","max"),
        start_x    = ("x","first"),
        start_y    = ("y","first"),
        end_x      = ("x","last"),
        end_y      = ("y","last"),
        n_events   = ("id","count")
    )
    return grp.rename(columns={"possession_team":"team_id"})

if __name__ == "__main__":
    # example usage: read processed events parquet and write possessions parquet
    src = Path("data/processed/events.parquet")
    dst = Path("data/processed/possessions.parquet")
    df = pd.read_parquet(src)
    poss = build_possessions(df)
    poss.to_parquet(dst, index=False)
    print(f"saved {len(poss)} possessions -> {dst}")
