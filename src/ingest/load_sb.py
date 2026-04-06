from pathlib import Path
import pandas as pd
import json

def load_single_json(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Flatten using dot notation, then we’ll rename
    df = pd.json_normalize(data, sep=".")
    df["match_id"] = int(json_path.stem) if json_path.stem.isdigit() else json_path.stem
    return df

def load_all_events(raw_dir="data/raw/events") -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {raw_dir}. Put match JSONs there.")
    dfs = [load_single_json(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def coalesce_outcome(df: pd.DataFrame) -> pd.Series:
    """
    Create a single 'outcome_name' by taking the first non-null among
    common outcome fields across event types.
    """
    # any column that ends with ".outcome.name" or ".outcome"
    candidates = [c for c in df.columns if c.endswith(".outcome.name")] + \
                 [c for c in df.columns if c.endswith(".outcome")]
    if not candidates:
        return pd.Series([None] * len(df), index=df.index)

    # Prefer the *.outcome.name variants; keep order stable
    # Build a DataFrame of candidates and take first non-null per row
    sub = df.reindex(columns=candidates)
    # If some outcome columns are dicts, convert to str
    for c in sub.columns:
        sub[c] = sub[c].apply(lambda v: v if (pd.isna(v) or isinstance(v, str)) else str(v))
    return sub.bfill(axis=1).iloc[:, 0]

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # always-present basics (or we create sensible defaults)
    out = pd.DataFrame({
        "match_id": df["match_id"],
        "team_name": df.get("team.name"),
        "player_name": df.get("player.name"),
        "type_name": df.get("type.name"),
        "timestamp": df.get("timestamp"),
        "minute": df.get("minute"),
        "second": df.get("second"),
        "possession_team_name": df.get("possession_team.name"),
        "period": df.get("period"),
        "duration": df.get("duration"),
        "under_pressure": df.get("under_pressure", False)
    })

    # location → x,y
    loc = df.get("location")
    out["x"] = loc.apply(lambda v: v[0] if isinstance(v, list) and len(v) > 0 else None) if loc is not None else None
    out["y"] = loc.apply(lambda v: v[1] if isinstance(v, list) and len(v) > 1 else None) if loc is not None else None

    # unified outcome
    out["outcome_name"] = coalesce_outcome(df)

    # ensure dtypes for time fields
    # StatsBomb timestamp format "00:12:34.567"
    out["timestamp"] = pd.to_timedelta(out["timestamp"])
    return out

def main():
    raw_dir = Path("data/raw/events")
    out_path = Path("data/processed/events.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {raw_dir} ...")
    df = load_all_events(raw_dir)
    df = clean_columns(df)

    df.to_parquet(out_path, index=False)
    print(f"✅ saved {len(df):,} events → {out_path}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
