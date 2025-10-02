from pathlib import Path
import pandas as pd
import yaml

def load_cfg(path="config/labels.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def label_bypass(events: pd.DataFrame, possessions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Label each opponent possession with bypass=1 if final third reached within window."""
    L = cfg["bypass"]["time_seconds"]; P = cfg["bypass"]["max_passes"]
    final_x = cfg["pitch"]["final_third_x"]

    ev = events.sort_values(["match_id","period","timestamp"]).copy()
    ev["is_pass"] = (ev["type"] == "Pass").astype(int)

    labels = []
    for (mid, pid, team_id), p in possessions.groupby(["match_id","poss_id","team_id"]):
        # events within this possession
        seg = ev[(ev.match_id==mid) & (ev.possession_team==team_id)].copy()
        if seg.empty:
            labels.append({"match_id": mid, "poss_id": pid, "team_id": team_id, "bypass": 0}); continue
        t0 = seg["timestamp"].min()
        window = seg[(seg["timestamp"] <= t0 + pd.Timedelta(seconds=L))]
        passes = window["is_pass"].cumsum()
        reached = window[(window["x"] >= final_x) & (passes <= P)]
        labels.append({"match_id": mid, "poss_id": pid, "team_id": team_id, "bypass": int(len(reached)>0)})
    return pd.DataFrame(labels)

if __name__ == "__main__":
    cfg = load_cfg()
    ev = pd.read_parquet("data/processed/events.parquet")
    poss = pd.read_parquet("data/processed/possessions.parquet")
    y = label_bypass(ev, poss, cfg)
    y.to_parquet("data/processed/labels.parquet", index=False)
    print("labels saved.")
