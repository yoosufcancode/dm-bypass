import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

events_cols = ["match_id","period","timestamp","possession_team","type","x","y","id"]
poss_cols   = ["match_id","poss_id","team_id","start_time","end_time","start_x","start_y","end_x","end_y","n_events"]

pd.DataFrame(columns=events_cols).to_parquet("data/processed/events.parquet", index=False)
pd.DataFrame(columns=poss_cols).to_parquet("data/processed/possessions.parquet", index=False)

print("Created empty parquet files in data/processed")
