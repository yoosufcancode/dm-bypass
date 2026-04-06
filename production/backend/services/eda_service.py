"""Stage 2: Exploratory Data Analysis."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable

EXCLUDE_COLS = {"match_id", "team_id", "team_name", "season", "player_id", "player_name"}


def run_eda(features_path: str, progress_cb: Callable[[int, str], None]) -> dict:
    progress_cb(5, "Loading dataset")
    df = pd.read_csv(features_path)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_COLS]
    df_num = df[numeric_cols]

    progress_cb(20, "Computing descriptive statistics")
    desc = df_num.describe().round(4)
    descriptive_stats = desc.to_dict()

    progress_cb(40, "Computing correlation matrix")
    corr = df_num.corr().round(4)
    correlation_matrix = corr.to_dict()

    progress_cb(60, "Counting missing values")
    missing = df.isnull().sum()
    missing_values = {k: int(v) for k, v in missing.items() if v > 0}

    progress_cb(75, "Computing bypass distribution")
    target = "bypasses_per_halftime"
    if target in df.columns:
        vals = df[target].dropna().values
        counts, bin_edges = np.histogram(vals, bins=20)
        bypass_distribution = {
            "counts": counts.tolist(),
            "bin_edges": [round(float(e), 4) for e in bin_edges],
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
        }
    else:
        bypass_distribution = {}

    progress_cb(100, "EDA complete")

    return {
        "descriptive_stats": descriptive_stats,
        "correlation_matrix": correlation_matrix,
        "missing_values": missing_values,
        "bypass_distribution": bypass_distribution,
        "row_count": len(df),
        "column_count": len(df.columns),
    }
