"""
End-to-end pipeline: feature engineering → feature selection → model building → evaluation.

Usage:
    python scripts/run_pipeline.py --league Spain --skip-features --team Barcelona
    python scripts/run_pipeline.py --league Spain --skip-features --team Barcelona --half-match
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(ROOT))

from scripts.model import (
    load_data, split_data, build_mlr, build_ridge, build_lasso,
    evaluate_loocv, evaluate_model_on_test_set, save_model
)

# ── Constants ────────────────────────────────────────────────────────────────

META_COLS = {
    "player_id", "player_name", "match_id", "team_id", "team_name",
    "league", "computed_at", "midfielder_type",
    "opp_long_ball_rate", "opp_avg_pass_length", "opp_direct_play_index",
    "opp_pass_forward_rate", "score_diff_start_of_half",
}
TARGET_HALF = "bypasses_per_halftime"
TARGET_FULL = "bypasses_per_match"

# Features that should be summed (counts) vs averaged (rates/ratios) when
# aggregating two halves into a full match.
COUNT_FEATURES = {
    "passes_attempted", "carries_attempted", "possessions_involved", "turnovers",
    "ball_recoveries", "interceptions", "tackles_won", "clearance_followed_by_recovery",
    "fouls_committed", "tactical_fouls", "set_piece_involvements", "set_piece_duels_won",
    "defensive_set_piece_clearances", "zone14_touches", "zone_entries",
    "aerial_duels_contested", "sliding_tackles", "third_man_runs", "wall_pass_events",
    "secondary_shot_assists", "shot_creating_actions", "progressive_passes",
    "final_third_entries_by_pass", "key_passes", "successful_dribbles",
    "carries_leading_to_shot", "carries_leading_to_key_pass", "final_third_carries",
    "penalty_area_carries", "carry_distance_total", "progressive_carries",
    "defensive_midfield_actions", "bypass_channel_defensive_actions",
    "penalty_area_deliveries", "switches_completed",
}


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_to_full_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse half-match rows (P1 + P2) into one row per player per match.

    Count features are summed; rate/ratio features are averaged.
    The target becomes bypasses_per_match = sum of both halves.
    """
    df = df.copy()
    df["base_match_id"] = df["match_id"].str.replace(r"_P[12]$", "", regex=True)

    feat_cols = [c for c in df.columns
                 if c not in META_COLS and c != TARGET_HALF and c != "base_match_id"]

    agg = {c: "sum" if c in COUNT_FEATURES else "mean" for c in feat_cols if c in df.columns}
    agg[TARGET_HALF] = "sum"

    df_full = (
        df.groupby(["player_id", "base_match_id", "team_id", "team_name"])
        .agg(agg)
        .reset_index()
        .rename(columns={"base_match_id": "match_id", TARGET_HALF: TARGET_FULL})
    )
    return df_full


# ── Pre-processing ────────────────────────────────────────────────────────────

def prepare(df: pd.DataFrame, target: str, min_passes: int = 5) -> tuple:
    """
    Filter, drop bad columns, impute, scale, and split.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split

    if "passes_attempted" in df.columns:
        before = len(df)
        df = df[df["passes_attempted"] >= min_passes].copy()
        print(f"Dropped {before - len(df)} rows with < {min_passes} passes → {len(df):,} remain")

    feat_cols = [
        c for c in df.columns
        if c not in META_COLS and c not in {TARGET_HALF, TARGET_FULL, "base_match_id"}
    ]

    non_const = df[feat_cols].nunique() > 1
    feat_cols = [c for c in feat_cols if non_const[c]]

    null_rate = df[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if null_rate[c] <= 0.5]

    print(f"Using {len(feat_cols)} features after filtering")

    X = df[feat_cols].copy()
    y = df[target].reset_index(drop=True)

    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X), columns=feat_cols
    )
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X_imp), columns=feat_cols
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}\n")

    return X_train, X_test, y_train, y_test, feat_cols


# ── Main ──────────────────────────────────────────────────────────────────────

def main(league: str = "Spain", skip_features: bool = False,
         team: str = None, half_match: bool = False):

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    csv_path = ROOT / "data" / "processed" / f"wyscout_{league}_features.csv"

    # ── 1. Feature engineering ────────────────────────────────────────────────
    if not skip_features:
        print("=" * 60)
        print(f"Step 1: Feature Engineering ({league})")
        print("=" * 60)
        from src.features.main_feature import compute_features_for_competition
        compute_features_for_competition(league=league)
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"Features CSV not found: {csv_path}")
        print(f"Skipping feature engineering — using {csv_path.name}")

    # ── 2. Prepare data ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Data Preparation")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path.name}")

    if team:
        before = len(df)
        df = df[df["team_name"] == team]
        if df.empty:
            available = sorted(pd.read_csv(csv_path)["team_name"].unique())
            raise ValueError(
                f"Team '{team}' not found. Available teams:\n  " + "\n  ".join(available)
            )
        print(f"Filtered to {team}: {len(df):,} rows (from {before:,})")

    granularity = "half-match"
    target = TARGET_HALF

    if not half_match:
        df = aggregate_to_full_match(df)
        target = TARGET_FULL
        granularity = "full-match"
        print(f"Aggregated to full-match: {len(df):,} rows")

    print(f"Granularity: {granularity}  |  Target: {target}")

    X_train, X_test, y_train, y_test, feat_cols = prepare(df, target)

    # ── 3. Build models ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 3: Model Building")
    print("=" * 60)

    mlr   = build_mlr(X_train, y_train)
    ridge = build_ridge(X_train, y_train)
    lasso = build_lasso(X_train, y_train)

    # ── 4. Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Evaluation (LOOCV + Test Set)")
    print("=" * 60)

    results = {}
    for name, model in [("MLR", mlr), ("Ridge", ridge), ("Lasso", lasso)]:
        loocv = evaluate_loocv(model, X_train, y_train, model_name=name)
        test  = evaluate_model_on_test_set(model, X_test, y_test, model_name=name)
        results[name] = {"loocv": loocv, "test": test}

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":            name,
            "Test Spearman ρ":  f"{r['test']['Spearman']:.4f}",
            "Test p-value":     f"{r['test']['Spearman_p']:.4f}",
            "LOOCV Spearman ρ": f"{r['loocv']['Spearman']:.4f}",
            "Test R²":          f"{r['test']['R2']:.4f}",
            "Test RMSE":        f"{r['test']['RMSE']:.4f}",
            "Test MAE":         f"{r['test']['MAE']:.4f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 6. Best model feature importance ─────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["test"]["Spearman"])
    best_model = {"MLR": mlr, "Ridge": ridge, "Lasso": lasso}[best_name]
    print(f"\nBest model: {best_name}")

    coefs = pd.Series(best_model.coef_, index=feat_cols)
    top = coefs.abs().nlargest(15)
    print("\nTop 15 features by absolute coefficient:")
    for feat, val in coefs[top.index].items():
        print(f"  {feat:45s}  {val:+.4f}")

    # ── 7. Save models ────────────────────────────────────────────────────────
    parts = [league]
    if team:
        parts.append(team.replace(" ", "_"))
    parts.append(granularity)
    suffix = "_".join(parts)

    save_model(mlr,   models_dir / f"wyscout_{suffix}_mlr_model.pkl")
    save_model(ridge, models_dir / f"wyscout_{suffix}_ridge_model.pkl")
    save_model(lasso, models_dir / f"wyscout_{suffix}_lasso_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="Spain")
    parser.add_argument("--team", default=None,
                        help="Filter to a single team (e.g. 'Barcelona')")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering, use existing CSV")
    parser.add_argument("--half-match", action="store_true",
                        help="Train on half-match granularity instead of full match (default: full)")
    args = parser.parse_args()
    main(league=args.league, skip_features=args.skip_features,
         team=args.team, half_match=args.half_match)
