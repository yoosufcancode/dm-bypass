"""Stage 3: Feature selection using multiple methods."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable
from sklearn.feature_selection import f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

EXCLUDE_COLS = {"match_id", "team_id", "team_name", "season", "player_id", "player_name"}

# ── Features to drop before selection ─────────────────────────────────────────
# These are team-level or role-driven confounds that are not player-inherent:
#   defensive_shape_compactness — team structural metric, not individual action
#   zone14_touches              — positional role artefact, not causal signal
CONFOUND_FEATURES = {"defensive_shape_compactness", "zone14_touches"}

# ── Column-type patterns for aggregation ──────────────────────────────────────
# Rate/positional/spatial features → mean across halves
# Count/action features → sum across halves
# Match-level constants (opponent context) → first value
RATE_PATTERNS   = ("rate", "position", "index", "_x", "_y", "variance",
                   "coverage", "compactness", "avg_", "average_")
MATCH_PATTERNS  = ("opp_", "score_diff")


def _classify_col(col: str) -> str:
    """Return 'match', 'rate', or 'count' aggregation type for a column."""
    if any(col.startswith(p) for p in MATCH_PATTERNS):
        return "match"
    if any(p in col for p in RATE_PATTERNS):
        return "rate"
    return "count"


def _aggregate_to_match_level(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Aggregate half-match rows (P1/P2) to full-match level per player, using
    semantic rules:
      - Match-level constants (opp_*, score_diff*)  → first()
      - Rate / positional / spatial features        → mean()
      - Count / action features                     → sum()
      - Target variable                             → sum()
    Groups by (player_id, player_name, team_name, base_match_id) where
    base_match_id strips the period suffix (e.g. '12345_P1' → '12345').
    """
    if "match_id" not in df.columns or "player_id" not in df.columns:
        return df

    df = df.copy()
    # Strip period suffix to combine P1 + P2 into one full-match row per player
    df["_base_match_id"] = df["match_id"].str.replace(r"_P\d+$", "", regex=True)

    group_by = [c for c in ("player_id", "player_name", "team_name", "_base_match_id")
                if c in df.columns]

    agg_spec: dict = {}
    for col in df.columns:
        if col in group_by or col in ("match_id", "_base_match_id"):
            continue
        if col == target_col:
            agg_spec[col] = "sum"
        elif col in EXCLUDE_COLS:
            agg_spec[col] = "first"
        else:
            kind = _classify_col(col)
            if kind == "match":
                agg_spec[col] = "first"
            elif kind == "rate":
                agg_spec[col] = "mean"
            else:
                agg_spec[col] = "sum"

    aggregated = df.groupby(group_by, as_index=False).agg(agg_spec)
    aggregated = aggregated.rename(columns={"_base_match_id": "match_id"})
    return aggregated


def rank_features(scores: np.ndarray, feature_names: list[str],
                  higher_is_better: bool = True) -> list[dict]:
    """Sort features by score and return ranked dicts."""
    pairs = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=higher_is_better)
    return [{"feature": f, "score": round(float(s), 6), "rank": i + 1}
            for i, (f, s) in enumerate(pairs)]


def run_feature_selection(
    features_path: str,
    target_col: str,
    n_top: int,
    progress_cb: Callable[[int, str], None],
) -> dict:
    """
    Run four selection methods (F-regression, MI, Random Forest, RFE) and return
    per-method rankings plus a consensus top-N list, persisting results to a sidecar JSON.

    Notebook-aligned steps applied before method runs:
      1. Aggregate half-match rows (P1/P2) to full-match level per player
      2. Drop team-level / role-driven confound features
      3. RFE uses RandomForestRegressor with n_features_to_select=n_top
    """
    progress_cb(5, "Loading dataset")
    df = pd.read_csv(features_path)

    # ── Step 1: Aggregate to match level ──────────────────────────────────────
    progress_cb(10, "Aggregating to match level")
    df = _aggregate_to_match_level(df, target_col)

    # ── Step 2: Drop confound features ────────────────────────────────────────
    drop_cols = CONFOUND_FEATURES & set(df.columns)
    if drop_cols:
        df = df.drop(columns=list(drop_cols))

    # ── Step 3: Build feature matrix ──────────────────────────────────────────
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_COLS and c != target_col
    ]

    # Drop features with >50% missing (notebook threshold)
    null_rates = df[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if null_rates[c] <= 0.5]

    df_clean = df[feature_cols + [target_col]].copy()
    df_clean[feature_cols] = df_clean[feature_cols].fillna(df_clean[feature_cols].median())
    df_clean = df_clean.dropna(subset=[target_col])

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    progress_cb(20, "Running univariate F-regression")
    f_scores, _ = f_regression(X_scaled, y)
    f_scores = np.nan_to_num(f_scores)
    univariate = rank_features(f_scores, feature_cols)

    progress_cb(38, "Running mutual information")
    mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
    mutual_info = rank_features(mi_scores, feature_cols)

    progress_cb(56, "Running Random Forest importance")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    random_forest = rank_features(rf.feature_importances_, feature_cols)

    # ── RFE with fixed n_features_to_select (matches notebook) ───────────────
    progress_cb(72, "Running RFE (RandomForest estimator, n_features_to_select=n_top)")
    rfe_estimator = RFE(
        estimator=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        n_features_to_select=n_top,
        step=1,
    )
    rfe_estimator.fit(X_scaled, y)
    rfe_selected = np.where(rfe_estimator.support_, 1.0, 0.0)
    rfe = rank_features(rfe_selected, feature_cols)

    progress_cb(90, "Computing consensus ranking")
    # Notebook consensus: selection_count×0.4 + combined_univariate×0.3 + rf_norm×0.3
    # combined_univariate = (F_Norm + MI_Norm + |Corr|) / 3  (matches notebook Combined_Score)
    # selection_count = how many of {Univariate top-N, RF top-N, RFE} selected the feature
    selected_univariate = {item["feature"] for item in univariate[:n_top]}
    selected_rf_set     = {item["feature"] for item in random_forest[:n_top]}
    selected_rfe_set    = {feature_cols[i] for i, s in enumerate(rfe_estimator.support_) if s}

    f_max  = f_scores.max() or 1.0
    mi_max = mi_scores.max() or 1.0
    rf_max = rf.feature_importances_.max() or 1.0

    correlations = np.array([
        abs(np.corrcoef(X_scaled[:, i], y)[0, 1]) if np.std(X_scaled[:, i]) > 0 else 0.0
        for i in range(len(feature_cols))
    ])
    corr_max = correlations.max() or 1.0

    f_norm_arr  = f_scores / f_max
    mi_norm_arr = mi_scores / mi_max
    corr_norm   = correlations / corr_max
    combined_univariate = (f_norm_arr + mi_norm_arr + corr_norm) / 3.0
    rf_norm_arr = rf.feature_importances_ / rf_max

    weighted: dict[str, float] = {}
    for i, feat in enumerate(feature_cols):
        count = sum([
            feat in selected_univariate,
            feat in selected_rf_set,
            feat in selected_rfe_set,
        ])
        weighted[feat] = (
            count * 0.4
            + combined_univariate[i] * 0.3
            + rf_norm_arr[i] * 0.3
        )

    consensus = sorted(
        [{"feature": f, "avg_rank": round(s, 4), "rank": i + 1}
         for i, (f, s) in enumerate(
             sorted(weighted.items(), key=lambda x: x[1], reverse=True)
         )],
        key=lambda x: x["rank"],
    )

    selected_features = [item["feature"] for item in consensus[:n_top]]

    sidecar = Path(features_path).parent / (
        Path(features_path).stem + "_selected_features.json"
    )
    sidecar.write_text(json.dumps({
        "selected_features": selected_features,
        "dropped_confounds": list(drop_cols),
        "n_match_rows": int(len(df_clean)),
    }))

    progress_cb(100, "Feature selection complete")

    return {
        "univariate":        univariate[:n_top],
        "mutual_info":       mutual_info[:n_top],
        "random_forest":     random_forest[:n_top],
        "rfe":               [r for r in rfe if r["score"] > 0.5][:n_top],
        "consensus":         consensus[:n_top],
        "selected_features": selected_features,
        "dropped_confounds": list(drop_cols),
        "n_match_rows":      int(len(df_clean)),
    }
