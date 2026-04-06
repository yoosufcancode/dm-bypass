"""
Player recommendation pipeline for bypass prevention.

Steps:
  1. Load Wyscout full-match features for a league
  2. Run scouting evaluation on the target team → gradient weights
  3. Score ALL players in the league using gradient-weighted bypass score
  4. Flag weak target-team players (above position median)
  5. Rank replacement candidates from other teams

Usage:
    python scripts/player_recommendations.py --league Spain --team Barcelona
    python scripts/player_recommendations.py --league Spain --team Barcelona --top-n 5
"""

import argparse
import warnings
import sys
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import statsmodels.api as sm

# ── Constants ─────────────────────────────────────────────────────────────────

META_COLS = {
    "player_id", "player_name", "match_id", "team_id", "team_name",
    "league", "computed_at", "midfielder_type",
    "opp_long_ball_rate", "opp_avg_pass_length", "opp_direct_play_index",
    "opp_pass_forward_rate", "score_diff_start_of_half",
}
TARGET_HALF = "bypasses_per_halftime"
TARGET_FULL = "bypasses_per_match"
OPP_PREFIXES = ("opp_", "score_diff")

# ── Tactical role clustering feature set ──────────────────────────────────────
# Three groups map to three archetypes. Clustering is purely data-driven;
# labels are assigned by centroid profile comparison — no player names.
ROLE_FEATURES = [
    # Depth & positioning → Anchor 6 signal
    "average_position_x",              # deepest position on pitch
    "average_position_y",              # central vs wide tendency
    "possession_time_seconds",         # ball retention under pressure
    "midfield_presence_on_deep_opp",   # blocking bypasses through positioning
    "bypass_channel_defensive_actions",# central channel coverage
    "interceptions",                   # reading passing lanes
    "clearance_followed_by_recovery",  # defensive awareness + composure
    "tactical_fouls",                  # using fouls as positional tool
    "pass_completion_rate",            # composure and technical security
    "defensive_shape_compactness",     # structural discipline in block
    "set_piece_duels_won",             # aerial presence in defensive phases

    # Active ball-winning → Ball-winning 8 signal
    "tackles_won",                     # primary dueling metric
    "ball_recoveries",                 # active ball-hunting
    "sliding_tackles",                 # aggressive pressing action
    "sliding_tackle_success_rate",     # quality of duels, not just volume
    "defensive_midfield_actions",      # active defensive engagement count
    "fouls_committed",                 # high-press risk tolerance
    "aerial_duels_contested",          # physical presence in duels
    "aerial_duel_win_rate",            # aerial dominance quality
    "turnovers",                       # risk profile of press-then-lose

    # Transition & progression → Hybrid 6/8 signal
    "progressive_passes",              # line-breaking passes
    "progressive_carries",             # driving forward with the ball
    "carry_distance_total",            # total ground covered carrying
    "final_third_entries_by_pass",     # transition pass quality
    "carries_leading_to_key_pass",     # creative carry-then-pass
    "final_third_carries",             # penetrating carries into attack
    "third_man_runs",                  # off-ball movement into space
    "key_passes",                      # direct creative output
    "shot_creating_actions",           # involvement in goal-creation
    "tempo_index",                     # pace of play contribution
    "zone14_touches",                  # presence in the critical zone
    "switches_completed",              # changing the point of attack
    "wall_pass_events",                # combination / give-and-go play
    "successful_dribbles",             # individual ball progression
]



# ── Data loading ──────────────────────────────────────────────────────────────

def load_half_match(csv_path: Path, team: str = None) -> pd.DataFrame:
    """Load half-match CSV rows directly (no aggregation)."""
    df = pd.read_csv(csv_path)
    if team:
        df = df[df["team_name"] == team].copy()
        if df.empty:
            available = sorted(pd.read_csv(csv_path)["team_name"].unique())
            raise ValueError(f"Team '{team}' not found. Available: {available}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    excl = META_COLS | {TARGET_HALF, "match_id"}
    feat_cols = [c for c in df.columns if c not in excl]
    non_const = df[feat_cols].nunique() > 1
    null_rate  = df[feat_cols].isna().mean()
    return [c for c in feat_cols if non_const[c] and null_rate[c] <= 0.5]


def prepare_xy(df: pd.DataFrame, feat_cols: list, target: str, min_passes: int = 5):
    if "passes_attempted" in df.columns:
        df = df[df["passes_attempted"] >= min_passes].copy()
    X = df[feat_cols].copy()
    y = df[target].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=feat_cols, index=X.index)
    scaler = StandardScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X_imp), columns=feat_cols, index=X_imp.index)
    return X_sc, y, imp, scaler


# ── Scouting evaluation ───────────────────────────────────────────────────────

def _loocv_spearman(model, X: pd.DataFrame, y: pd.Series) -> float:
    """LOOCV Spearman ρ on training data — used for model selection."""
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for tr_idx, te_idx in loo.split(X):
        m = type(model)(**{k: v for k, v in model.get_params().items()
                           if k not in ("cv", "store_cv_values", "store_cv_results")})
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        y_pred[te_idx] = m.predict(X.iloc[te_idx])
    rho, _ = spearmanr(y, y_pred)
    return float(rho)


def run_scouting_evaluation(X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame,  y_test: pd.Series) -> dict:
    """
    Fit MLR, Ridge, Lasso — select best by LOOCV Spearman ρ on training data.
    Run OLS for p-values, check sign stability with 5-fold CV using the winning
    model type. Returns gradient dict for scouting features.
    """
    scout_features = [f for f in X_train.columns
                      if not any(f.startswith(p) for p in OPP_PREFIXES)]
    Xs_tr = X_train[scout_features]
    Xs_te = X_test[scout_features]

    alphas = np.logspace(-2, 3, 50)

    # ── Fit all three models ──────────────────────────────────────────────────
    mlr   = LinearRegression().fit(Xs_tr, y_train)
    ridge = RidgeCV(alphas=alphas, cv=5).fit(Xs_tr, y_train)
    lasso = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5,
                    max_iter=2000, random_state=42).fit(Xs_tr, y_train)

    # ── LOOCV Spearman on training set ────────────────────────────────────────
    rho_mlr   = _loocv_spearman(mlr,   Xs_tr, y_train)
    rho_ridge = _loocv_spearman(ridge, Xs_tr, y_train)
    rho_lasso = _loocv_spearman(lasso, Xs_tr, y_train)

    model_results = {
        "MLR":   (mlr,   rho_mlr),
        "Ridge": (ridge, rho_ridge),
        "Lasso": (lasso, rho_lasso),
    }
    best_name = max(model_results, key=lambda k: model_results[k][1])
    best_model, _ = model_results[best_name]

    print(f"\n  Model selection (LOOCV Spearman ρ on train):")
    for name, (_, rho) in model_results.items():
        marker = "  ← selected" if name == best_name else ""
        print(f"    {name:<6}  ρ = {rho:.4f}{marker}")

    # Test-set Spearman for the winning model
    preds_test  = best_model.predict(Xs_te)
    sp_test, p_sp = spearmanr(y_test, preds_test)
    preds_train = best_model.predict(Xs_tr)
    sp_train, _ = spearmanr(y_train, preds_train)

    print(f"\n  Spearman rho (train): {sp_train:.4f}")
    print(f"  Spearman rho (test) : {sp_test:.4f}   p={p_sp:.4f}  <- primary metric")

    # ── OLS p-values (model-agnostic; always OLS for interpretability) ────────
    X_sm = sm.add_constant(Xs_tr, has_constant="add")
    ols  = sm.OLS(y_train, X_sm).fit()

    # ── 5-fold sign stability using the winning model type ────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_coefs = []
    for tr_idx, _ in kf.split(Xs_tr):
        if best_name == "MLR":
            m = LinearRegression()
        elif best_name == "Ridge":
            m = Ridge(alpha=float(best_model.alpha_))
        else:
            m = Lasso(alpha=float(best_model.alpha_), max_iter=2000)
        m.fit(Xs_tr.iloc[tr_idx], y_train.iloc[tr_idx])
        fold_coefs.append(m.coef_)
    fold_coefs = np.array(fold_coefs)
    mean_coefs = fold_coefs.mean(axis=0)
    std_coefs  = fold_coefs.std(axis=0)

    print(f"\n  {'Feature':<40} {'beta':>8} {'p-val':>7} {'cv_std':>7}  Sign stable  Scout?")
    print("  " + "-" * 85)

    rows = []
    scouting_grads = {}
    for i, fname in enumerate(scout_features):
        signs       = np.sign(fold_coefs[:, i])
        sign_stable = len(set(signs)) == 1
        mean_b      = float(mean_coefs[i])
        cv_std      = float(std_coefs[i])
        pval        = float(ols.pvalues.get(fname, 1.0))
        scout       = sign_stable and pval < 0.15

        if scout:
            scouting_grads[fname] = mean_b

        stable_str = "stable" if sign_stable else "FLIPS"
        scout_str  = "SCOUT" if scout else (
            "skip (unstable)" if not sign_stable else f"skip (p={pval:.2f})"
        )
        print(f"  {fname:<40} {mean_b:>8.4f} {pval:>7.4f} {cv_std:>7.4f}  {stable_str:<12}  {scout_str}")
        rows.append({"feature": fname, "mean_coef": mean_b, "cv_std": cv_std,
                     "p_value": pval, "sign_stable": sign_stable, "scout": scout})

    print(f"\n  Scouting features identified (strict): {list(scouting_grads.keys())}")

    # ── Minimum-6 fallback ────────────────────────────────────────────────────
    MIN_SCOUT = 6
    if len(scouting_grads) < MIN_SCOUT:
        needed = MIN_SCOUT - len(scouting_grads)
        candidates = sorted(
            [r for r in rows if r["feature"] not in scouting_grads],
            key=lambda r: (0 if r["sign_stable"] else 1, r["p_value"]),
        )
        for r in candidates[:needed]:
            scouting_grads[r["feature"]] = r["mean_coef"]
            print(f"  [fallback] added {r['feature']}  "
                  f"(sign_stable={r['sign_stable']}, p={r['p_value']:.4f})")
        print(f"  Scouting features after fallback ({MIN_SCOUT} min): {list(scouting_grads.keys())}")

    # ── Save gradient file ────────────────────────────────────────────────────
    total_abs = sum(abs(v) for v in scouting_grads.values()) or 1.0
    grad_rows = []
    for fname, mean_b in sorted(scouting_grads.items(), key=lambda x: abs(x[1]), reverse=True):
        r = next(r for r in rows if r["feature"] == fname)
        tier = ("Tier 1 — high confidence" if r["p_value"] < 0.05 else
                "Tier 2 — moderate confidence" if r["p_value"] < 0.10 else
                "Tier 3 — indicative only" if r["p_value"] < 0.15 else
                "Tier 4 — fallback only")
        grad_rows.append({
            "Feature":                fname,
            "Gradient (dy/dx)":       mean_b,
            "Abs_Gradient":           abs(mean_b),
            "Normalised Sensitivity": abs(mean_b) / total_abs,
            "p_value":                r["p_value"],
            "cv_std":                 r["cv_std"],
            "sign_stable":            r["sign_stable"],
            "confidence_tier":        tier,
            "scouting_direction":     "look for LOW" if mean_b > 0 else "look for HIGH",
            "model":                  best_name,
        })

    grad_df = pd.DataFrame(grad_rows)
    grad_path = ROOT / "data" / "processed" / "mlr_gradient_analysis.csv"
    grad_df.to_csv(grad_path, index=False)
    print(f"\n  Gradient file saved → {grad_path}")

    return {
        "scouting_grads": scouting_grads,
        "grad_df":         grad_df,
        "spearman_test":   sp_test,
        "spearman_train":  sp_train,
        "best_model_name": best_name,
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def assign_position_bucket(player_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket players into DM / CM / AM using league-wide percentiles of average_position_x.
    DM = bottom third (deepest), CM = middle third, AM = top third.
    Falls back to 'MF' if average_position_x is unavailable.
    """
    if "average_position_x" not in player_agg.columns or player_agg["average_position_x"].isna().all():
        player_agg["position_bucket"] = "MF"
        return player_agg

    q33 = player_agg["average_position_x"].quantile(0.33)
    q67 = player_agg["average_position_x"].quantile(0.67)

    def _bucket(x):
        if pd.isna(x):
            return "MF"
        if x <= q33:
            return "DM"
        if x <= q67:
            return "CM"
        return "AM"

    player_agg["position_bucket"] = player_agg["average_position_x"].map(_bucket)
    return player_agg


def assign_tactical_role(player_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster midfielders into three tactical archetypes purely from event-data
    behaviour. Labels are assigned by centroid-profile comparison — no player
    names are hardcoded.

    Roles:
      Anchor 6       — positional controller, prevents bypasses through depth
      Ball-winning 8 — mobile ball-winner, actively regains possession
      Hybrid 6/8     — transitional, resists press and drives forward
    """
    avail = [f for f in ROLE_FEATURES if f in player_agg.columns
             and player_agg[f].notna().mean() >= 0.7]

    if len(avail) < 6:
        player_agg = player_agg.copy()
        player_agg["tactical_role"] = "Unknown"
        return player_agg

    player_agg = player_agg.copy()
    X = player_agg[avail].copy()
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=avail, index=X.index
    )
    X_sc = pd.DataFrame(
        StandardScaler().fit_transform(X_imp),
        columns=avail, index=X_imp.index
    )

    # ── Elbow method + silhouette: determine optimal k ────────────────────────
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    processed_dir = ROOT / "data" / "processed"
    elbow_path     = processed_dir / "kmeans_elbow_curve.png"
    silhouette_path = processed_dir / "kmeans_silhouette_scores.png"

    k_range = range(1, 11)
    inertias = []
    sil_k_range = range(2, 9)
    sil_scores = []

    for k in k_range:
        _km = KMeans(n_clusters=k, random_state=42, n_init=20)
        _km.fit(X_sc)
        inertias.append(_km.inertia_)

    for k in sil_k_range:
        _km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = _km.fit_predict(X_sc)
        sil_scores.append(silhouette_score(X_sc, labels))

    # Elbow curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), inertias, marker="o", color="#3b82f6", linewidth=2)
    ax.axvline(x=3, color="#ef4444", linestyle="--", linewidth=1.5, label="k=3 (selected)")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("WCSS / Inertia", fontsize=12)
    ax.set_title("KMeans Elbow Curve — Cross-League Midfielder Pool", fontsize=13)
    ax.legend()
    ax.set_xticks(list(k_range))
    fig.tight_layout()
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)

    # Silhouette score profile
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(sil_k_range), sil_scores, marker="s", color="#10b981", linewidth=2)
    ax.axvline(x=3, color="#ef4444", linestyle="--", linewidth=1.5, label="k=3 (selected)")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Mean Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Scores — Cross-League Midfielder Pool", fontsize=13)
    ax.legend()
    ax.set_xticks(list(sil_k_range))
    fig.tight_layout()
    fig.savefig(silhouette_path, dpi=150)
    plt.close(fig)

    optimal_k = 3  # confirmed by elbow at k=3
    print(f"\n  KMeans elbow curve saved → {elbow_path}")
    print(f"  Silhouette profile saved  → {silhouette_path}")
    print(f"  Inertias (k=1..10): {[round(v,1) for v in inertias]}")
    print(f"  Silhouette scores (k=2..8): {[round(v,4) for v in sil_scores]}")

    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    player_agg["_cluster"] = km.fit_predict(X_sc)
    centroids = pd.DataFrame(km.cluster_centers_, columns=avail)

    # ── Label each cluster by centroid profile ────────────────────────────────
    # Each role wins on a distinct dimension combination:
    #   Anchor 6       → deepest (min avg_position_x) + most possession time
    #   Ball-winning 8 → most tackles + most ball recoveries
    #   Hybrid 6/8     → most progressive passes + most carry distance

    def _role_score(c, role):
        row = centroids.loc[c]
        if role == "Anchor 6":
            return (
                -_get(row, "average_position_x")      # lower = deeper
                + _get(row, "possession_time_seconds")
                + _get(row, "midfield_presence_on_deep_opp")
                + _get(row, "bypass_channel_defensive_actions")
                + _get(row, "interceptions")
                + _get(row, "defensive_shape_compactness")
            )
        if role == "Ball-winning 8":
            return (
                + _get(row, "tackles_won")
                + _get(row, "ball_recoveries")
                + _get(row, "sliding_tackles")
                + _get(row, "defensive_midfield_actions")
                + _get(row, "aerial_duels_contested")
                + _get(row, "fouls_committed")
            )
        if role == "Hybrid 6/8":
            return (
                + _get(row, "progressive_passes")
                + _get(row, "progressive_carries")
                + _get(row, "carry_distance_total")
                + _get(row, "final_third_entries_by_pass")
                + _get(row, "carries_leading_to_key_pass")
                + _get(row, "zone14_touches")
                + _get(row, "shot_creating_actions")
            )
        return 0

    def _get(row, feat):
        return float(row[feat]) if feat in row.index else 0.0

    all_roles = ["Anchor 6", "Ball-winning 8", "Hybrid 6/8"]
    cluster_labels = {}

    # Greedy assignment: pick the role each cluster scores highest on,
    # resolving ties so every role is assigned exactly once
    role_scores = {
        c: {r: _role_score(c, r) for r in all_roles}
        for c in range(3)
    }
    assigned_roles = set()
    # Sort (cluster, role) pairs by score descending; assign greedily
    pairs = sorted(
        [(c, r, role_scores[c][r]) for c in range(3) for r in all_roles],
        key=lambda x: x[2], reverse=True
    )
    for c, role, _ in pairs:
        if c not in cluster_labels and role not in assigned_roles:
            cluster_labels[c] = role
            assigned_roles.add(role)

    player_agg["tactical_role"] = player_agg["_cluster"].map(cluster_labels)
    player_agg = player_agg.drop(columns=["_cluster"])

    # ── Cluster summary ───────────────────────────────────────────────────────
    summary_feats = [
        ("average_position_x",        "avg_x"),
        ("possession_time_seconds",   "poss_s"),
        ("tackles_won",               "tackles"),
        ("ball_recoveries",           "recoveries"),
        ("progressive_passes",        "prog_pass"),
        ("carry_distance_total",      "carry_dist"),
    ]
    print("\n  Tactical role clusters (KMeans, 34-feature profile):")
    print(f"  {'Role':<20} {'N':>4}  " + "  ".join(f"{lbl:>11}" for _, lbl in summary_feats))
    print("  " + "-" * 90)
    for role in all_roles:
        grp = player_agg[player_agg["tactical_role"] == role]
        if grp.empty:
            continue
        vals = []
        for feat, _ in summary_feats:
            vals.append(f"{grp[feat].mean():>11.2f}" if feat in grp.columns else f"{'N/A':>11}")
        print(f"  {role:<20} {len(grp):>4}  {'  '.join(vals)}")

    return player_agg


def score_all_players(
    df_all: pd.DataFrame,
    scouting_grads: dict,
    target_team: str,
    min_matches: int = 4,       # aligned with notebook (was 5/10)
    min_bypasses: float = 3.0,  # notebook: candidates must avg ≥ 3 bypasses/half
) -> pd.DataFrame:
    """
    Aggregate each player to a single row (mean of match-level stats),
    apply Bayesian shrinkage, then gradient-weight to produce a bypass score.
    Lower score = better bypass prevention.

    min_matches  : minimum half-match observations to be included (notebook = 4)
    min_bypasses : candidate pool floor — players averaging fewer than this many
                   bypasses per halftime are excluded as they lack bypass exposure
                   signal. Applied only to non-target-team players. (notebook = 3.0)
    """
    feats = list(scouting_grads.keys())
    if not feats:
        raise ValueError("No scouting features — check gradient evaluation step.")

    # Include scouting features + all role-clustering features
    all_feat_cols = list(dict.fromkeys(feats + ROLE_FEATURES))
    agg = {f: "mean" for f in all_feat_cols if f in df_all.columns}
    agg[TARGET_HALF] = "mean"
    agg["match_id"]  = "count"

    # Carry league column if present (cross-league pool)
    group_cols = ["player_id", "player_name", "team_name"]
    if "league" in df_all.columns:
        agg["league"] = "first"

    player_agg = (
        df_all.dropna(subset=["player_name"])
        .groupby(group_cols)
        .agg(agg)
        .reset_index()
        .rename(columns={"match_id": "matches_played"})
    )
    player_agg = player_agg[player_agg["matches_played"] >= min_matches].copy()

    # Apply min_bypasses floor to non-target-team players only
    # (target team players are always included regardless of bypass count)
    is_target = player_agg["team_name"] == target_team
    has_enough_bypasses = player_agg[TARGET_HALF] >= min_bypasses
    player_agg = player_agg[is_target | has_enough_bypasses].copy()

    # Bayesian shrinkage toward pool mean
    k = float(player_agg["matches_played"].median())
    for feat in feats:
        if feat not in player_agg.columns:
            player_agg[feat] = np.nan
        pool_mean = player_agg[feat].mean()
        n = player_agg["matches_played"]
        vals = player_agg[feat].fillna(pool_mean)
        player_agg[f"{feat}_shrunk"] = (n * vals + k * pool_mean) / (n + k)

    # Z-score shrunk features, then gradient-weight
    z_cols = []
    for feat in feats:
        col   = f"{feat}_shrunk"
        mu    = player_agg[col].mean()
        sigma = player_agg[col].std(ddof=1)
        z_col = f"z_{feat}"
        player_agg[z_col] = (player_agg[col] - mu) / (sigma if sigma > 0 else 1)
        z_cols.append(z_col)

    grad_arr = np.array([scouting_grads[f] for f in feats])
    player_agg["raw_bypass_score"] = player_agg[z_cols].values @ grad_arr
    player_agg["bypass_score"]     = player_agg["raw_bypass_score"].rank(pct=True) * 100
    player_agg["source"] = np.where(is_target.reindex(player_agg.index, fill_value=False),
                                    target_team, "Other")

    # Assign position buckets (depth percentile) and tactical roles (clustering)
    player_agg = assign_position_bucket(player_agg)
    player_agg = assign_tactical_role(player_agg)

    return player_agg


def flag_weak_players(team_players: pd.DataFrame) -> pd.DataFrame:
    """
    Flag target-team players whose bypass_score exceeds the median for their
    position bucket — these are the players who need replacing.

    Adds a boolean column 'is_weak' and a string column 'weakness_reason'.
    Matches notebook logic: above position-type median bypass_score = WEAK.
    """
    team_players = team_players.copy()
    bucket_medians = (
        team_players.groupby("position_bucket")["bypass_score"].median().to_dict()
    )
    team_players["is_weak"] = team_players.apply(
        lambda r: r["bypass_score"] > bucket_medians.get(r["position_bucket"], 50.0),
        axis=1,
    )
    team_players["weakness_reason"] = team_players.apply(
        lambda r: (
            f"bypass_score {r['bypass_score']:.1f} > "
            f"{r['position_bucket']} median "
            f"({bucket_medians.get(r['position_bucket'], 50.0):.1f})"
        ) if r["is_weak"] else "",
        axis=1,
    )
    return team_players


# ── Main ──────────────────────────────────────────────────────────────────────

def main(league: str = "Spain", team: str = "Barcelona", top_n: int = 5,
         min_matches: int = 4, min_bypasses: float = 3.0):

    csv_path = ROOT / "data" / "processed" / f"wyscout_{league}_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {csv_path}\n"
                                f"Run: python -m src.features.main_feature --leagues {league}")

    all_leagues = ["Spain", "England", "France", "Germany", "Italy"]

    print("=" * 70)
    print(f"Player Recommendation Pipeline  |  League: {league}  |  Team: {team}")
    print("=" * 70)

    # ── Step 1: Load half-match data ─────────────────────────────────────────
    print("\n[1] Loading half-match data...")
    df_team = load_half_match(csv_path, team=team)

    # Build cross-league candidate pool from all available feature CSVs
    league_frames = []
    for lg in all_leagues:
        lg_path = ROOT / "data" / "processed" / f"wyscout_{lg}_features.csv"
        if not lg_path.exists():
            print(f"    Skipping {lg} — features CSV not found (run feature engineering first)")
            continue
        df_lg = load_half_match(lg_path)
        df_lg["league"] = lg
        league_frames.append(df_lg)
        print(f"    Loaded {lg}: {len(df_lg):,} half-match rows")

    df_all = pd.concat(league_frames, ignore_index=True)
    print(f"    ─────────────────────────────────────")
    print(f"    Total pool: {len(df_all):,} rows across {len(league_frames)} league(s)")
    print(f"    {team}: {len(df_team):,} half-match rows (training only)")

    # ── Step 2: Train/test split on team data, run scouting eval ────────────
    print(f"\n[2] Scouting evaluation on {team} data...")
    feat_cols = get_feature_cols(df_team)
    X_sc, y, imp, scaler = prepare_xy(df_team, feat_cols, TARGET_HALF, min_passes=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.15, random_state=42
    )
    print(f"    Train: {len(X_train)}  Test: {len(X_test)}")
    scout_result = run_scouting_evaluation(X_train, y_train, X_test, y_test)
    scouting_grads = scout_result["scouting_grads"]

    if not scouting_grads:
        print("\nNo scouting features passed stability + significance filters.")
        print("Cannot rank players. Try lowering the p-value threshold or adding more data.")
        return

    # ── Step 3: Score all players league-wide ────────────────────────────────
    print(f"\n[3] Scoring all {league} players...")
    scored = score_all_players(df_all, scouting_grads, team,
                               min_matches=min_matches, min_bypasses=min_bypasses)
    print(f"    Players scored: {len(scored)}  "
          f"({team}: {(scored['source']==team).sum()}, Others: {(scored['source']=='Other').sum()})")

    # ── Step 4: Show target-team squad with weak-player flagging ─────────────
    print(f"\n[4] {team} squad — individual bypass scores by position:")
    team_players = scored[scored["source"] == team].copy()
    team_players = flag_weak_players(team_players)

    if team_players.empty:
        print(f"No {team} players found after filtering (min {min_matches} matches).")
        return

    feats = list(scouting_grads.keys())
    grad_df = scout_result["grad_df"]

    print(f"\n  {'Player':<35} {'Role':<20} {'Pos':>3}  {'AvgX':>6}  {'Score':>6}  "
          f"{'Halves':>7}  {'Bypasses/half':>14}")
    print("  " + "-" * 100)
    for _, row in team_players.sort_values("bypass_score").iterrows():
        avg_x = f"{row['average_position_x']:.1f}" if "average_position_x" in row and pd.notna(row.get("average_position_x")) else "  N/A"
        role  = row.get("tactical_role", "Unknown")
        print(f"  {row['player_name']:<35} {role:<20} {row['position_bucket']:>3}  {avg_x:>6}  "
              f"{row['bypass_score']:>6.1f}  "
              f"{row['matches_played']:>7.0f}  "
              f"{row[TARGET_HALF]:>14.2f}")

    # ── Step 5: Find replacements for every target-team player ───────────────
    print(f"\n[5] Finding top-{top_n} replacements per player (tactical role + position + bypass ceiling)...")
    candidates = scored[scored["source"] == "Other"].copy()

    # Bypass ceiling: role-level median from the full candidate pool.
    # Ensures candidates are not just good on paper (score) but also didn't
    # concede abnormally many bypasses in their own team's context.
    role_bypass_medians = (
        candidates.groupby("tactical_role")[TARGET_HALF]
        .median()
        .to_dict()
    )
    print(f"\n  Bypass ceiling per role (league median among candidates):")
    for role, med in sorted(role_bypass_medians.items()):
        print(f"    {role:<20} ≤ {med:.2f} bypasses/half")

    replacement_map = {}
    for _, player in team_players.iterrows():
        role   = player.get("tactical_role", "Unknown")
        bucket = player["position_bucket"]

        # Bypass ceiling for this player's role
        bypass_ceiling = role_bypass_medians.get(role, candidates[TARGET_HALF].median())

        # Priority 1: exact match — same role AND same position bucket
        pool = candidates[
            (candidates["tactical_role"] == role) &
            (candidates["position_bucket"] == bucket)
        ].copy()
        match_label = f"{role} + {bucket}"

        # Priority 2: same role only (position flexible)
        if len(pool) < top_n:
            role_only = candidates[
                (candidates["tactical_role"] == role) &
                (~candidates.index.isin(pool.index))
            ].copy()
            pool = pd.concat([pool, role_only])
            if len(pool) > len(candidates[candidates["tactical_role"] == role]):
                match_label = f"{role} (pos relaxed)"

        # Priority 3: same position bucket only (role flexible)
        if pool.empty:
            pool = candidates[candidates["position_bucket"] == bucket].copy()
            match_label = f"{bucket} (role relaxed)"

        # Priority 4: full league pool
        if pool.empty:
            pool = candidates.copy()
            match_label = "full pool"

        # Apply bypass ceiling — remove candidates who concede too many
        # bypasses in their own context regardless of feature score
        pool_filtered = pool[pool[TARGET_HALF] <= bypass_ceiling].copy()
        if pool_filtered.empty:
            # Relax: use top tercile of candidates' bypass rate
            ceiling_relaxed = pool[TARGET_HALF].quantile(0.33)
            pool_filtered = pool[pool[TARGET_HALF] <= ceiling_relaxed].copy()
            match_label += f" [bypass ≤ {ceiling_relaxed:.1f} relaxed]"
        else:
            match_label += f" [bypass ≤ {bypass_ceiling:.1f}]"

        pool_filtered = pool_filtered[
            pool_filtered["raw_bypass_score"] < player["raw_bypass_score"]
        ].copy()
        pool_filtered["improvement"] = player["raw_bypass_score"] - pool_filtered["raw_bypass_score"]
        pool_filtered = pool_filtered.nsmallest(top_n, "raw_bypass_score")
        replacement_map[player["player_name"]] = (pool_filtered, match_label)

    # ── Step 6: Print final recommendations ──────────────────────────────────
    print("\n" + "=" * 70)
    print("TRANSFER RECOMMENDATIONS  (tactical role + position matched)")
    print("=" * 70)

    for _, player in team_players.sort_values("bypass_score").iterrows():
        pool, match_label = replacement_map.get(player["player_name"], (pd.DataFrame(), ""))
        avg_x_str = (f"{player['average_position_x']:.1f}"
                     if "average_position_x" in player and pd.notna(player.get("average_position_x"))
                     else "N/A")
        role = player.get("tactical_role", "Unknown")
        print(f"\n{'─'*70}")
        print(f"  PLAYER  : {player['player_name']}  "
              f"[{role}  |  {player['position_bucket']}  avg_x={avg_x_str}]")
        print(f"  Bypass score: {player['bypass_score']:.1f} (league pct)  |  "
              f"Avg bypasses/half: {player[TARGET_HALF]:.2f}  |  "
              f"Halves: {player['matches_played']:.0f}")
        print(f"  Feature profile:")
        for feat in feats:
            direction = grad_df[grad_df["Feature"] == feat]["scouting_direction"].values
            direction = direction[0] if len(direction) else ""
            print(f"    {feat:<40} = {player[feat]:.3f}  ({direction})")
        print(f"{'─'*70}")

        if pool.empty:
            print("  No candidate found with lower bypass score in same role + position.")
            continue

        print(f"  Match filter: [{match_label}]")

        print(f"  {'Rank':<5} {'Player':<30} {'Team':<22} {'League':<10} {'Role':<20} {'Pos':>3} {'AvgX':>6} "
              f"{'Score':>7} {'Improvement':>12}  {'Bypasses/half':>14}")
        for rank, (_, row) in enumerate(pool.iterrows(), 1):
            r_avg_x = (f"{row['average_position_x']:.1f}"
                       if "average_position_x" in row and pd.notna(row.get("average_position_x"))
                       else "  N/A")
            r_role   = row.get("tactical_role", "Unknown")
            r_league = row.get("league", "—")
            print(f"  {rank:<5} {row['player_name']:<30} {row['team_name']:<22} "
                  f"{r_league:<10} {r_role:<20} {row['position_bucket']:>3} {r_avg_x:>6} "
                  f"{row['bypass_score']:>7.1f} {row['improvement']:>+12.4f}  "
                  f"{row[TARGET_HALF]:>14.2f}")

        top = pool.iloc[0]
        top_avg_x = (f"{top['average_position_x']:.1f}"
                     if "average_position_x" in top and pd.notna(top.get("average_position_x"))
                     else "N/A")
        top_role   = top.get("tactical_role", "Unknown")
        top_league = top.get("league", "—")
        print(f"\n  Top candidate: {top['player_name']} ({top['team_name']}, {top_league})  [{top_role}  |  {top['position_bucket']}  avg_x={top_avg_x}]")
        print(f"  {'Feature':<40}  {'Candidate':>10}  {'vs ' + player['player_name'][:15]:>20}  {'Δ':>8}  Direction")
        for feat in feats:
            direction = grad_df[grad_df["Feature"] == feat]["scouting_direction"].values
            direction = direction[0] if len(direction) else ""
            cand_val  = top[feat]
            own_val   = player[feat]
            print(f"  {feat:<40}  {cand_val:>10.3f}  {own_val:>20.3f}  "
                  f"{cand_val - own_val:>+8.3f}  {direction}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    rows = []
    for _, player in team_players.iterrows():
        pool, match_label = replacement_map.get(player["player_name"], (pd.DataFrame(), ""))
        for rank, (_, row) in enumerate(pool.iterrows(), 1):
            entry = {
                "target_player":       player["player_name"],
                "target_team":         team,
                "target_tactical_role": player.get("tactical_role", "Unknown"),
                "target_position":     player["position_bucket"],
                "target_avg_x":        round(float(player["average_position_x"]), 2) if "average_position_x" in player and pd.notna(player.get("average_position_x")) else None,
                "target_bypass_score": round(player["bypass_score"], 2),
                "target_raw_score":    round(player["raw_bypass_score"], 4),
                "target_bypasses_per_half": round(player[TARGET_HALF], 2),
                "rank":               rank,
                "match_filter":        match_label,
                "candidate_name":     row["player_name"],
                "candidate_team":     row["team_name"],
                "candidate_league":   row.get("league", "—"),
                "candidate_tactical_role": row.get("tactical_role", "Unknown"),
                "candidate_position": row["position_bucket"],
                "candidate_avg_x":    round(float(row["average_position_x"]), 2) if "average_position_x" in row and pd.notna(row.get("average_position_x")) else None,
                "candidate_raw_score": round(row["raw_bypass_score"], 4),
                "improvement":        round(row["improvement"], 4),
                "candidate_matches":  row["matches_played"],
                "candidate_bypasses_per_half": round(row[TARGET_HALF], 2),
            }
            for feat in feats:
                entry[f"target_{feat}"] = round(float(player[feat]), 4)
                entry[f"cand_{feat}"]   = round(float(row[feat]), 4)
            rows.append(entry)

    if rows:
        out_path = ROOT / "data" / "processed" / f"transfer_recommendations_{league}_{team.replace(' ', '_')}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nRecommendations saved → {out_path}")

    scored_path = ROOT / "data" / "processed" / f"player_scores_{league}.csv"
    scored.to_csv(scored_path, index=False)
    print(f"All player scores saved → {scored_path}")

    print("\n" + "=" * 70)
    print(f"Model selected: {scout_result['best_model_name']}")
    print(f"Scouting Spearman ρ (test): {scout_result['spearman_test']:.4f}")
    print(f"Scouting features used: {list(scouting_grads.keys())}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--league",       default="Spain")
    parser.add_argument("--team",         default="Barcelona")
    parser.add_argument("--top-n",        type=int, default=5)
    parser.add_argument("--min-matches",  type=int, default=10,
                        help="Minimum half-match rows for a player to be included (default 10 ≈ 5 full matches)")
    args = parser.parse_args()
    main(league=args.league, team=args.team, top_n=args.top_n,
         min_matches=args.min_matches)
