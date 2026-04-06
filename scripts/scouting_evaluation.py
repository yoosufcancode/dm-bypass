"""
Scouting-appropriate model evaluation metrics.

Two-model approach:
  1. PREDICTIVE model  (final_model on best_features)  → Spearman ρ evaluation
  2. SCOUTING model    (Ridge on ALL player features)   → gradient weights for ranking

Called from model_building.ipynb after final_model, X_train_best, X_test_best,
best_features, y_train, y_test, X_train_all, X_test_all are defined.
"""
import statsmodels.api as sm
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
import numpy as np
import pandas as pd
from pathlib import Path


def evaluate_scouting_model(
    final_model, X_train_best, X_test_best,
    best_features, y_train, y_test,
    X_train_all=None, X_test_all=None,
    grad_path="../data/processed/mlr_gradient_analysis.csv",
):
    print("=" * 80)
    print("SCOUTING MODEL EVALUATION")
    print("=" * 80)

    # ── 1. Spearman Rank Correlation (predictive model on best features) ──────
    preds_train = final_model.predict(X_train_best)
    preds_test  = final_model.predict(X_test_best)

    spear_train, _ = spearmanr(y_train, preds_train)
    spear_test,  _ = spearmanr(y_test,  preds_test)
    kendall_test, _ = kendalltau(y_test, preds_test)

    print("\n1. RANKING QUALITY  (Ridge on best features — primary evaluation)")
    print(f"   Spearman rho (train) : {spear_train:.4f}")
    print(f"   Spearman rho (test)  : {spear_test:.4f}  <- primary metric")
    print(f"   Kendall tau  (test)  : {kendall_test:.4f}")
    pct = (spear_test + 1) / 2 * 100
    print(f"   Model correctly ranks {pct:.1f}% of match-half pairs by bypass count")

    # ── 2. Scouting Ridge — run on ALL player features (no opponent context) ──
    # Ridge is used here (not Lasso) to retain ALL player features with
    # non-zero coefficients, giving richer scouting profiles.
    # Opponent features are excluded — they're confounders, not player attributes.
    OPP_PREFIXES = ("opp_", "score_diff")
    if X_train_all is not None:
        all_features = list(X_train_all.columns)
        scout_features = [f for f in all_features
                          if not any(f.startswith(p) for p in OPP_PREFIXES)]
        X_scout_train = X_train_all[scout_features]
        X_scout_test  = X_test_all[scout_features]
        print(f"\n   Scouting Ridge trained on {len(scout_features)} player features "
              f"(opponent context excluded)")
    else:
        # Fallback: use best_features only
        scout_features = [f for f in best_features
                          if not any(f.startswith(p) for p in OPP_PREFIXES)]
        X_scout_train = X_train_best[scout_features]
        X_scout_test  = X_test_best[scout_features]
        print(f"\n   Note: X_train_all not provided — using best_features subset.")

    # ── 3. Coefficient p-values via OLS (on scouting features) ───────────────
    print("\n2. COEFFICIENT SIGNIFICANCE  (p-values via OLS on player features)")
    print("   Features with p < 0.05 are reliable scouting criteria")
    print(f"   {'Feature':<38} {'beta':>10} {'p-value':>10}  Reliable?")
    print("   " + "-" * 70)

    X_sm = sm.add_constant(X_scout_train, has_constant="add")
    ols  = sm.OLS(y_train, X_sm).fit()

    trusted = []
    for fname in scout_features:
        b    = ols.params.get(fname, np.nan)
        pval = ols.pvalues.get(fname, np.nan)
        tag  = "YES" if pval < 0.05 else ("maybe" if pval < 0.10 else "NO")
        if pval < 0.05:
            trusted.append(fname)
        print(f"   {fname:<38} {b:>10.4f} {pval:>10.4f}  {tag}")

    print(f"\n   Trusted features (p<0.05): {trusted}")

    # ── 4. Sign Stability + CV Std (on scouting features) ────────────────────
    print("\n3 & 4. COEFFICIENT STABILITY ACROSS 5-FOLD CV  (scouting Ridge)")
    print(f"   {'Feature':<38} {'Mean_b':>8} {'Std_b':>8} {'Sign stable':>12}  Scout?")
    print("   " + "-" * 80)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_coefs = []
    for tr_idx, _ in kf.split(X_scout_train):
        Xf = X_scout_train.iloc[tr_idx]
        yf = y_train.iloc[tr_idx] if hasattr(y_train, "iloc") else y_train[tr_idx]
        fold_coefs.append(Ridge(alpha=1.0).fit(Xf, yf).coef_)

    fold_coefs = np.array(fold_coefs)
    mean_coefs = fold_coefs.mean(axis=0)
    std_coefs  = fold_coefs.std(axis=0)

    stability_rows = []
    scouting_grads = {}

    for i, fname in enumerate(scout_features):
        signs       = np.sign(fold_coefs[:, i])
        sign_stable = len(set(signs)) == 1
        mean_b      = mean_coefs[i]
        cv_std      = std_coefs[i]
        pval        = float(ols.pvalues.get(fname, 1.0))
        scout       = sign_stable and pval < 0.15

        if scout:
            scouting_grads[fname] = float(mean_b)

        stable_str = "stable" if sign_stable else "FLIPS"
        if not sign_stable:
            scout_str = "skip (unstable)"
        elif pval >= 0.15:
            scout_str = f"skip (p={pval:.2f})"
        else:
            scout_str = "SCOUT"
        print(f"   {fname:<38} {mean_b:>8.4f} {cv_std:>8.4f} {stable_str:>12}  {scout_str}")
        stability_rows.append({
            "feature": fname, "mean_coef": mean_b, "std_coef": cv_std,
            "sign_stable": sign_stable, "use_for_scouting": scout,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Spearman rho (test)         : {spear_test:.4f}  (predictive model)")
    print(f"  Kendall tau  (test)         : {kendall_test:.4f}")
    print(f"  Player features evaluated   : {len(scout_features)}")
    print(f"  Significant (p<0.05)        : {len(trusted)}/{len(scout_features)}")
    print(f"  Sign-stable                 : {sum(r['sign_stable'] for r in stability_rows)}/{len(stability_rows)}")
    print(f"  Safe scouting features      : {sum(r['use_for_scouting'] for r in stability_rows)}")

    # ── Save gradient file ────────────────────────────────────────────────────
    total_abs = sum(abs(v) for v in scouting_grads.values()) or 1.0
    rows = []
    for fname, mean_b in sorted(scouting_grads.items(), key=lambda x: abs(x[1]), reverse=True):
        pval  = float(ols.pvalues.get(fname, np.nan))
        cvstd = next(r["std_coef"] for r in stability_rows if r["feature"] == fname)
        if pval < 0.05:
            tier = "Tier 1 — high confidence"
        elif pval < 0.10:
            tier = "Tier 2 — moderate confidence"
        else:
            tier = "Tier 3 — indicative only"
        rows.append({
            "Feature":                fname,
            "Gradient (dy/dx)":       mean_b,
            "Abs_Gradient":           abs(mean_b),
            "Normalised Sensitivity": abs(mean_b) / total_abs,
            "p_value":                pval,
            "cv_std":                 cvstd,
            "sign_stable":            True,
            "confidence_tier":        tier,
            "scouting_direction":     "look for LOW" if mean_b > 0 else "look for HIGH",
        })

    grad_df = pd.DataFrame(rows)
    grad_df.to_csv(grad_path, index=False)

    print(f"\nGradient file saved: {grad_path}  ({len(grad_df)} scouting features)")
    print(f"{'Feature':<38} {'beta':>8} {'p':>7} {'cv_std':>8}  {'Tier':<30}  Direction")
    print("-" * 105)
    for _, row in grad_df.iterrows():
        print(f"  {row['Feature']:<36} {row['Gradient (dy/dx)']:>8.4f} "
              f"{row['p_value']:>7.3f} {row['cv_std']:>8.4f}  {row['confidence_tier']:<30}  {row['scouting_direction']}")

    return {
        "spearman_train":    spear_train,
        "spearman_test":     spear_test,
        "kendall_test":      kendall_test,
        "trusted_features":  trusted,
        "scouting_features": list(scouting_grads.keys()),
        "grad_df":           grad_df,
    }
