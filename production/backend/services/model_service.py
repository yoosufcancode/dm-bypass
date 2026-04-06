"""Stage 4: Model building — MLR, Ridge, Lasso with LOOCV + test evaluation."""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def run_model_building(
    features_path: str,
    selected_features: list[str],
    target_col: str,
    test_size: float,
    random_state: int,
    progress_cb: Callable[[int, str], None],
) -> dict:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.model import (
        build_mlr,
        build_ridge,
        build_lasso,
        evaluate_loocv,
        evaluate_model_on_test_set,
        save_model,
    )

    # Default target column is bypasses_per_halftime (half-match rows, no aggregation)
    if not target_col:
        target_col = "bypasses_per_halftime"

    progress_cb(5, "Loading dataset")
    df = pd.read_csv(features_path)

    available = [f for f in selected_features if f in df.columns]
    df_clean = df[available + [target_col] + (["season"] if "season" in df.columns else [])].dropna(
        subset=available + [target_col]
    )
    X = df_clean[available]
    y = df_clean[target_col]

    progress_cb(10, "Splitting train/test")
    if "season" in df_clean.columns:
        # Temporal split: last season → test, all prior seasons → train (no data leakage)
        seasons = sorted(df_clean["season"].unique())
        test_season = seasons[-1]
        train_mask = df_clean["season"] != test_season
        test_mask  = df_clean["season"] == test_season
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

    progress_cb(15, "Fitting scaler")
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=available, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=available, index=X_test.index)
    X_all_sc   = pd.DataFrame(scaler.transform(X),           columns=available, index=X.index)
    y_all      = y  # aligned with X_all_sc

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = models_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    results = []
    model_configs = [
        ("MLR",   lambda: build_mlr(X_train_sc, y_train),   "mlr_model.pkl"),
        ("Ridge", lambda: build_ridge(X_train_sc, y_train), "ridge_model.pkl"),
        ("Lasso", lambda: build_lasso(X_train_sc, y_train), "lasso_model.pkl"),
    ]

    for i, (name, builder, filename) in enumerate(model_configs):
        base_pct = 20 + i * 25
        progress_cb(base_pct, f"Training {name}")
        model = builder()

        progress_cb(base_pct + 8, f"LOOCV evaluation — {name}")
        loocv_metrics = evaluate_loocv(model, X_all_sc, y_all, name)

        progress_cb(base_pct + 16, f"Test set evaluation — {name}")
        test_metrics = evaluate_model_on_test_set(model, X_test_sc, y_test, name)

        model_path = models_dir / filename
        save_model(model, model_path)

        results.append({
            "name": name,
            "loocv": {
                "spearman":   round(float(loocv_metrics["Spearman"]),   4),
                "spearman_p": round(float(loocv_metrics["Spearman_p"]), 4),
                "r2":         round(float(loocv_metrics["R2"]),         4),
                "rmse":       round(float(loocv_metrics["RMSE"]),       4),
                "mae":        round(float(loocv_metrics["MAE"]),        4),
            },
            "test": {
                "spearman":   round(float(test_metrics["Spearman"]),   4),
                "spearman_p": round(float(test_metrics["Spearman_p"]), 4),
                "r2":         round(float(test_metrics["R2"]),         4),
                "rmse":       round(float(test_metrics["RMSE"]),       4),
                "mae":        round(float(test_metrics["MAE"]),        4),
            },
            "model_path": str(model_path),
        })

    progress_cb(95, "Selecting best model by LOOCV Spearman ρ")
    best = max(results, key=lambda r: r["loocv"]["spearman"])

    progress_cb(100, "Model building complete")

    return {
        "models": results,
        "feature_count": len(available),
        "best_model": best["name"],
        "scaler_path": str(scaler_path),
    }
