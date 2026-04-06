"""Stage 5: Best model analysis — coefficients, gradient sensitivity."""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable


def run_best_model_analysis(
    model_path: str,
    scaler_path: str,
    features_path: str,
    selected_features: list[str],
    target_col: str,
    progress_cb: Callable[[int, str], None],
) -> dict:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    progress_cb(5, "Loading model and scaler")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    progress_cb(30, "Loading dataset")
    df = pd.read_csv(features_path)
    available = [f for f in selected_features if f in df.columns]
    df_clean = df[available + [target_col]].dropna()
    X_raw = df_clean[available]

    # Detect model name from class name
    model_name = type(model).__name__

    # --- Coefficients ---
    progress_cb(55, "Extracting coefficients")
    if hasattr(model, "coef_"):
        coefs = np.array(model.coef_).flatten()
    else:
        raise ValueError(f"Model {model_name} has no coef_ attribute")

    abs_sum = np.sum(np.abs(coefs))
    rel_importance = (np.abs(coefs) / abs_sum * 100) if abs_sum > 0 else np.zeros_like(coefs)

    coefficients = sorted(
        [
            {
                "feature": f,
                "coefficient": round(float(c), 6),
                "relative_importance": round(float(r), 4),
            }
            for f, c, r in zip(available, coefs, rel_importance)
        ],
        key=lambda x: abs(x["coefficient"]),
        reverse=True,
    )

    # --- Gradient sensitivity ---
    # For a linear model: sensitivity = coef * std(feature in scaled space)
    # (how much the prediction shifts per 1-std move in each raw feature)
    progress_cb(80, "Computing gradient sensitivity")
    raw_stds = X_raw.std().values
    gradient_sensitivity = sorted(
        [
            {
                "feature": f,
                "sensitivity": round(float(c * s), 6),
            }
            for f, c, s in zip(available, coefs, raw_stds)
        ],
        key=lambda x: abs(x["sensitivity"]),
        reverse=True,
    )

    progress_cb(100, "Best model analysis complete")

    return {
        "coefficients": coefficients,
        "gradient_sensitivity": gradient_sensitivity,
        "model_name": model_name,
    }
