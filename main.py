"""
Main script to load and evaluate saved models.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from model import load_and_split, load_model, evaluate_model_on_test_set


def load_and_evaluate_models(
    models_dir: str = "models",
    data_path: str = "data/processed/Barcelona_2014_2015_selected_features_scaled.csv",
    test_size: float = 0.15,
) -> Dict[str, Dict[str, float]]:
    """Load all models from directory and evaluate them on a test set.
    
    Parameters
    ----------
    models_dir : str, default="models"
        Directory containing saved model files (.pkl)
    data_path : str, default="data/processed/Barcelona_2014_2015_selected_features_scaled.csv"
        Path to the data file for evaluation
    test_size : float, default=0.15
        Proportion of the dataset to include in the test split.
    
    Returns
    -------
    results : dict
        Dictionary with model names as keys and evaluation metrics as values
        Each value contains: 'R2', 'RMSE', 'MAE', 'MAPE'
    
    """
    models_dir = Path(models_dir)
    data_path = Path(data_path)
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load and split data
    print("=" * 80)
    print("LOADING AND SPLITTING DATA")
    print("=" * 80)
    X_train, X_test, y_train, y_test, feature_cols = load_and_split(data_path, test_size=test_size)
    

    
    # Find all model files
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    print(f"\n{'=' * 80}")
    print(f"LOADING MODELS FROM {models_dir}")
    print(f"{'=' * 80}")
    print(f"Found {len(model_files)} model(s):")
    for f in model_files:
        print(f"  - {f.name}")
    
    # Load models and evaluate
    results = {}
    models = {}
    
    for model_file in model_files:
        model_name = model_file.stem.replace('_model', '').upper()
        
        print(f"\n{'=' * 80}")
        print(f"EVALUATING {model_name} MODEL ON TEST SET")
        print(f"{'=' * 80}")
        
        # Load model
        model = load_model(model_file)
        models[model_name] = model
        
        # Evaluate on test set
        metrics = evaluate_model_on_test_set(model, X_test, y_test, model_name=model_name)
        results[model_name] = metrics
    
    # Print comparison
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<15} {'R²':>12} {'RMSE':>12} {'MAE':>12} {'MAPE':>12}")
    print("-" * 63)
    for model_name in sorted(results.keys()):
        m = results[model_name]
        print(f"{model_name:<15} {m['R2']:.4f} {m['RMSE']:.4f} {m['MAE']:.4f} {m['MAPE']:.2f}%")
    
    return results


if __name__ == "__main__":
    results = load_and_evaluate_models(test_size=0.15)
    
    # Find and display best model
    best_model = min(results, key=lambda x: results[x]['RMSE'])
    best_metrics = results[best_model]
    
    print(f"\n{'=' * 80}")
    print("BEST MODEL")
    print(f"{'=' * 80}")
    print(f"Model: {best_model}")
    print(f"  R²:  {best_metrics['R2']:.4f}")
    print(f"  RMSE: {best_metrics['RMSE']:.4f}")
    print(f"  MAE:  {best_metrics['MAE']:.4f}")
    print(f"  MAPE: {best_metrics['MAPE']:.2f}%")
    print(f"{'=' * 80}")
    print("\n✓ Evaluation complete!")
