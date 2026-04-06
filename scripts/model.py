"""
Model building utilities for bypass prediction.
"""

import pandas as pd
import numpy as np
import warnings
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
from typing import Tuple, List, Optional, Union, Dict, Any

warnings.filterwarnings('ignore')


def load_data(
    data_path: Union[str, Path],
    target_col: str = 'bypasses_per_halftime',
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load data from CSV and separate features from target.
    
    Parameters
    ----------
    data_path : str or Path
        Path to CSV file
    target_col : str, default='bypasses_per_match'
        Target variable column name
    
    Returns
    -------
    X, y, feature_cols : DataFrame, Series, list
        Features, target, and feature column names
    """
    exclude_cols = ['match_id', 'team_id', 'team_name', 'season', 'bypasses_per_halftime']
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns, Missing: {df.isnull().sum().sum()}")
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y, feature_cols


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    random_state: int = 42,
    shuffle: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets.
    
    Parameters
    ----------
    X, y : DataFrame, Series
        Features and target
    test_size : float, default=0.10
        Test set proportion
    random_state : int, default=42
        Random seed
    shuffle : bool, default=True
        Shuffle before splitting
    
    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrame, DataFrame, Series, Series
        Train/test splits
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    
    print(f"Split: {X_train.shape[0]} train ({X_train.shape[0]/len(X)*100:.1f}%), "
          f"{X_test.shape[0]} test ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def load_and_split(
    data_path: Union[str, Path],
    test_size: float = 0.15,
    random_state: int = 42,
    target_col: str = 'bypasses_per_halftime',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Load data and split into train/test sets.
    
    Parameters
    ----------
    data_path : str or Path
        Path to CSV file
    test_size : float, default=0.10
        Test set proportion
    random_state : int, default=42
        Random seed
    target_col : str, default='bypasses_per_halftime'
        Target variable column
    
    Returns
    -------\
    X_train, X_test, y_train, y_test, feature_cols
        Train/test splits and feature names
    """
    X, y, feature_cols = load_data(data_path, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test, feature_cols


def build_mlr(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LinearRegression:
    """Build and train Multiple Linear Regression model.
    
    Parameters
    ----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    
    Returns
    -------\
    model : LinearRegression
        Trained MLR model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"MLR trained: intercept={model.intercept_:.4f}, coefficients={len(model.coef_)}")
    return model


def build_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: Optional[np.ndarray] = None,
    cv: Optional[int] = None,
    random_state: int = 42
) -> RidgeCV:
    """Build and train Ridge regression model with cross-validation.
    
    Parameters
    ----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    alphas : array-like, optional
        Regularization strengths (default: np.logspace(-2, 3, 50))
    cv : int or cross-validation generator, optional
        Cross-validation strategy (default: LeaveOneOut for small datasets)
    random_state : int, default=42
        Random seed
    
    Returns
    -------\
    model : RidgeCV
        Trained Ridge model with optimal alpha
    """
    if alphas is None:
        alphas = np.logspace(-2, 3, 50)
    if cv is None:
        cv = LeaveOneOut()
    
    model = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    print(f"Ridge trained: optimal alpha={model.alpha_:.4f}")
    return model


def build_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: Optional[np.ndarray] = None,
    cv: Optional[int] = None,
    random_state: int = 42,
    max_iter: int = 2000,
    n_jobs: int = -1
) -> LassoCV:
    """Build and train Lasso regression model with cross-validation.
    
    Parameters
    ----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    alphas : array-like, optional
        Regularization strengths (default: np.logspace(-4, 1, 50))
    cv : int or cross-validation generator, optional
        Cross-validation strategy (default: LeaveOneOut for small datasets)
    random_state : int, default=42
        Random seed
    max_iter : int, default=2000
        Maximum iterations for convergence
    n_jobs : int, default=-1
        Number of parallel jobs
    
    Returns
    -------\
    model : LassoCV
        Trained Lasso model with optimal alpha
    """
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)
    if cv is None:
        cv = LeaveOneOut()
    
    model = LassoCV(
        alphas=alphas,
        cv=cv,
        random_state=random_state,
        max_iter=max_iter,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    print(f"Lasso trained: optimal alpha={model.alpha_:.4f}")
    return model


def evaluate_loocv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """Evaluate model using Leave-One-Out Cross-Validation (LOOCV).
    
    Parameters
    ----------
    model : sklearn model
        Trained model to evaluate
    X : DataFrame
        Full feature matrix
    y : Series
        Full target vector
    model_name : str, default=\"Model\"
        Name of the model for display
    
    Returns
    -------\
    metrics : dict
        Dictionary containing:
        - \'R2\': R² score (mean)
        - \'R2_std\': R² score standard deviation
        - \'RMSE\': Root Mean Squared Error (mean)
        - \'RMSE_std\': RMSE standard deviation
        - \'MAE\': Mean Absolute Error
        - \'MAPE\': Mean Absolute Percentage Error
    
    """
    loo = LeaveOneOut()
    
    # Manually implement LOOCV to avoid cloning issues with saved models
    y_pred = np.zeros(len(y))
    r2_scores_list = []
    mse_scores_list = []
    
    for train_idx, test_idx in loo.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Predict on test fold
        y_pred[test_idx] = model.predict(X_test_fold)
        
        # Calculate scores for this fold
        fold_r2 = r2_score(y_test_fold, y_pred[test_idx])
        fold_mse = mean_squared_error(y_test_fold, y_pred[test_idx])
        r2_scores_list.append(fold_r2)
        mse_scores_list.append(-fold_mse)  # Negative for consistency
    
    # Calculate metrics from predictions
    spearman_rho, spearman_p = spearmanr(y, y_pred)
    if y.var() == 0:
        print("Warning: Target variable 'y' has zero variance. R2 score is undefined and will be set to 0.")
        r2 = 0.0
    else:
        r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # Calculate standard deviations, filtering out NaNs from single-sample R2 scores
    r2_scores_list_filtered = [score for score in r2_scores_list if not np.isnan(score)]
    r2_std = np.std(r2_scores_list_filtered) if r2_scores_list_filtered else np.nan
    rmse_std = np.sqrt(np.std(mse_scores_list))

    metrics = {
        'Spearman': spearman_rho,
        'Spearman_p': spearman_p,
        'R2': r2,
        'R2_std': r2_std,
        'RMSE': rmse,
        'RMSE_std': rmse_std,
        'MAE': mae,
    }

    print(f"\n{model_name} - LOOCV Evaluation:")
    print(f"  Spearman rho: {spearman_rho:.4f}  (p={spearman_p:.4f})  <- primary metric")
    print(f"  R²:   {r2:.4f} ± {r2_std:.4f}")
    print(f"  RMSE: {rmse:.4f} ± {rmse_std:.4f}")
    print(f"  MAE:  {mae:.4f}\n")

    return metrics


def evaluate_model_on_test_set(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """Evaluate model on a dedicated test set.
    
    Parameters
    ----------
    model : sklearn model
        Trained model to evaluate
    X_test : DataFrame
        Test feature matrix
    y_test : Series
        Test target vector
    model_name : str, default=\"Model\"
        Name of the model for display
    
    Returns
    -------\
    metrics : dict
        Dictionary containing:
        - \'R2\': R² score
        - \'RMSE\': Root Mean Squared Error
        - \'MAE\': Mean Absolute Error
        - \'MAPE\': Mean Absolute Percentage Error
    
    """
    y_pred = model.predict(X_test)

    spearman_rho, spearman_p = spearmanr(y_test, y_pred)
    if y_test.var() == 0:
        print("Warning: Target variable 'y_test' has zero variance. R2 score is undefined and will be set to 0.")
        r2 = 0.0
    else:
        r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        'Spearman': spearman_rho,
        'Spearman_p': spearman_p,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
    }

    print(f"\n{model_name} - Test Set Evaluation:")
    print(f"  Spearman rho: {spearman_rho:.4f}  (p={spearman_p:.4f})  <- primary metric")
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}\n")

    return metrics


def save_model(
    model: Any,
    file_path: Union[str, Path],\
) -> None:
    """Save a trained model to a pickle file.
    
    Parameters
    ----------
    model : sklearn model
        Trained model to save
    file_path : str or Path
        Path where to save the model (should end with .pkl)
    
    """
    file_path = Path(file_path)
    
    # Create parent directory if it doesn\\'t exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .pkl extension
    if file_path.suffix != '.pkl':
        file_path = file_path.with_suffix('.pkl')   
    
    # Save model
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {file_path}")


def load_model(
    file_path: Union[str, Path]
) -> Any:
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {file_path}")
    
    return model