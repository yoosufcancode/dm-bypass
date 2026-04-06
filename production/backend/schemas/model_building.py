from pydantic import BaseModel
from typing import Any


class ModelBuildRequest(BaseModel):
    features_path: str
    selected_features: list[str]
    target_col: str = "bypasses_per_halftime"
    test_size: float = 0.15
    random_state: int = 42


class ModelMetrics(BaseModel):
    spearman: float
    spearman_p: float
    r2: float
    rmse: float
    mae: float


class ModelResult(BaseModel):
    name: str
    loocv: ModelMetrics
    test: ModelMetrics
    model_path: str


class ModelBuildResult(BaseModel):
    models: list[ModelResult]
    feature_count: int
    best_model: str           # selected by LOOCV Spearman ρ
    scaler_path: str
