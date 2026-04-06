from pydantic import BaseModel
from typing import Any


class BestModelRequest(BaseModel):
    model_path: str
    scaler_path: str
    features_path: str
    selected_features: list[str]
    target_col: str = "bypasses_per_halftime"


class CoefficientInfo(BaseModel):
    feature: str
    coefficient: float
    relative_importance: float  # abs(coef) / sum(abs) * 100


class BestModelResult(BaseModel):
    coefficients: list[CoefficientInfo]
    gradient_sensitivity: list[dict[str, Any]]
    model_name: str
