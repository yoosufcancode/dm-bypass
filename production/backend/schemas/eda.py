from pydantic import BaseModel
from typing import Any


class EDARequest(BaseModel):
    features_path: str


class EDAResult(BaseModel):
    descriptive_stats: dict[str, Any]
    correlation_matrix: dict[str, Any]
    missing_values: dict[str, int]
    bypass_distribution: dict[str, Any]
    row_count: int
    column_count: int
