"""
FastAPI application entry point.

Run from project root:
    uvicorn production.api.main:app --reload --port 8000
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from production.backend.config import settings
from production.backend.api.routers import ingestion, eda, feature_selection, model_building, best_model, replacement

app = FastAPI(
    title="DM-Bypass Pipeline API",
    description="6-stage midfielder analytics pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREFIX = "/api/v1"

app.include_router(ingestion.router, prefix=PREFIX)
app.include_router(eda.router, prefix=PREFIX)
app.include_router(feature_selection.router, prefix=PREFIX)
app.include_router(model_building.router, prefix=PREFIX)
app.include_router(best_model.router, prefix=PREFIX)
app.include_router(replacement.router, prefix=PREFIX)


@app.get(f"{PREFIX}/health", tags=["Utility"])
def health():
    """Return API liveness status."""
    return {"status": "ok"}


@app.get(f"{PREFIX}/pipeline/state", tags=["Utility"])
def pipeline_state():
    """
    Inspect disk artifacts and report which pipeline stages are complete.
    Allows the frontend to resume mid-pipeline after a page refresh.
    """
    processed = Path(settings.data_processed_dir)
    models = Path(settings.models_dir)
    LEAGUE_NAMES = ["Spain", "England", "France", "Germany", "Italy"]

    def has_features() -> bool:
        """Return True if at least one non-selected wyscout features CSV exists."""
        if not processed.exists():
            return False
        return any(
            f for f in processed.glob("wyscout_*_features.csv")
            if "selected" not in f.name
        )

    def has_selected_features() -> bool:
        """Return True if a selected-features sidecar JSON exists for any league."""
        if not processed.exists():
            return False
        return any(processed.glob("wyscout_*_features_selected_features.json"))

    def has_models() -> bool:
        """Return True if any trained model pickle exists."""
        if not models.exists():
            return False
        return any(models.glob("*.pkl"))

    def has_scaler() -> bool:
        """Return True if the shared scaler pickle exists."""
        return (models / "scaler.pkl").exists()

    stage1_done = has_features()
    # Stage 2 (EDA) produces no mandatory disk artifact, so it mirrors stage 1 completion.
    stage2_done = stage1_done
    stage3_done = has_selected_features()
    stage4_done = has_models() and has_scaler()
    stage5_done = stage4_done
    stage6_done = stage4_done and has_selected_features()

    selected_features: list[str] = []
    if processed.exists():
        for league in LEAGUE_NAMES:
            sidecar = processed / f"wyscout_{league}_features_selected_features.json"
            if sidecar.exists():
                try:
                    data = json.loads(sidecar.read_text())
                    selected_features = data.get("selected_features", [])
                    break
                except Exception:
                    pass

    features_paths: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    leagues_processed: list[str] = []

    if processed.exists():
        import pandas as pd
        for league in LEAGUE_NAMES:
            csv_path = processed / f"wyscout_{league}_features.csv"
            if csv_path.exists() and "player_selected" not in csv_path.name:
                features_paths[league] = str(csv_path)
                leagues_processed.append(league)
                try:
                    row_counts[league] = len(pd.read_csv(csv_path, usecols=[0]))
                except Exception:
                    row_counts[league] = 0

    model_paths: dict[str, str] = {}
    if models.exists():
        for pkl in models.glob("*.pkl"):
            if pkl.stem != "scaler":
                model_paths[pkl.stem] = str(pkl)

    return {
        "stages": {
            "1": stage1_done,
            "2": stage2_done,
            "3": stage3_done,
            "4": stage4_done,
            "5": stage5_done,
            "6": stage6_done,
        },
        "features_paths":    features_paths,
        "row_counts":        row_counts,
        "leagues_processed": leagues_processed,
        "selected_features": selected_features,
        "model_paths":       model_paths,
        "scaler_path":       str(models / "scaler.pkl") if has_scaler() else None,
    }


frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir), html=True), name="static")
