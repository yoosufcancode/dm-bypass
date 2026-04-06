"""
Stage 1 router — ingestion + utility endpoints.

Pipeline endpoints
  POST /stage1/ingest               trigger ingestion job
  GET  /stage1/status/{job_id}      poll job status

Utility endpoints
  GET  /stage1/teams                list available leagues and teams from existing CSVs
  GET  /stage1/available-data       list already-processed wyscout_{league}_features.csv files
"""
import uuid
import asyncio
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from production.backend.api.dependencies import get_task_store, get_executor
from production.backend.services.task_store import TaskStore, TaskStatus
from production.backend.schemas.ingestion import IngestRequest
from production.backend.config import settings
from production.backend.workers.pipeline_worker import run_stage1

router = APIRouter(prefix="/stage1", tags=["Stage 1 – Ingestion"])

ALL_LEAGUES = ["Spain", "England", "France", "Germany", "Italy"]


def task_response(record) -> dict:
    """Serialize a TaskRecord to a status response dict."""
    return {
        "job_id": record.job_id,
        "status": record.status,
        "progress": record.progress,
        "message": record.message,
        "result": record.result,
        "error": record.error,
    }


@router.post("/ingest")
async def ingest(
    body: IngestRequest,
    store: TaskStore = Depends(get_task_store),
    executor=Depends(get_executor),
):
    """Enqueue a Stage 1 ingestion job and return its job_id."""
    job_id = str(uuid.uuid4())
    await store.create(job_id)

    asyncio.create_task(
        run_stage1(
            job_id=job_id,
            params={
                "leagues":       body.leagues,
                "skip_download": body.skip_download,
            },
            store=store,
            executor=executor,
        )
    )

    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def ingest_status(job_id: str, store: TaskStore = Depends(get_task_store)):
    """Poll the status of an ingestion job."""
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return task_response(record)


@router.get("/teams")
def list_teams():
    """
    Return available leagues and teams from existing wyscout_{league}_features.csv files.
    If CSVs don't exist yet, return leagues list only.
    """
    processed_dir = Path(settings.data_processed_dir)

    leagues_found = []
    teams_by_league: dict[str, list[str]] = {}

    for league in ALL_LEAGUES:
        csv_path = processed_dir / f"wyscout_{league}_features.csv"
        if not csv_path.exists():
            continue
        leagues_found.append(league)
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"team_name"})
            if "team_name" in df.columns:
                teams_by_league[league] = sorted(df["team_name"].dropna().unique().tolist())
            else:
                teams_by_league[league] = []
        except Exception:
            teams_by_league[league] = []

    return {
        "leagues": ALL_LEAGUES,
        "leagues_with_data": leagues_found,
        "teams_by_league": teams_by_league,
    }


@router.get("/available-data")
def available_data():
    """
    Scan data/processed/ for wyscout_{league}_features.csv files and return metadata.
    Returns {league, features_path, row_count} for each found file.
    """
    processed_dir = Path(settings.data_processed_dir)
    if not processed_dir.exists():
        return []

    results: list[dict] = []
    for league in ALL_LEAGUES:
        csv_path = processed_dir / f"wyscout_{league}_features.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"player_id"})
            row_count = int(df["player_id"].count()) if "player_id" in df.columns else 0
        except Exception:
            row_count = 0

        results.append({
            "league":        league,
            "features_path": str(csv_path),
            "row_count":     row_count,
        })

    return results
