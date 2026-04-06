"""
Stage 6 router — replacement finder.

  POST /stage6/find-replacements     trigger replacement job
  GET  /stage6/status/{job_id}       poll job status
  GET  /stage6/players               list all available players across leagues
"""
import uuid
import asyncio
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from production.backend.api.dependencies import get_task_store, get_executor
from production.backend.services.task_store import TaskStore
from production.backend.schemas.replacement import ReplacementRequest
from production.backend.config import settings
from production.backend.workers.pipeline_worker import run_stage6

router = APIRouter(prefix="/stage6", tags=["Stage 6 – Replacement Finder"])

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


@router.post("/find-replacements")
async def find_replacements(
    body: ReplacementRequest,
    store: TaskStore = Depends(get_task_store),
    executor=Depends(get_executor),
):
    """Enqueue a Stage 6 replacement-finder job and return its job_id."""
    job_id = str(uuid.uuid4())
    await store.create(job_id)

    asyncio.create_task(
        run_stage6(
            job_id=job_id,
            params={
                "league":       body.league,
                "team":         body.team,
                "top_n":        body.top_n,
                "min_matches":  body.min_matches,
            },
            store=store,
            executor=executor,
        )
    )

    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def replacement_status(job_id: str, store: TaskStore = Depends(get_task_store)):
    """Poll the status of a replacement-finder job."""
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return task_response(record)


@router.get("/players")
def list_players():
    """
    Scan all wyscout_{league}_features.csv files and return unique player names
    tagged with their team and league.
    """
    processed_dir = Path(settings.data_processed_dir)
    if not processed_dir.exists():
        return []

    rows = []
    seen = set()

    for league in ALL_LEAGUES:
        csv_path = processed_dir / f"wyscout_{league}_features.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(
                csv_path,
                usecols=lambda c: c in {"player_name", "team_name"},
            )
            if "player_name" not in df.columns:
                continue
            team_col = "team_name" if "team_name" in df.columns else None
            for _, row in df.drop_duplicates(subset=["player_name"]).iterrows():
                name = str(row["player_name"])
                if name in seen:
                    continue
                seen.add(name)
                rows.append({
                    "player_name": name,
                    "team":        str(row[team_col]) if team_col else "",
                    "league":      league,
                })
        except Exception:
            continue

    return sorted(rows, key=lambda x: x["player_name"])
