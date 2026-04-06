"""
Stage 3 router — feature selection.

  POST /stage3/select              trigger feature selection job
  GET  /stage3/status/{job_id}     poll job status
"""
import uuid
import asyncio

from fastapi import APIRouter, Depends, HTTPException

from production.backend.api.dependencies import get_task_store, get_executor
from production.backend.services.task_store import TaskStore
from production.backend.schemas.feature_selection import FeatureSelectionRequest
from production.backend.workers.pipeline_worker import run_stage3

router = APIRouter(prefix="/stage3", tags=["Stage 3 – Feature Selection"])


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


@router.post("/select")
async def select_features(
    body: FeatureSelectionRequest,
    store: TaskStore = Depends(get_task_store),
    executor=Depends(get_executor),
):
    """Enqueue a Stage 3 feature selection job and return its job_id."""
    job_id = str(uuid.uuid4())
    await store.create(job_id)

    asyncio.create_task(
        run_stage3(
            job_id=job_id,
            params={
                "features_path": body.features_path,
                "target_col": body.target_col,
                "n_top": body.n_top,
            },
            store=store,
            executor=executor,
        )
    )

    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def select_status(job_id: str, store: TaskStore = Depends(get_task_store)):
    """Poll the status of a feature selection job."""
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return task_response(record)
