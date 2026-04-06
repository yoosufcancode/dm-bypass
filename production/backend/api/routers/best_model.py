"""
Stage 5 router — best model analysis.

  POST /stage5/analyze             trigger analysis job
  GET  /stage5/status/{job_id}     poll job status
"""
import uuid
import asyncio

from fastapi import APIRouter, Depends, HTTPException

from production.backend.api.dependencies import get_task_store, get_executor
from production.backend.services.task_store import TaskStore
from production.backend.schemas.best_model import BestModelRequest
from production.backend.workers.pipeline_worker import run_stage5

router = APIRouter(prefix="/stage5", tags=["Stage 5 – Best Model Analysis"])


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


@router.post("/analyze")
async def analyze_best_model(
    body: BestModelRequest,
    store: TaskStore = Depends(get_task_store),
    executor=Depends(get_executor),
):
    """Enqueue a Stage 5 best-model analysis job and return its job_id."""
    job_id = str(uuid.uuid4())
    await store.create(job_id)

    asyncio.create_task(
        run_stage5(
            job_id=job_id,
            params={
                "model_path":        body.model_path,
                "scaler_path":       body.scaler_path,
                "features_path":     body.features_path,
                "selected_features": body.selected_features,
                "target_col":        body.target_col,
            },
            store=store,
            executor=executor,
        )
    )

    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def analyze_status(job_id: str, store: TaskStore = Depends(get_task_store)):
    """Poll the status of a best-model analysis job."""
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return task_response(record)
