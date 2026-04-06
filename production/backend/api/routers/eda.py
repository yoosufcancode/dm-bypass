"""
Stage 2 router — EDA.

  POST /stage2/analyze             trigger EDA job
  GET  /stage2/status/{job_id}     poll job status
"""
import uuid
import asyncio

from fastapi import APIRouter, Depends, HTTPException

from production.backend.api.dependencies import get_task_store, get_executor
from production.backend.services.task_store import TaskStore
from production.backend.schemas.eda import EDARequest
from production.backend.workers.pipeline_worker import run_stage2

router = APIRouter(prefix="/stage2", tags=["Stage 2 – EDA"])


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
async def analyze(
    body: EDARequest,
    store: TaskStore = Depends(get_task_store),
    executor=Depends(get_executor),
):
    """Enqueue a Stage 2 EDA job and return its job_id."""
    job_id = str(uuid.uuid4())
    await store.create(job_id)

    asyncio.create_task(
        run_stage2(
            job_id=job_id,
            params={"features_path": body.features_path},
            store=store,
            executor=executor,
        )
    )

    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def analyze_status(job_id: str, store: TaskStore = Depends(get_task_store)):
    """Poll the status of an EDA job."""
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return task_response(record)
