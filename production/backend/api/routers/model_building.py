"""
Stage 4 router — model building.

  POST /stage4/build               trigger model building job
  GET  /stage4/status/{job_id}     poll job status
"""
import uuid
import asyncio

from fastapi import APIRouter, Depends, HTTPException

from production.backend.api.dependencies import get_task_store, get_executor
from production.backend.services.task_store import TaskStore
from production.backend.schemas.model_building import ModelBuildRequest
from production.backend.workers.pipeline_worker import run_stage4

router = APIRouter(prefix="/stage4", tags=["Stage 4 – Model Building"])


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


@router.post("/build")
async def build_models(
    body: ModelBuildRequest,
    store: TaskStore = Depends(get_task_store),
    executor=Depends(get_executor),
):
    """Enqueue a Stage 4 model building job and return its job_id."""
    job_id = str(uuid.uuid4())
    await store.create(job_id)

    asyncio.create_task(
        run_stage4(
            job_id=job_id,
            params={
                "features_path":     body.features_path,
                "selected_features": body.selected_features,
                "target_col":        body.target_col,
                "test_size":         body.test_size,
                "random_state":      body.random_state,
            },
            store=store,
            executor=executor,
        )
    )

    return {"job_id": job_id}


@router.get("/status/{job_id}")
async def build_status(job_id: str, store: TaskStore = Depends(get_task_store)):
    """Poll the status of a model building job."""
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return task_response(record)
