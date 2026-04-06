"""Singleton dependencies injected via FastAPI's DI system."""
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from production.backend.services.task_store import TaskStore
from production.backend.config import settings


task_store = TaskStore()


@lru_cache(maxsize=1)
def get_executor() -> ProcessPoolExecutor:
    """Return the process-pool executor singleton, creating it on first call."""
    return ProcessPoolExecutor(max_workers=settings.max_workers)


def get_task_store() -> TaskStore:
    """Return the shared in-process task store."""
    return task_store
