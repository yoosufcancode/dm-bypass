import asyncio
from typing import Any, Optional
from enum import Enum


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class TaskRecord:
    def __init__(self, job_id: str):
        """Initialise a task record in the pending state."""
        self.job_id = job_id
        self.status: TaskStatus = TaskStatus.pending
        self.progress: int = 0  # integer percentage in [0, 100]
        self.message: str = "Queued"
        self.result: Optional[Any] = None
        self.error: Optional[str] = None


class TaskStore:
    def __init__(self):
        """In-memory store for background task records, safe for async access."""
        self.tasks: dict[str, TaskRecord] = {}
        self.lock = asyncio.Lock()

    async def create(self, job_id: str) -> TaskRecord:
        """Create and register a new TaskRecord, returning it."""
        async with self.lock:
            record = TaskRecord(job_id)
            self.tasks[job_id] = record
            return record

    async def get(self, job_id: str) -> Optional[TaskRecord]:
        """Return the TaskRecord for job_id, or None if not found."""
        async with self.lock:
            return self.tasks.get(job_id)

    async def update(
        self,
        job_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update any subset of a TaskRecord's fields atomically."""
        async with self.lock:
            record = self.tasks.get(job_id)
            if record is None:
                return
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = progress
            if message is not None:
                record.message = message
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error

    def update_sync(
        self,
        job_id: str,
        loop: asyncio.AbstractEventLoop,
        **kwargs,
    ) -> None:
        """Thread-safe update from worker threads."""
        future = asyncio.run_coroutine_threadsafe(self.update(job_id, **kwargs), loop)
        future.result(timeout=5)
