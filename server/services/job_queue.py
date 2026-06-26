"""Job queue service for async video generation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    """Represents a video generation job."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = Field(default=JobStatus.PENDING)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = None
    output_path: Path | None = None

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class JobQueue:
    """In-memory job queue for video generation tasks."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._active_count = 0

    @property
    def active_count(self) -> int:
        """Number of currently active jobs."""
        return self._active_count

    @property
    def pending_count(self) -> int:
        """Number of pending jobs."""
        return sum(
            1 for job in self._jobs.values() 
            if job.status == JobStatus.PENDING
        )

    def create_job(
        self,
        callback_url: str | None = None,
        **metadata: Any,
    ) -> Job:
        """Create a new job and add it to the queue."""
        job = Job(callback_url=callback_url, metadata=metadata)
        self._jobs[job.job_id] = job
        logger.info(f"Job created: {job.job_id}")
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self, 
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())

        if status is not None:
            jobs = [j for j in jobs if j.status == status]

        # Return most recent first
        jobs.reverse()
        return jobs[offset:offset + limit]

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: float | None = None,
        error: str | None = None,
        result_url: str | None = None,
        output_path: Path | None = None,
        **metadata: Any,
    ) -> Job | None:
        """Update job status and metadata."""
        job = self._jobs.get(job_id)
        if job is None:
            return None

        if status is not None:
            job.status = status
            if status == JobStatus.PROCESSING:
                job.started_at = datetime.now()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.now()

        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        if result_url is not None:
            job.result_url = result_url
        if output_path is not None:
            job.output_path = output_path

        job.metadata.update(metadata)

        logger.info(f"Job {job_id} updated: status={job.status}, progress={job.progress}")
        return job

    async def acquire(self) -> bool:
        """Acquire a slot for processing. Returns True if acquired."""
        if self._semaphore.locked():
            return False
        await self._semaphore.acquire()
        self._active_count += 1
        return True

    def release(self) -> None:
        """Release a processing slot."""
        self._semaphore.release()
        self._active_count = max(0, self._active_count - 1)

    def cleanup_old_jobs(self, max_age_hours: float = 24.0) -> int:
        """Remove old completed/failed jobs."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            job_id for job_id, job in self._jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            and job.created_at < cutoff
        ]

        for job_id in to_remove:
            del self._jobs[job_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")

        return len(to_remove)

    async def process_job(
        self,
        job: Job,
        process_fn: Callable[..., dict],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Process a job with the given function."""
        try:
            self.update_job(job.job_id, status=JobStatus.PROCESSING)

            # Run the processing function in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: process_fn(*args, **kwargs)
            )

            self.update_job(
                job.job_id,
                status=JobStatus.COMPLETED,
                progress=100.0,
                metadata=result,
            )

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            self.update_job(
                job.job_id,
                status=JobStatus.FAILED,
                error=str(e),
            )

        finally:
            self.release()
