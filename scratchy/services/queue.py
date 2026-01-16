"""Job queue management service."""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session

from scratchy.models.database import Job


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class QueuedJob:
    """A job in the queue."""
    job_id: str
    key_id: str
    request_data: dict
    created_at: datetime = field(default_factory=datetime.utcnow)
    cancelled: bool = False
    future: Optional[asyncio.Future] = None


class QueueService:
    """Service for managing the generation queue."""

    JOB_PREFIX = "job_"

    def __init__(
        self,
        session_factory,
        max_depth: int = 10,
        job_ttl_hours: int = 1,
    ):
        """
        Initialize the queue service.

        Args:
            session_factory: SQLAlchemy session factory
            max_depth: Maximum queue depth
            job_ttl_hours: Hours to keep completed job results
        """
        self._session_factory = session_factory
        self._max_depth = max_depth
        self._job_ttl_hours = job_ttl_hours

        # In-memory queue for pending jobs
        self._queue: asyncio.Queue[QueuedJob] = asyncio.Queue(maxsize=max_depth)
        self._pending_jobs: dict[str, QueuedJob] = {}
        self._current_job: Optional[QueuedJob] = None
        self._lock = asyncio.Lock()

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self._session_factory()

    @staticmethod
    def _generate_job_id() -> str:
        """Generate a new job ID."""
        return f"{QueueService.JOB_PREFIX}{uuid.uuid4().hex[:16]}"

    @property
    def depth(self) -> int:
        """Current queue depth."""
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        """Maximum queue capacity."""
        return self._max_depth

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()

    async def enqueue(
        self,
        key_id: str,
        request_data: dict,
    ) -> tuple[str, int]:
        """
        Add a job to the queue.

        Args:
            key_id: The API key ID
            request_data: The generation request data

        Returns:
            Tuple of (job_id, position_in_queue)

        Raises:
            QueueFullError: If the queue is at capacity
        """
        if self.is_full:
            raise QueueFullError("Queue is at capacity")

        job_id = self._generate_job_id()
        job = QueuedJob(
            job_id=job_id,
            key_id=key_id,
            request_data=request_data,
        )

        # Create database record
        with self._get_session() as session:
            db_job = Job(
                id=job_id,
                key_id=key_id,
                status=JobStatus.QUEUED.value,
                prompt_hash=request_data.get("prompt_hash"),
                seed=request_data.get("seed"),
                width=request_data.get("width"),
                height=request_data.get("height"),
                steps=request_data.get("steps"),
                output_format=request_data.get("output_format"),
                webhook_url=request_data.get("webhook_url"),
                created_at=job.created_at,
                expires_at=job.created_at + timedelta(hours=self._job_ttl_hours),
            )
            session.add(db_job)
            session.commit()

        async with self._lock:
            await self._queue.put(job)
            self._pending_jobs[job_id] = job
            position = self._queue.qsize()

        return job_id, position

    async def dequeue(self) -> Optional[QueuedJob]:
        """
        Get the next job from the queue.

        Returns:
            The next job or None if queue is empty
        """
        try:
            job = await asyncio.wait_for(self._queue.get(), timeout=0.1)

            async with self._lock:
                self._current_job = job
                if job.job_id in self._pending_jobs:
                    del self._pending_jobs[job.job_id]

            # Update database status
            with self._get_session() as session:
                db_job = session.query(Job).filter(Job.id == job.job_id).first()
                if db_job:
                    db_job.status = JobStatus.PROCESSING.value
                    db_job.started_at = datetime.utcnow()
                    session.commit()

            return job
        except asyncio.TimeoutError:
            return None

    async def complete_job(
        self,
        job_id: str,
        seed: int,
        generation_time: float,
        warnings: list[str] = None,
    ) -> None:
        """Mark a job as completed."""
        async with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                self._current_job = None

        with self._get_session() as session:
            db_job = session.query(Job).filter(Job.id == job_id).first()
            if db_job:
                db_job.status = JobStatus.COMPLETED.value
                db_job.seed = seed
                db_job.generation_time = generation_time
                db_job.completed_at = datetime.utcnow()
                db_job.warnings = json.dumps(warnings or [])
                session.commit()

    async def fail_job(
        self,
        job_id: str,
        error_message: str,
    ) -> None:
        """Mark a job as failed."""
        async with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                self._current_job = None

        with self._get_session() as session:
            db_job = session.query(Job).filter(Job.id == job_id).first()
            if db_job:
                db_job.status = JobStatus.FAILED.value
                db_job.error_message = error_message
                db_job.completed_at = datetime.utcnow()
                session.commit()

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Returns:
            True if job was cancelled, False if not found or already processed
        """
        async with self._lock:
            # Check if in pending queue
            if job_id in self._pending_jobs:
                self._pending_jobs[job_id].cancelled = True

                with self._get_session() as session:
                    db_job = session.query(Job).filter(Job.id == job_id).first()
                    if db_job:
                        db_job.status = JobStatus.CANCELLED.value
                        db_job.completed_at = datetime.utcnow()
                        session.commit()

                return True

            # Check if currently processing
            if self._current_job and self._current_job.job_id == job_id:
                self._current_job.cancelled = True

                with self._get_session() as session:
                    db_job = session.query(Job).filter(Job.id == job_id).first()
                    if db_job:
                        db_job.status = JobStatus.CANCELLED.value
                        db_job.completed_at = datetime.utcnow()
                        session.commit()

                return True

        return False

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get the status of a job."""
        with self._get_session() as session:
            db_job = session.query(Job).filter(Job.id == job_id).first()
            if not db_job:
                return None

            return {
                "job_id": db_job.id,
                "status": db_job.status,
                "seed": db_job.seed,
                "generation_time": db_job.generation_time,
                "warnings": json.loads(db_job.warnings) if db_job.warnings else [],
                "error_message": db_job.error_message,
                "created_at": db_job.created_at,
                "completed_at": db_job.completed_at,
                "expires_at": db_job.expires_at,
            }

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get the position of a job in the queue (1-indexed)."""
        position = 1
        for queued_job in list(self._pending_jobs.values()):
            if queued_job.job_id == job_id:
                return position
            position += 1
        return None

    def is_job_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled."""
        if job_id in self._pending_jobs:
            return self._pending_jobs[job_id].cancelled
        if self._current_job and self._current_job.job_id == job_id:
            return self._current_job.cancelled
        return False

    async def cleanup_expired_jobs(self) -> int:
        """
        Clean up expired jobs from the database.

        Returns:
            Number of jobs cleaned up
        """
        now = datetime.utcnow()
        with self._get_session() as session:
            expired = session.query(Job).filter(
                Job.expires_at < now,
                Job.status.in_([JobStatus.COMPLETED.value, JobStatus.FAILED.value]),
            ).all()

            count = len(expired)
            for job in expired:
                job.status = JobStatus.EXPIRED.value

            session.commit()
            return count


class QueueFullError(Exception):
    """Raised when the queue is at capacity."""
    pass
