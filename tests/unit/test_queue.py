"""Unit tests for queue service."""

import pytest
import asyncio
import tempfile
import os

from scratchy.models.database import init_database
from scratchy.services.queue import QueueService, QueueFullError, JobStatus


@pytest.fixture
def db_session():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    engine, Session = init_database(db_path)
    yield Session

    os.unlink(db_path)


@pytest.fixture
def queue_service(db_session):
    """Create queue service with test database."""
    return QueueService(db_session, max_depth=3, job_ttl_hours=1)


class TestQueueService:
    """Tests for QueueService."""

    @pytest.mark.asyncio
    async def test_enqueue_job(self, queue_service):
        """Test enqueueing a job."""
        job_id, position = await queue_service.enqueue(
            key_id="test-key",
            request_data={"prompt": "test prompt"},
        )

        assert job_id.startswith("job_")
        assert position == 1

    @pytest.mark.asyncio
    async def test_queue_depth(self, queue_service):
        """Test queue depth tracking."""
        assert queue_service.depth == 0

        await queue_service.enqueue("key1", {"prompt": "test"})
        assert queue_service.depth == 1

        await queue_service.enqueue("key2", {"prompt": "test"})
        assert queue_service.depth == 2

    @pytest.mark.asyncio
    async def test_queue_capacity(self, queue_service):
        """Test queue capacity."""
        assert queue_service.capacity == 3
        assert queue_service.is_full is False

        await queue_service.enqueue("key1", {"prompt": "test"})
        await queue_service.enqueue("key2", {"prompt": "test"})
        await queue_service.enqueue("key3", {"prompt": "test"})

        assert queue_service.is_full is True

    @pytest.mark.asyncio
    async def test_queue_full_error(self, queue_service):
        """Test that QueueFullError is raised when queue is full."""
        await queue_service.enqueue("key1", {"prompt": "test"})
        await queue_service.enqueue("key2", {"prompt": "test"})
        await queue_service.enqueue("key3", {"prompt": "test"})

        with pytest.raises(QueueFullError):
            await queue_service.enqueue("key4", {"prompt": "test"})

    @pytest.mark.asyncio
    async def test_dequeue_job(self, queue_service):
        """Test dequeueing a job."""
        job_id, _ = await queue_service.enqueue("key1", {"prompt": "test"})

        job = await queue_service.dequeue()

        assert job is not None
        assert job.job_id == job_id
        assert job.key_id == "key1"

    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self, queue_service):
        """Test dequeueing from empty queue."""
        job = await queue_service.dequeue()
        assert job is None

    @pytest.mark.asyncio
    async def test_complete_job(self, queue_service):
        """Test completing a job."""
        job_id, _ = await queue_service.enqueue("key1", {"prompt": "test"})
        await queue_service.dequeue()

        await queue_service.complete_job(
            job_id=job_id,
            seed=12345,
            generation_time=2.5,
            warnings=["test warning"],
        )

        status = queue_service.get_job_status(job_id)
        assert status["status"] == JobStatus.COMPLETED.value
        assert status["seed"] == 12345
        assert status["generation_time"] == 2.5

    @pytest.mark.asyncio
    async def test_fail_job(self, queue_service):
        """Test failing a job."""
        job_id, _ = await queue_service.enqueue("key1", {"prompt": "test"})
        await queue_service.dequeue()

        await queue_service.fail_job(job_id=job_id, error_message="Test error")

        status = queue_service.get_job_status(job_id)
        assert status["status"] == JobStatus.FAILED.value
        assert status["error_message"] == "Test error"

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, queue_service):
        """Test cancelling a queued job."""
        job_id, _ = await queue_service.enqueue("key1", {"prompt": "test"})

        cancelled = await queue_service.cancel_job(job_id)

        assert cancelled is True
        status = queue_service.get_job_status(job_id)
        assert status["status"] == JobStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, queue_service):
        """Test cancelling a non-existent job."""
        cancelled = await queue_service.cancel_job("non-existent-job")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_get_queue_position(self, queue_service):
        """Test getting queue position."""
        job_id1, _ = await queue_service.enqueue("key1", {"prompt": "test1"})
        job_id2, _ = await queue_service.enqueue("key2", {"prompt": "test2"})
        job_id3, _ = await queue_service.enqueue("key3", {"prompt": "test3"})

        assert queue_service.get_queue_position(job_id1) == 1
        assert queue_service.get_queue_position(job_id2) == 2
        assert queue_service.get_queue_position(job_id3) == 3

    @pytest.mark.asyncio
    async def test_is_job_cancelled(self, queue_service):
        """Test checking if job is cancelled."""
        job_id, _ = await queue_service.enqueue("key1", {"prompt": "test"})

        assert queue_service.is_job_cancelled(job_id) is False

        await queue_service.cancel_job(job_id)

        assert queue_service.is_job_cancelled(job_id) is True
