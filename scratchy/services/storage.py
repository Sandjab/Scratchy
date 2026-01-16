"""Job result storage service."""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class StorageService:
    """Service for storing and retrieving job results."""

    def __init__(
        self,
        jobs_dir: Path,
        ttl_hours: int = 1,
    ):
        """
        Initialize the storage service.

        Args:
            jobs_dir: Directory for storing job results
            ttl_hours: Hours to keep results before cleanup
        """
        self._jobs_dir = Path(jobs_dir)
        self._ttl_hours = ttl_hours
        self._cleanup_task: Optional[asyncio.Task] = None

        # Ensure directory exists
        self._jobs_dir.mkdir(parents=True, exist_ok=True)

    def _get_job_dir(self, job_id: str) -> Path:
        """Get the directory for a specific job."""
        return self._jobs_dir / job_id

    def store_result(
        self,
        job_id: str,
        image_data: bytes,
        output_format: str,
        metadata: dict,
    ) -> Path:
        """
        Store the result of a completed job.

        Args:
            job_id: The job ID
            image_data: Raw image bytes
            output_format: Image format (png, jpeg, webp)
            metadata: Job metadata

        Returns:
            Path to the stored image
        """
        job_dir = self._get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Store image
        image_path = job_dir / f"result.{output_format}"
        image_path.write_bytes(image_data)

        # Store metadata
        metadata["stored_at"] = datetime.utcnow().isoformat()
        metadata["expires_at"] = (
            datetime.utcnow() + timedelta(hours=self._ttl_hours)
        ).isoformat()

        metadata_path = job_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return image_path

    def get_result(self, job_id: str) -> Optional[tuple[bytes, str, dict]]:
        """
        Retrieve the result of a job.

        Args:
            job_id: The job ID

        Returns:
            Tuple of (image_data, format, metadata) or None if not found/expired
        """
        job_dir = self._get_job_dir(job_id)

        if not job_dir.exists():
            return None

        # Load metadata
        metadata_path = job_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            metadata = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, IOError):
            return None

        # Check expiry on access
        if "expires_at" in metadata:
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            if datetime.utcnow() > expires_at:
                # Clean up expired result
                self.delete_result(job_id)
                return None

        # Find and read image file
        for ext in ["png", "jpeg", "webp"]:
            image_path = job_dir / f"result.{ext}"
            if image_path.exists():
                image_data = image_path.read_bytes()
                return image_data, ext, metadata

        return None

    def result_exists(self, job_id: str) -> bool:
        """Check if a result exists for a job."""
        job_dir = self._get_job_dir(job_id)
        if not job_dir.exists():
            return False

        # Check if result file exists
        for ext in ["png", "jpeg", "webp"]:
            if (job_dir / f"result.{ext}").exists():
                return True

        return False

    def is_expired(self, job_id: str) -> bool:
        """Check if a job result has expired."""
        job_dir = self._get_job_dir(job_id)
        metadata_path = job_dir / "metadata.json"

        if not metadata_path.exists():
            return True

        try:
            metadata = json.loads(metadata_path.read_text())
            if "expires_at" in metadata:
                expires_at = datetime.fromisoformat(metadata["expires_at"])
                return datetime.utcnow() > expires_at
        except (json.JSONDecodeError, IOError):
            return True

        return False

    def delete_result(self, job_id: str) -> bool:
        """
        Delete a job result.

        Args:
            job_id: The job ID

        Returns:
            True if deleted, False if not found
        """
        job_dir = self._get_job_dir(job_id)
        if not job_dir.exists():
            return False

        try:
            shutil.rmtree(job_dir)
            return True
        except IOError as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    def cleanup_expired(self) -> int:
        """
        Clean up all expired job results.

        Returns:
            Number of jobs cleaned up
        """
        cleaned = 0
        now = datetime.utcnow()

        if not self._jobs_dir.exists():
            return 0

        for job_dir in self._jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue

            metadata_path = job_dir / "metadata.json"
            if not metadata_path.exists():
                # No metadata, delete it
                try:
                    shutil.rmtree(job_dir)
                    cleaned += 1
                except IOError:
                    pass
                continue

            try:
                metadata = json.loads(metadata_path.read_text())
                if "expires_at" in metadata:
                    expires_at = datetime.fromisoformat(metadata["expires_at"])
                    if now > expires_at:
                        shutil.rmtree(job_dir)
                        cleaned += 1
            except (json.JSONDecodeError, IOError):
                # Corrupted metadata, delete it
                try:
                    shutil.rmtree(job_dir)
                    cleaned += 1
                except IOError:
                    pass

        return cleaned

    async def start_cleanup_task(self, interval_minutes: int = 5):
        """Start the background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_minutes * 60)
                    cleaned = self.cleanup_expired()
                    if cleaned > 0:
                        logger.info(f"Cleaned up {cleaned} expired job results")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup_task(self):
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        if not self._jobs_dir.exists():
            return {
                "total_jobs": 0,
                "total_size_bytes": 0,
                "jobs_dir": str(self._jobs_dir),
            }

        total_jobs = 0
        total_size = 0

        for job_dir in self._jobs_dir.iterdir():
            if job_dir.is_dir():
                total_jobs += 1
                for f in job_dir.iterdir():
                    if f.is_file():
                        total_size += f.stat().st_size

        return {
            "total_jobs": total_jobs,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "jobs_dir": str(self._jobs_dir),
        }
