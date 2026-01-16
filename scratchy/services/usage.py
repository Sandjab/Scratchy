"""Usage logging service."""

import hashlib
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from scratchy.models.database import UsageLog


class UsageLogger:
    """Service for logging API usage."""

    def __init__(self, session_factory):
        """
        Initialize the usage logger.

        Args:
            session_factory: SQLAlchemy session factory
        """
        self._session_factory = session_factory

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self._session_factory()

    def log_success(
        self,
        key_id: str,
        generation_time: float,
        model: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        prompt_hash: Optional[str] = None,
    ) -> None:
        """Log a successful generation."""
        with self._get_session() as session:
            log = UsageLog(
                key_id=key_id,
                timestamp=datetime.utcnow(),
                credits_used=1,
                status="success",
                generation_time=generation_time,
                model=model,
                width=width,
                height=height,
                steps=steps,
                prompt_hash=prompt_hash,
            )
            session.add(log)
            session.commit()

    def log_failure(
        self,
        key_id: str,
        error: str,
        model: Optional[str] = None,
    ) -> None:
        """Log a failed generation."""
        with self._get_session() as session:
            log = UsageLog(
                key_id=key_id,
                timestamp=datetime.utcnow(),
                credits_used=0,
                status="failed",
                model=model,
            )
            session.add(log)
            session.commit()

    def log_refund(
        self,
        key_id: str,
    ) -> None:
        """Log a refund."""
        with self._get_session() as session:
            log = UsageLog(
                key_id=key_id,
                timestamp=datetime.utcnow(),
                credits_used=0,
                status="refunded",
            )
            session.add(log)
            session.commit()

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Hash a prompt for privacy-preserving logging."""
        return hashlib.sha256(prompt.encode()).hexdigest()
