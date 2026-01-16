"""Authentication and API key management service."""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from scratchy.models.database import ApiKey, RateLimitBucket


class AuthService:
    """Service for API key authentication and management."""

    KEY_PREFIX = "sk_"
    KEY_LENGTH = 32  # Length of the random part

    def __init__(self, session_factory, default_rate_limit: int = 10):
        """
        Initialize the auth service.

        Args:
            session_factory: SQLAlchemy session factory
            default_rate_limit: Default rate limit in requests per minute
        """
        self._session_factory = session_factory
        self._default_rate_limit = default_rate_limit

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self._session_factory()

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def _generate_key() -> str:
        """Generate a new API key."""
        random_part = secrets.token_urlsafe(AuthService.KEY_LENGTH)[:AuthService.KEY_LENGTH]
        return f"{AuthService.KEY_PREFIX}{random_part}"

    def create_key(
        self,
        name: str,
        credits: int = 0,
        rate_limit: Optional[int] = None,
    ) -> tuple[str, ApiKey]:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            credits: Initial credit balance
            rate_limit: Rate limit (uses default if not specified)

        Returns:
            Tuple of (plaintext key, ApiKey model)
        """
        key = self._generate_key()
        key_hash = self._hash_key(key)

        api_key = ApiKey(
            id=str(uuid.uuid4()),
            key_hash=key_hash,
            name=name,
            credits=credits,
            rate_limit=rate_limit or self._default_rate_limit,
            created_at=datetime.utcnow(),
            is_active=True,
        )

        with self._get_session() as session:
            session.add(api_key)
            session.commit()
            session.refresh(api_key)

        return key, api_key

    def validate_key(self, key: str) -> Optional[ApiKey]:
        """
        Validate an API key and return the associated record.

        Args:
            key: The plaintext API key

        Returns:
            ApiKey if valid and active, None otherwise
        """
        if not key or not key.startswith(self.KEY_PREFIX):
            return None

        key_hash = self._hash_key(key)

        with self._get_session() as session:
            api_key = session.query(ApiKey).filter(
                ApiKey.key_hash == key_hash,
                ApiKey.is_active == True,
            ).first()

            if api_key:
                # Update last used timestamp
                api_key.last_used_at = datetime.utcnow()
                session.commit()
                # Refresh to load all attributes after commit (which expires them)
                session.refresh(api_key)
                # Detach from session to avoid lazy loading issues
                session.expunge(api_key)

            return api_key

    def get_key_by_id(self, key_id: str) -> Optional[ApiKey]:
        """Get an API key by its ID."""
        with self._get_session() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if api_key:
                session.expunge(api_key)
            return api_key

    def list_keys(self, include_inactive: bool = False) -> list[ApiKey]:
        """List all API keys."""
        with self._get_session() as session:
            query = session.query(ApiKey)
            if not include_inactive:
                query = query.filter(ApiKey.is_active == True)
            keys = query.order_by(ApiKey.created_at.desc()).all()
            for key in keys:
                session.expunge(key)
            return keys

    def update_key(
        self,
        key_id: str,
        name: Optional[str] = None,
        credits: Optional[int] = None,
        rate_limit: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[ApiKey]:
        """Update an API key."""
        with self._get_session() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if not api_key:
                return None

            if name is not None:
                api_key.name = name
            if credits is not None:
                api_key.credits = credits
            if rate_limit is not None:
                api_key.rate_limit = rate_limit
            if is_active is not None:
                api_key.is_active = is_active

            session.commit()
            session.refresh(api_key)
            session.expunge(api_key)
            return api_key

    def delete_key(self, key_id: str) -> bool:
        """Delete (deactivate) an API key."""
        with self._get_session() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            if not api_key:
                return False

            api_key.is_active = False
            session.commit()
            return True

    def check_rate_limit(self, key_id: str, rate_limit: int) -> tuple[bool, int]:
        """
        Check if a request is within rate limits using sliding window.

        Args:
            key_id: The API key ID
            rate_limit: Requests per minute limit

        Returns:
            Tuple of (is_allowed, requests_remaining)
        """
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        with self._get_session() as session:
            # Clean up old buckets
            session.query(RateLimitBucket).filter(
                RateLimitBucket.key_id == key_id,
                RateLimitBucket.window_start < window_start,
            ).delete()

            # Count requests in current window
            bucket = session.query(RateLimitBucket).filter(
                RateLimitBucket.key_id == key_id,
                RateLimitBucket.window_start >= window_start,
            ).first()

            current_count = bucket.request_count if bucket else 0
            remaining = max(0, rate_limit - current_count)
            is_allowed = current_count < rate_limit

            if is_allowed:
                # Increment counter
                if bucket:
                    bucket.request_count += 1
                else:
                    bucket = RateLimitBucket(
                        key_id=key_id,
                        window_start=now,
                        request_count=1,
                    )
                    session.add(bucket)

                session.commit()
                remaining = max(0, rate_limit - (current_count + 1))

            return is_allowed, remaining

    def get_requests_remaining(self, key_id: str, rate_limit: int) -> int:
        """Get the number of requests remaining in the current rate limit window."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        with self._get_session() as session:
            bucket = session.query(RateLimitBucket).filter(
                RateLimitBucket.key_id == key_id,
                RateLimitBucket.window_start >= window_start,
            ).first()

            current_count = bucket.request_count if bucket else 0
            return max(0, rate_limit - current_count)
