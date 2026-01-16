"""Unit tests for authentication service."""

import pytest
import tempfile
import os
from pathlib import Path

from scratchy.models.database import init_database
from scratchy.services.auth import AuthService


@pytest.fixture
def db_session():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    engine, Session = init_database(db_path)
    yield Session

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def auth_service(db_session):
    """Create auth service with test database."""
    return AuthService(db_session, default_rate_limit=10)


class TestAuthService:
    """Tests for AuthService."""

    def test_create_key(self, auth_service):
        """Test creating a new API key."""
        plaintext_key, api_key = auth_service.create_key(
            name="test_key",
            credits=100,
            rate_limit=20,
        )

        assert plaintext_key.startswith("sk_")
        assert len(plaintext_key) == 35  # "sk_" + 32 chars
        assert api_key.name == "test_key"
        assert api_key.credits == 100
        assert api_key.rate_limit == 20
        assert api_key.is_active is True

    def test_validate_key_success(self, auth_service):
        """Test validating a valid API key."""
        plaintext_key, _ = auth_service.create_key(name="test_key", credits=50)

        validated = auth_service.validate_key(plaintext_key)

        assert validated is not None
        assert validated.name == "test_key"
        assert validated.credits == 50

    def test_validate_key_invalid(self, auth_service):
        """Test validating an invalid API key."""
        result = auth_service.validate_key("sk_invalid_key_12345678901234567")
        assert result is None

    def test_validate_key_wrong_prefix(self, auth_service):
        """Test validating a key with wrong prefix."""
        result = auth_service.validate_key("invalid_prefix_key")
        assert result is None

    def test_validate_key_empty(self, auth_service):
        """Test validating empty key."""
        result = auth_service.validate_key("")
        assert result is None

        result = auth_service.validate_key(None)
        assert result is None

    def test_validate_inactive_key(self, auth_service):
        """Test that inactive keys are rejected."""
        plaintext_key, api_key = auth_service.create_key(name="test_key")

        # Deactivate the key
        auth_service.delete_key(api_key.id)

        result = auth_service.validate_key(plaintext_key)
        assert result is None

    def test_list_keys(self, auth_service):
        """Test listing API keys."""
        auth_service.create_key(name="key1")
        auth_service.create_key(name="key2")
        auth_service.create_key(name="key3")

        keys = auth_service.list_keys()
        assert len(keys) == 3

    def test_list_keys_exclude_inactive(self, auth_service):
        """Test that inactive keys are excluded by default."""
        _, key1 = auth_service.create_key(name="key1")
        auth_service.create_key(name="key2")

        auth_service.delete_key(key1.id)

        keys = auth_service.list_keys(include_inactive=False)
        assert len(keys) == 1

        keys = auth_service.list_keys(include_inactive=True)
        assert len(keys) == 2

    def test_update_key(self, auth_service):
        """Test updating an API key."""
        _, api_key = auth_service.create_key(name="original", credits=10)

        updated = auth_service.update_key(
            key_id=api_key.id,
            name="updated",
            credits=50,
            rate_limit=30,
        )

        assert updated.name == "updated"
        assert updated.credits == 50
        assert updated.rate_limit == 30

    def test_update_key_not_found(self, auth_service):
        """Test updating a non-existent key."""
        result = auth_service.update_key(
            key_id="non-existent-id",
            name="test",
        )
        assert result is None

    def test_rate_limit_check_allowed(self, auth_service):
        """Test rate limit check when within limits."""
        _, api_key = auth_service.create_key(name="test", rate_limit=10)

        is_allowed, remaining = auth_service.check_rate_limit(api_key.id, 10)

        assert is_allowed is True
        assert remaining == 9

    def test_rate_limit_check_exceeded(self, auth_service):
        """Test rate limit check when limit exceeded."""
        _, api_key = auth_service.create_key(name="test", rate_limit=2)

        # Use up the limit
        auth_service.check_rate_limit(api_key.id, 2)
        auth_service.check_rate_limit(api_key.id, 2)

        # Should be blocked
        is_allowed, remaining = auth_service.check_rate_limit(api_key.id, 2)

        assert is_allowed is False
        assert remaining == 0
