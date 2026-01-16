"""Unit tests for credit service."""

import pytest
import tempfile
import os

from scratchy.models.database import init_database, ApiKey
from scratchy.services.credits import CreditService


@pytest.fixture
def db_session():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    engine, Session = init_database(db_path)

    # Create a test API key
    with Session() as session:
        api_key = ApiKey(
            id="test-key-id",
            key_hash="test_hash",
            name="test_key",
            credits=100,
            rate_limit=10,
        )
        session.add(api_key)
        session.commit()

    yield Session

    os.unlink(db_path)


@pytest.fixture
def credit_service(db_session):
    """Create credit service with test database."""
    return CreditService(db_session)


class TestCreditService:
    """Tests for CreditService."""

    def test_get_balance(self, credit_service):
        """Test getting credit balance."""
        balance = credit_service.get_balance("test-key-id")
        assert balance == 100

    def test_get_balance_not_found(self, credit_service):
        """Test getting balance for non-existent key."""
        balance = credit_service.get_balance("non-existent")
        assert balance is None

    def test_has_credits_true(self, credit_service):
        """Test has_credits when credits available."""
        assert credit_service.has_credits("test-key-id", amount=1) is True
        assert credit_service.has_credits("test-key-id", amount=100) is True

    def test_has_credits_false(self, credit_service):
        """Test has_credits when insufficient credits."""
        assert credit_service.has_credits("test-key-id", amount=101) is False

    def test_has_credits_not_found(self, credit_service):
        """Test has_credits for non-existent key."""
        assert credit_service.has_credits("non-existent", amount=1) is False

    def test_deduct_success(self, credit_service):
        """Test successful credit deduction."""
        success, new_balance = credit_service.deduct(
            "test-key-id",
            amount=10,
            reason="generation",
        )

        assert success is True
        assert new_balance == 90
        assert credit_service.get_balance("test-key-id") == 90

    def test_deduct_insufficient(self, credit_service):
        """Test deduction with insufficient credits."""
        success, balance = credit_service.deduct(
            "test-key-id",
            amount=200,
            reason="generation",
        )

        assert success is False
        assert balance == 100  # Balance unchanged

    def test_deduct_not_found(self, credit_service):
        """Test deduction for non-existent key."""
        success, balance = credit_service.deduct(
            "non-existent",
            amount=10,
            reason="generation",
        )

        assert success is False
        assert balance == 0

    def test_refund(self, credit_service):
        """Test credit refund."""
        # First deduct
        credit_service.deduct("test-key-id", amount=20, reason="generation")

        # Then refund
        success, new_balance = credit_service.refund(
            "test-key-id",
            amount=20,
            reason="refund",
        )

        assert success is True
        assert new_balance == 100

    def test_add_credits(self, credit_service):
        """Test adding credits."""
        success, new_balance = credit_service.add_credits(
            "test-key-id",
            amount=50,
            reason="admin_adjustment",
        )

        assert success is True
        assert new_balance == 150

    def test_set_balance(self, credit_service):
        """Test setting balance to specific value."""
        success, new_balance = credit_service.set_balance(
            "test-key-id",
            new_balance=500,
            reason="admin_adjustment",
        )

        assert success is True
        assert new_balance == 500

    def test_transaction_history(self, credit_service):
        """Test retrieving transaction history."""
        credit_service.deduct("test-key-id", amount=10, reason="generation")
        credit_service.deduct("test-key-id", amount=20, reason="generation")
        credit_service.refund("test-key-id", amount=10, reason="refund")

        history = credit_service.get_transaction_history("test-key-id")

        assert len(history) == 3
        # Most recent first
        assert history[0].amount == 10  # refund
        assert history[1].amount == -20  # deduction
        assert history[2].amount == -10  # deduction
