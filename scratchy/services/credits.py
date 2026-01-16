"""Credit management service."""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from scratchy.models.database import ApiKey, CreditTransaction


class CreditService:
    """Service for managing API key credits."""

    def __init__(self, session_factory):
        """
        Initialize the credit service.

        Args:
            session_factory: SQLAlchemy session factory
        """
        self._session_factory = session_factory

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self._session_factory()

    def get_balance(self, key_id: str) -> Optional[int]:
        """
        Get the credit balance for an API key.

        Args:
            key_id: The API key ID

        Returns:
            Credit balance or None if key not found
        """
        with self._get_session() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).first()
            return api_key.credits if api_key else None

    def has_credits(self, key_id: str, amount: int = 1) -> bool:
        """
        Check if an API key has sufficient credits.

        Args:
            key_id: The API key ID
            amount: Required credit amount

        Returns:
            True if sufficient credits available
        """
        balance = self.get_balance(key_id)
        return balance is not None and balance >= amount

    def deduct(
        self,
        key_id: str,
        amount: int = 1,
        reason: str = "generation",
        description: Optional[str] = None,
    ) -> tuple[bool, int]:
        """
        Deduct credits from an API key.

        Args:
            key_id: The API key ID
            amount: Credits to deduct
            reason: Reason for deduction
            description: Additional details

        Returns:
            Tuple of (success, new_balance)
        """
        with self._get_session() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).with_for_update().first()

            if not api_key or api_key.credits < amount:
                return False, api_key.credits if api_key else 0

            api_key.credits -= amount
            new_balance = api_key.credits

            # Record transaction
            transaction = CreditTransaction(
                key_id=key_id,
                amount=-amount,
                reason=reason,
                description=description,
                balance_after=new_balance,
                timestamp=datetime.utcnow(),
            )
            session.add(transaction)
            session.commit()

            return True, new_balance

    def refund(
        self,
        key_id: str,
        amount: int = 1,
        reason: str = "refund",
        description: Optional[str] = None,
    ) -> tuple[bool, int]:
        """
        Refund credits to an API key.

        Args:
            key_id: The API key ID
            amount: Credits to refund
            reason: Reason for refund
            description: Additional details

        Returns:
            Tuple of (success, new_balance)
        """
        with self._get_session() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).with_for_update().first()

            if not api_key:
                return False, 0

            api_key.credits += amount
            new_balance = api_key.credits

            # Record transaction
            transaction = CreditTransaction(
                key_id=key_id,
                amount=amount,
                reason=reason,
                description=description,
                balance_after=new_balance,
                timestamp=datetime.utcnow(),
            )
            session.add(transaction)
            session.commit()

            return True, new_balance

    def add_credits(
        self,
        key_id: str,
        amount: int,
        reason: str = "admin_adjustment",
        description: Optional[str] = None,
    ) -> tuple[bool, int]:
        """
        Add credits to an API key (admin operation).

        Args:
            key_id: The API key ID
            amount: Credits to add
            reason: Reason for addition
            description: Additional details

        Returns:
            Tuple of (success, new_balance)
        """
        return self.refund(key_id, amount, reason, description)

    def set_balance(
        self,
        key_id: str,
        new_balance: int,
        reason: str = "admin_adjustment",
        description: Optional[str] = None,
    ) -> tuple[bool, int]:
        """
        Set the credit balance for an API key to a specific value.

        Args:
            key_id: The API key ID
            new_balance: New credit balance
            reason: Reason for change
            description: Additional details

        Returns:
            Tuple of (success, new_balance)
        """
        with self._get_session() as session:
            api_key = session.query(ApiKey).filter(ApiKey.id == key_id).with_for_update().first()

            if not api_key:
                return False, 0

            old_balance = api_key.credits
            delta = new_balance - old_balance

            api_key.credits = new_balance

            # Record transaction
            transaction = CreditTransaction(
                key_id=key_id,
                amount=delta,
                reason=reason,
                description=description or f"Balance set from {old_balance} to {new_balance}",
                balance_after=new_balance,
                timestamp=datetime.utcnow(),
            )
            session.add(transaction)
            session.commit()

            return True, new_balance

    def get_transaction_history(
        self,
        key_id: str,
        limit: int = 100,
    ) -> list[CreditTransaction]:
        """
        Get credit transaction history for an API key.

        Args:
            key_id: The API key ID
            limit: Maximum number of transactions to return

        Returns:
            List of CreditTransaction records
        """
        with self._get_session() as session:
            transactions = (
                session.query(CreditTransaction)
                .filter(CreditTransaction.key_id == key_id)
                .order_by(CreditTransaction.timestamp.desc())
                .limit(limit)
                .all()
            )
            for t in transactions:
                session.expunge(t)
            return transactions
