"""SQLAlchemy database models and initialization."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

# Global engine cache
_engines = {}


class ApiKey(Base):
    """API key model for authentication."""

    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    credits = Column(Integer, default=0, nullable=False)
    rate_limit = Column(Integer, default=10, nullable=False)  # requests per minute
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)


class Job(Base):
    """Job model for tracking generation requests."""

    __tablename__ = "jobs"

    id = Column(String(64), primary_key=True)
    key_id = Column(String(36), ForeignKey("api_keys.id"), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # queued, processing, completed, failed, cancelled, expired
    prompt_hash = Column(String(64), nullable=True)
    seed = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    steps = Column(Integer, nullable=True)
    output_format = Column(String(10), nullable=True)
    generation_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    warnings = Column(Text, nullable=True)  # JSON array
    webhook_url = Column(String(2048), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)


class CreditTransaction(Base):
    """Credit transaction history."""

    __tablename__ = "credit_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(36), ForeignKey("api_keys.id"), nullable=False, index=True)
    amount = Column(Integer, nullable=False)  # Positive for additions, negative for deductions
    reason = Column(String(50), nullable=False)  # generation, refund, admin_adjustment
    description = Column(Text, nullable=True)
    balance_after = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class RateLimitBucket(Base):
    """Rate limit tracking using sliding window."""

    __tablename__ = "rate_limit_buckets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(36), ForeignKey("api_keys.id"), nullable=False, index=True)
    window_start = Column(DateTime, nullable=False)
    request_count = Column(Integer, default=0, nullable=False)


class UsageLog(Base):
    """Usage logging for analytics."""

    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(36), ForeignKey("api_keys.id"), nullable=False, index=True)
    status = Column(String(20), nullable=False)  # success, failed
    model = Column(String(50), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    steps = Column(Integer, nullable=True)
    generation_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    credits_used = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


def get_engine(db_path: str):
    """Get or create a database engine for the given path."""
    if db_path not in _engines:
        # Ensure directory exists
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        _engines[db_path] = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            pool_pre_ping=True,
        )
    return _engines[db_path]


def init_database(db_path: str) -> tuple:
    """
    Initialize the database and return engine and session factory.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Tuple of (engine, Session factory)
    """
    engine = get_engine(db_path)

    # Create all tables
    Base.metadata.create_all(engine)

    # Create session factory
    Session = sessionmaker(bind=engine)

    return engine, Session
