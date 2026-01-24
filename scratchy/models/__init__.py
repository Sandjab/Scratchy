"""Data models for Scratchy."""

from scratchy.models.database import (
    ApiKey,
    Job,
    CreditTransaction,
    RateLimitBucket,
    UsageLog,
    init_database,
    get_engine,
)
from scratchy.models.requests import (
    GenerateRequest,
    ApiKeyCreateRequest,
    ApiKeyUpdateRequest,
)
from scratchy.models.responses import (
    ProblemDetail,
    GenerateResponse,
    BalanceResponse,
    JobResponse,
    HealthLiveResponse,
    HealthReadyResponse,
    ApiKeyResponse,
    ApiKeyCreateResponse,
    ApiKeyListResponse,
    AnalyticsResponse,
    AnalyticsExportResponse,
)

__all__ = [
    # Database models
    "ApiKey",
    "Job",
    "CreditTransaction",
    "RateLimitBucket",
    "UsageLog",
    "init_database",
    "get_engine",
    # Request models
    "GenerateRequest",
    "ApiKeyCreateRequest",
    "ApiKeyUpdateRequest",
    # Response models
    "ProblemDetail",
    "GenerateResponse",
    "BalanceResponse",
    "JobResponse",
    "HealthLiveResponse",
    "HealthReadyResponse",
    "ApiKeyResponse",
    "ApiKeyCreateResponse",
    "ApiKeyListResponse",
    "AnalyticsResponse",
    "AnalyticsExportResponse",
]
