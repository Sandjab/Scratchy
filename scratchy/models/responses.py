"""Pydantic response models for API endpoints."""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details response."""

    type: str = Field(..., description="URI reference identifying the problem type")
    title: str = Field(..., description="Short human-readable summary")
    status: int = Field(..., description="HTTP status code")
    detail: str = Field(..., description="Human-readable explanation")
    instance: Optional[str] = Field(None, description="URI reference to this occurrence")


class GenerateResponse(BaseModel):
    """Response model for image generation."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: queued, processing, completed, failed")
    image: Optional[str] = Field(None, description="Base64-encoded image data")
    seed: Optional[int] = Field(None, description="Seed used for generation")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    warnings: List[str] = Field(default_factory=list, description="Any warnings during generation")
    credits_used: int = Field(1, description="Credits consumed")
    credits_remaining: int = Field(..., description="Remaining credit balance")


class BalanceResponse(BaseModel):
    """Response model for account balance."""

    credits: int = Field(..., description="Current credit balance")
    rate_limit: int = Field(..., description="Rate limit in requests per minute")
    requests_remaining: int = Field(..., description="Requests remaining in current window")


class JobResponse(BaseModel):
    """Response model for job status/retrieval."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    image: Optional[str] = Field(None, description="Base64-encoded image data (if completed)")
    seed: Optional[int] = Field(None, description="Seed used for generation")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")
    error_message: Optional[str] = Field(None, description="Error message (if failed)")
    created_at: Optional[datetime] = Field(None, description="Job creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Result expiration timestamp")


class HealthLiveResponse(BaseModel):
    """Response model for liveness probe."""

    status: str = Field(..., description="Server status: ok")
    timestamp: datetime = Field(..., description="Current server time")


class HealthReadyResponse(BaseModel):
    """Response model for readiness probe."""

    status: str = Field(..., description="Ready status: ready or not_ready")
    model: str = Field(..., description="Configured model name")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    database: str = Field(..., description="Database status: connected or error")
    queue_depth: int = Field(..., description="Current queue depth")
    queue_capacity: int = Field(..., description="Maximum queue capacity")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_memory_free_gb: Optional[float] = Field(None, description="Free GPU memory in GB")


class ApiKeyResponse(BaseModel):
    """Response model for API key details."""

    id: str = Field(..., description="Unique key identifier")
    name: str = Field(..., description="Human-readable name")
    credits: int = Field(..., description="Current credit balance")
    rate_limit: int = Field(..., description="Rate limit in requests per minute")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="Whether key is active")


class ApiKeyCreateResponse(BaseModel):
    """Response model for API key creation (includes plaintext key)."""

    id: str = Field(..., description="Unique key identifier")
    key: str = Field(..., description="Plaintext API key (shown only once!)")
    name: str = Field(..., description="Human-readable name")
    credits: int = Field(..., description="Initial credit balance")
    rate_limit: int = Field(..., description="Rate limit in requests per minute")
    created_at: datetime = Field(..., description="Creation timestamp")


class ApiKeyListResponse(BaseModel):
    """Response model for listing API keys."""

    keys: List[ApiKeyResponse] = Field(..., description="List of API keys")
    total: int = Field(..., description="Total number of keys")


class AnalyticsResponse(BaseModel):
    """Analytics data for a single API key."""

    key_id: str = Field(..., description="API key ID")
    name: str = Field(..., description="Key name")
    total_requests: int = Field(..., description="Total requests made")
    successful_generations: int = Field(..., description="Successful generations")
    failed_generations: int = Field(..., description="Failed generations")
    total_credits_used: int = Field(..., description="Total credits consumed")
    total_credits_refunded: int = Field(0, description="Total credits refunded")
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")


class AnalyticsExportResponse(BaseModel):
    """Response model for analytics export."""

    data: List[AnalyticsResponse] = Field(..., description="Analytics data")
    exported_at: datetime = Field(..., description="Export timestamp")
