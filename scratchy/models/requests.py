"""Pydantic request models for API endpoints."""

from typing import Optional, Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for image generation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text description of the image to generate",
    )
    negative_prompt: Optional[str] = Field(
        None,
        max_length=500,
        description="What to avoid in the generation (SDXL only)",
    )
    width: int = Field(
        1024,
        ge=256,
        le=2048,
        description="Image width (will be rounded to multiple of 64)",
    )
    height: int = Field(
        1024,
        ge=256,
        le=2048,
        description="Image height (will be rounded to multiple of 64)",
    )
    steps: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Number of inference steps (uses model default if not specified)",
    )
    guidance_scale: Optional[float] = Field(
        None,
        ge=0.0,
        le=20.0,
        description="Classifier-free guidance scale (uses model default if not specified)",
    )
    seed: Optional[int] = Field(
        None,
        ge=0,
        description="Random seed for reproducibility",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        "png",
        description="Output image format",
    )
    webhook_url: Optional[str] = Field(
        None,
        max_length=2048,
        description="URL to receive completion webhook",
    )


class ApiKeyCreateRequest(BaseModel):
    """Request model for creating an API key."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for the key",
    )
    credits: int = Field(
        0,
        ge=0,
        description="Initial credit balance",
    )
    rate_limit: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Rate limit in requests per minute (uses default if not specified)",
    )


class ApiKeyUpdateRequest(BaseModel):
    """Request model for updating an API key."""

    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="New name for the key",
    )
    credits: Optional[int] = Field(
        None,
        ge=0,
        description="New credit balance",
    )
    rate_limit: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="New rate limit",
    )
    is_active: Optional[bool] = Field(
        None,
        description="Whether the key is active",
    )
