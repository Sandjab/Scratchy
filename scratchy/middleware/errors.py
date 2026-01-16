"""Error handling middleware with RFC 7807 Problem Details."""

from typing import Any, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from scratchy.models.responses import ProblemDetail


# Base URL for problem type URIs
PROBLEM_TYPE_BASE = "https://scratchy.api/errors"


class ProblemDetailException(HTTPException):
    """Exception that renders as RFC 7807 Problem Details."""

    def __init__(
        self,
        status_code: int,
        problem_type: str,
        title: str,
        detail: str,
        instance: Optional[str] = None,
        headers: Optional[dict] = None,
        extra: Optional[dict] = None,
    ):
        """
        Create a Problem Details exception.

        Args:
            status_code: HTTP status code
            problem_type: Problem type identifier (appended to base URL)
            title: Short human-readable title
            detail: Detailed explanation
            instance: URI identifying this specific occurrence
            headers: Additional response headers
            extra: Additional fields to include in the response
        """
        self.problem_type = f"{PROBLEM_TYPE_BASE}/{problem_type}"
        self.title = title
        self.detail_message = detail
        self.instance = instance
        self.extra = extra or {}

        super().__init__(status_code=status_code, detail=detail, headers=headers)


# Common problem types
class InsufficientCreditsError(ProblemDetailException):
    """Raised when user doesn't have enough credits."""

    def __init__(self, credits_available: int, credits_required: int = 1):
        super().__init__(
            status_code=402,
            problem_type="insufficient-credits",
            title="Insufficient Credits",
            detail=f"Your account has {credits_available} credits. This request requires {credits_required} credit(s).",
            extra={
                "credits_available": credits_available,
                "credits_required": credits_required,
            },
        )


class QueueFullError(ProblemDetailException):
    """Raised when the generation queue is at capacity."""

    def __init__(self, queue_depth: int, queue_capacity: int):
        super().__init__(
            status_code=503,
            problem_type="queue-full",
            title="Queue At Capacity",
            detail=f"The generation queue is full ({queue_depth}/{queue_capacity}). Please try again later.",
            headers={"Retry-After": "10"},
            extra={
                "queue_depth": queue_depth,
                "queue_capacity": queue_capacity,
            },
        )


class ModelNotLoadedError(ProblemDetailException):
    """Raised when the model is not loaded."""

    def __init__(self):
        super().__init__(
            status_code=503,
            problem_type="model-not-loaded",
            title="Model Not Ready",
            detail="The image generation model is still loading. Please try again in a few moments.",
            headers={"Retry-After": "30"},
        )


class JobNotFoundError(ProblemDetailException):
    """Raised when a job is not found."""

    def __init__(self, job_id: str):
        super().__init__(
            status_code=404,
            problem_type="job-not-found",
            title="Job Not Found",
            detail=f"No job found with ID: {job_id}",
            extra={"job_id": job_id},
        )


class JobExpiredError(ProblemDetailException):
    """Raised when a job has expired."""

    def __init__(self, job_id: str):
        super().__init__(
            status_code=410,
            problem_type="job-expired",
            title="Job Expired",
            detail=f"Job {job_id} has expired and its results are no longer available.",
            extra={"job_id": job_id},
        )


class GenerationFailedError(ProblemDetailException):
    """Raised when image generation fails."""

    def __init__(self, message: str):
        super().__init__(
            status_code=500,
            problem_type="generation-failed",
            title="Generation Failed",
            detail=message,
        )


async def problem_detail_exception_handler(
    request: Request,
    exc: ProblemDetailException,
) -> JSONResponse:
    """Handle ProblemDetailException and return RFC 7807 response."""
    content = {
        "type": exc.problem_type,
        "title": exc.title,
        "status": exc.status_code,
        "detail": exc.detail_message,
    }

    if exc.instance:
        content["instance"] = exc.instance
    else:
        content["instance"] = str(request.url.path)

    # Add any extra fields
    content.update(exc.extra)

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=exc.headers,
        media_type="application/problem+json",
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Convert standard HTTPException to RFC 7807 format."""
    # Map common status codes to problem types
    type_map = {
        400: "bad-request",
        401: "unauthorized",
        403: "forbidden",
        404: "not-found",
        405: "method-not-allowed",
        409: "conflict",
        422: "validation-error",
        429: "rate-limit-exceeded",
        500: "internal-error",
        502: "bad-gateway",
        503: "service-unavailable",
        504: "gateway-timeout",
    }

    title_map = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        422: "Validation Error",
        429: "Rate Limit Exceeded",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }

    problem_type = type_map.get(exc.status_code, "error")
    title = title_map.get(exc.status_code, "Error")

    content = {
        "type": f"{PROBLEM_TYPE_BASE}/{problem_type}",
        "title": title,
        "status": exc.status_code,
        "detail": str(exc.detail),
        "instance": str(request.url.path),
    }

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=getattr(exc, "headers", None),
        media_type="application/problem+json",
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors as RFC 7807."""
    errors = []
    for error in exc.errors():
        loc = ".".join(str(l) for l in error["loc"])
        errors.append({
            "field": loc,
            "message": error["msg"],
            "type": error["type"],
        })

    content = {
        "type": f"{PROBLEM_TYPE_BASE}/validation-error",
        "title": "Validation Error",
        "status": 422,
        "detail": "One or more fields failed validation.",
        "instance": str(request.url.path),
        "errors": errors,
    }

    return JSONResponse(
        status_code=422,
        content=content,
        media_type="application/problem+json",
    )
