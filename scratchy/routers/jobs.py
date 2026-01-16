"""Job retrieval and management endpoints."""

import base64
from typing import Annotated

from fastapi import APIRouter, Request, Depends, Response

from scratchy.models.responses import JobResponse
from scratchy.models.database import ApiKey
from scratchy.middleware.auth import get_current_key
from scratchy.middleware.errors import JobNotFoundError, JobExpiredError

router = APIRouter(prefix="/v1/jobs", tags=["Jobs"])


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job result",
    description="Retrieve the result of a completed generation job.",
)
async def get_job(
    request: Request,
    job_id: str,
    api_key: Annotated[ApiKey, Depends(get_current_key)],
):
    """Get a job result by ID."""
    storage = request.app.state.storage
    queue = request.app.state.queue

    # Check if result exists in storage
    result = storage.get_result(job_id)

    if result is None:
        # Check if it was expired
        if storage.is_expired(job_id):
            raise JobExpiredError(job_id)

        # Check queue status
        status = queue.get_job_status(job_id)
        if status:
            return JobResponse(
                job_id=job_id,
                status=status["status"],
                seed=status.get("seed"),
                generation_time=status.get("generation_time"),
                warnings=status.get("warnings", []),
                created_at=status["created_at"],
                expires_at=status.get("expires_at"),
            )

        raise JobNotFoundError(job_id)

    image_data, output_format, metadata = result

    return JobResponse(
        job_id=job_id,
        status="completed",
        image=base64.b64encode(image_data).decode(),
        seed=metadata.get("seed"),
        generation_time=metadata.get("generation_time"),
        warnings=metadata.get("warnings", []),
        created_at=metadata.get("created_at"),
        expires_at=metadata.get("expires_at"),
    )


@router.get(
    "/{job_id}/raw",
    summary="Get job result (raw binary)",
    description="Retrieve the raw image binary of a completed job.",
    response_class=Response,
)
async def get_job_raw(
    request: Request,
    job_id: str,
    api_key: Annotated[ApiKey, Depends(get_current_key)],
):
    """Get a job result as raw binary."""
    storage = request.app.state.storage

    result = storage.get_result(job_id)

    if result is None:
        if storage.is_expired(job_id):
            raise JobExpiredError(job_id)
        raise JobNotFoundError(job_id)

    image_data, output_format, metadata = result

    content_type = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(output_format, "image/png")

    return Response(
        content=image_data,
        media_type=content_type,
        headers={
            "X-Scratchy-Job-Id": job_id,
            "X-Scratchy-Seed": str(metadata.get("seed", "")),
        },
    )


@router.delete(
    "/{job_id}",
    summary="Cancel or delete job",
    description="Cancel a queued/processing job or delete a completed job's result.",
)
async def cancel_job(
    request: Request,
    job_id: str,
    api_key: Annotated[ApiKey, Depends(get_current_key)],
):
    """Cancel a job or delete its result."""
    queue = request.app.state.queue
    storage = request.app.state.storage
    credits = request.app.state.credits

    # Try to cancel in queue first
    cancelled = await queue.cancel_job(job_id)

    if cancelled:
        # Refund credit for cancelled job
        credits.refund(
            api_key.id,
            amount=1,
            reason="refund",
            description=f"Job {job_id} cancelled",
        )
        return {"status": "cancelled", "job_id": job_id, "credit_refunded": True}

    # Check if it's a completed job in storage
    if storage.result_exists(job_id):
        storage.delete_result(job_id)
        return {"status": "deleted", "job_id": job_id}

    # Check if expired
    status = queue.get_job_status(job_id)
    if status and status["status"] in ["completed", "failed", "cancelled"]:
        return {"status": status["status"], "job_id": job_id, "message": "Job already finalized"}

    raise JobNotFoundError(job_id)
