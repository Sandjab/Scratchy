"""Image generation endpoints."""

import asyncio
import base64
import json
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Request, Depends, Response
from fastapi.responses import StreamingResponse

from scratchy.models.requests import GenerateRequest
from scratchy.models.responses import GenerateResponse
from scratchy.models.database import ApiKey
from scratchy.middleware.auth import get_current_key
from scratchy.middleware.errors import (
    InsufficientCreditsError,
    QueueFullError,
    ModelNotLoadedError,
    GenerationFailedError,
)
from scratchy.services.queue import QueueService, QueueFullError as QueueFullException
from scratchy.services.generator import GeneratorService
from scratchy.utils.sanitize import sanitize_prompt
from scratchy.utils.exif import add_exif_metadata

router = APIRouter(prefix="/v1", tags=["Generation"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate image",
    description="Generate an image from a text prompt. Returns JSON with base64-encoded image.",
)
async def generate(
    request: Request,
    gen_request: GenerateRequest,
    api_key: Annotated[ApiKey, Depends(get_current_key)],
):
    """Generate an image and return as JSON with base64 encoding."""
    app = request.app
    generator: GeneratorService = app.state.generator
    queue: QueueService = app.state.queue
    credits = app.state.credits
    storage = app.state.storage
    settings = app.state.settings

    # Check model loaded
    if not generator.is_loaded:
        raise ModelNotLoadedError()

    # Check credits
    balance = credits.get_balance(api_key.id)
    if balance is None or balance < 1:
        raise InsufficientCreditsError(credits_available=balance or 0)

    # Sanitize prompt
    prompt = sanitize_prompt(
        gen_request.prompt,
        max_length=settings.security.max_prompt_length,
    )
    negative_prompt = None
    if gen_request.negative_prompt:
        negative_prompt = sanitize_prompt(
            gen_request.negative_prompt,
            max_length=settings.security.max_negative_prompt_length,
        )

    # Deduct credit upfront
    success, new_balance = credits.deduct(
        api_key.id,
        amount=1,
        reason="generation",
        description=f"Image generation",
    )

    if not success:
        raise InsufficientCreditsError(credits_available=new_balance)

    try:
        # Generate image
        result = generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=gen_request.width,
            height=gen_request.height,
            steps=gen_request.steps,
            guidance_scale=gen_request.guidance_scale,
            seed=gen_request.seed,
            output_format=gen_request.output_format,
        )

        # Add EXIF metadata
        image_bytes = add_exif_metadata(
            result.image_bytes,
            result.output_format,
            prompt=prompt,
            model_name=settings.model.name,
            seed=result.seed,
            steps=gen_request.steps or settings.model.default_steps,
            width=gen_request.width,
            height=gen_request.height,
            guidance_scale=gen_request.guidance_scale or settings.model.default_guidance_scale,
        )

        # Generate job ID and store result
        job_id = f"job_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{result.seed}"

        storage.store_result(
            job_id=job_id,
            image_data=image_bytes,
            output_format=result.output_format,
            metadata={
                "seed": result.seed,
                "generation_time": result.generation_time,
                "warnings": result.warnings,
                "width": gen_request.width,
                "height": gen_request.height,
            },
        )

        # Log usage
        app.state.usage_logger.log_success(
            key_id=api_key.id,
            generation_time=result.generation_time,
            model=settings.model.name,
            width=gen_request.width,
            height=gen_request.height,
            steps=gen_request.steps or settings.model.default_steps,
        )

        return GenerateResponse(
            job_id=job_id,
            status="completed",
            image=base64.b64encode(image_bytes).decode(),
            seed=result.seed,
            generation_time=result.generation_time,
            warnings=result.warnings,
            credits_used=1,
            credits_remaining=new_balance,
        )

    except Exception as e:
        # Refund credit on failure
        credits.refund(
            api_key.id,
            amount=1,
            reason="refund",
            description=f"Generation failed: {str(e)[:100]}",
        )

        # Log failure
        app.state.usage_logger.log_failure(
            key_id=api_key.id,
            error=str(e),
        )

        raise GenerationFailedError(str(e))


@router.post(
    "/generate/raw",
    summary="Generate image (raw binary)",
    description="Generate an image and return raw binary data.",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}, "image/jpeg": {}, "image/webp": {}},
            "description": "Generated image as binary",
        }
    },
)
async def generate_raw(
    request: Request,
    gen_request: GenerateRequest,
    api_key: Annotated[ApiKey, Depends(get_current_key)],
):
    """Generate an image and return raw binary data with metadata in headers."""
    app = request.app
    generator: GeneratorService = app.state.generator
    credits = app.state.credits
    settings = app.state.settings

    # Check model loaded
    if not generator.is_loaded:
        raise ModelNotLoadedError()

    # Check credits
    balance = credits.get_balance(api_key.id)
    if balance is None or balance < 1:
        raise InsufficientCreditsError(credits_available=balance or 0)

    # Sanitize prompt
    prompt = sanitize_prompt(
        gen_request.prompt,
        max_length=settings.security.max_prompt_length,
    )

    # Deduct credit
    success, new_balance = credits.deduct(api_key.id, amount=1, reason="generation")
    if not success:
        raise InsufficientCreditsError(credits_available=new_balance)

    try:
        result = generator.generate(
            prompt=prompt,
            negative_prompt=gen_request.negative_prompt,
            width=gen_request.width,
            height=gen_request.height,
            steps=gen_request.steps,
            guidance_scale=gen_request.guidance_scale,
            seed=gen_request.seed,
            output_format=gen_request.output_format,
        )

        # Add EXIF metadata
        image_bytes = add_exif_metadata(
            result.image_bytes,
            result.output_format,
            prompt=prompt,
            model_name=settings.model.name,
            seed=result.seed,
            steps=gen_request.steps or settings.model.default_steps,
            width=gen_request.width,
            height=gen_request.height,
            guidance_scale=gen_request.guidance_scale or settings.model.default_guidance_scale,
        )

        content_type = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }.get(gen_request.output_format, "image/png")

        # Log usage
        app.state.usage_logger.log_success(
            key_id=api_key.id,
            generation_time=result.generation_time,
            model=settings.model.name,
        )

        return Response(
            content=image_bytes,
            media_type=content_type,
            headers={
                "X-Scratchy-Job-Id": f"raw_{result.seed}",
                "X-Scratchy-Seed": str(result.seed),
                "X-Scratchy-Generation-Time": f"{result.generation_time:.2f}",
                "X-Scratchy-Warnings": json.dumps(result.warnings) if result.warnings else "",
                "X-Scratchy-Credits-Remaining": str(new_balance),
            },
        )

    except Exception as e:
        credits.refund(api_key.id, amount=1, reason="refund")
        app.state.usage_logger.log_failure(key_id=api_key.id, error=str(e))
        raise GenerationFailedError(str(e))


@router.post(
    "/generate/stream",
    summary="Generate image with SSE progress",
    description="Generate an image with real-time progress updates via Server-Sent Events.",
)
async def generate_stream(
    request: Request,
    gen_request: GenerateRequest,
    api_key: Annotated[ApiKey, Depends(get_current_key)],
):
    """Generate an image with SSE progress streaming."""
    app = request.app
    generator: GeneratorService = app.state.generator
    credits = app.state.credits
    queue: QueueService = app.state.queue
    storage = app.state.storage
    settings = app.state.settings

    # Check model loaded
    if not generator.is_loaded:
        raise ModelNotLoadedError()

    # Check credits
    balance = credits.get_balance(api_key.id)
    if balance is None or balance < 1:
        raise InsufficientCreditsError(credits_available=balance or 0)

    # Sanitize prompt
    prompt = sanitize_prompt(gen_request.prompt, max_length=settings.security.max_prompt_length)

    # Deduct credit
    success, new_balance = credits.deduct(api_key.id, amount=1, reason="generation")
    if not success:
        raise InsufficientCreditsError(credits_available=new_balance)

    async def event_generator():
        job_id = f"job_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(gen_request)}"
        total_steps = gen_request.steps or settings.model.default_steps

        try:
            # Send queued event
            yield f"event: queued\ndata: {json.dumps({'job_id': job_id, 'position': 1})}\n\n"

            # Send started event
            yield f"event: started\ndata: {json.dumps({'job_id': job_id})}\n\n"

            # Progress callback
            progress_queue = asyncio.Queue()

            def progress_callback(step: int, total: int):
                try:
                    progress_queue.put_nowait((step, total))
                except asyncio.QueueFull:
                    pass

            # Start generation in background
            loop = asyncio.get_event_loop()
            gen_task = loop.run_in_executor(
                None,
                lambda: generator.generate(
                    prompt=prompt,
                    negative_prompt=gen_request.negative_prompt,
                    width=gen_request.width,
                    height=gen_request.height,
                    steps=gen_request.steps,
                    guidance_scale=gen_request.guidance_scale,
                    seed=gen_request.seed,
                    output_format=gen_request.output_format,
                    progress_callback=progress_callback,
                ),
            )

            # Stream progress events
            while not gen_task.done():
                try:
                    step, total = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    yield f"event: progress\ndata: {json.dumps({'step': step, 'total_steps': total})}\n\n"
                except asyncio.TimeoutError:
                    continue

            result = await gen_task

            # Store result
            image_bytes = add_exif_metadata(
                result.image_bytes,
                result.output_format,
                prompt=prompt,
                model_name=settings.model.name,
                seed=result.seed,
                steps=total_steps,
                width=gen_request.width,
                height=gen_request.height,
                guidance_scale=gen_request.guidance_scale or settings.model.default_guidance_scale,
            )

            storage.store_result(
                job_id=job_id,
                image_data=image_bytes,
                output_format=result.output_format,
                metadata={"seed": result.seed, "generation_time": result.generation_time},
            )

            # Send completed event
            completed_data = {
                "job_id": job_id,
                "retrieval_url": f"/v1/jobs/{job_id}",
                "seed": result.seed,
                "generation_time": result.generation_time,
                "warnings": result.warnings,
            }
            yield f"event: completed\ndata: {json.dumps(completed_data)}\n\n"

            app.state.usage_logger.log_success(
                key_id=api_key.id,
                generation_time=result.generation_time,
                model=settings.model.name,
            )

        except Exception as e:
            credits.refund(api_key.id, amount=1, reason="refund")
            app.state.usage_logger.log_failure(key_id=api_key.id, error=str(e))
            yield f"event: failed\ndata: {json.dumps({'job_id': job_id, 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
