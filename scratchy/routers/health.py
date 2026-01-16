"""Health check endpoints."""

from datetime import datetime

from fastapi import APIRouter, Request

from scratchy.models.responses import HealthLiveResponse, HealthReadyResponse

router = APIRouter(prefix="/v1/health", tags=["Health"])


@router.get(
    "/live",
    response_model=HealthLiveResponse,
    summary="Liveness probe",
    description="Returns OK if the server process is running.",
)
async def liveness():
    """Check if the server is alive."""
    return HealthLiveResponse(
        status="ok",
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/ready",
    response_model=HealthReadyResponse,
    summary="Readiness probe",
    description="Returns detailed status including model, database, and queue state.",
)
async def readiness(request: Request):
    """Check if the server is ready to accept requests."""
    app = request.app

    # Get services from app state
    generator = app.state.generator
    queue = app.state.queue

    # Check model status
    model_loaded = generator.is_loaded if generator else False

    # Check GPU
    gpu_info = generator.get_gpu_info() if generator else {"available": False}

    # Check database by getting engine
    db_status = "connected"
    try:
        from scratchy.models.database import get_engine
        engine = get_engine(str(app.state.settings.storage.db_path))
        with engine.connect() as conn:
            conn.execute("SELECT 1")
    except Exception:
        db_status = "error"

    # Determine overall status
    is_ready = model_loaded and db_status == "connected"

    response = HealthReadyResponse(
        status="ready" if is_ready else "not_ready",
        model=app.state.settings.model.name,
        model_loaded=model_loaded,
        database=db_status,
        queue_depth=queue.depth if queue else 0,
        queue_capacity=queue.capacity if queue else 0,
        gpu_available=gpu_info.get("available", False),
        gpu_memory_free_gb=gpu_info.get("free_memory_gb"),
    )

    return response
