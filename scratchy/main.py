"""
Scratchy - Production-ready AI Image Generation API Server

Main application entry point.
"""

import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from scratchy.config import get_settings
from scratchy.models.database import init_database
from scratchy.services.auth import AuthService
from scratchy.services.credits import CreditService
from scratchy.services.queue import QueueService
from scratchy.services.storage import StorageService
from scratchy.services.generator import GeneratorService
from scratchy.services.webhooks import WebhookService
from scratchy.services.usage import UsageLogger
from scratchy.middleware.errors import (
    ProblemDetailException,
    problem_detail_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from scratchy.routers import (
    health_router,
    generate_router,
    jobs_router,
    account_router,
    admin_router,
)


def setup_logging(settings) -> None:
    """Configure logging based on settings."""
    log_format = (
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": "%(message)s"}'
        if settings.logging.format == "json"
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, settings.logging.level),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    logger = logging.getLogger(__name__)

    # Ensure directories exist
    Path(settings.storage.jobs_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.storage.db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.storage.backup_dir).mkdir(parents=True, exist_ok=True)

    # Initialize database
    logger.info("Initializing database...")
    engine, Session = init_database(str(settings.storage.db_path))

    # Initialize services
    logger.info("Initializing services...")
    auth_service = AuthService(Session, settings.auth.default_rate_limit)
    credit_service = CreditService(Session)
    queue_service = QueueService(
        Session,
        max_depth=settings.queue.max_depth,
        job_ttl_hours=settings.storage.jobs_ttl_hours,
    )
    storage_service = StorageService(
        settings.storage.jobs_dir,
        ttl_hours=settings.storage.jobs_ttl_hours,
    )
    generator_service = GeneratorService(settings.model)
    webhook_service = WebhookService()
    usage_logger = UsageLogger(Session)

    # Store in app state
    app.state.settings = settings
    app.state.auth_service = auth_service
    app.state.credits = credit_service
    app.state.queue = queue_service
    app.state.storage = storage_service
    app.state.generator = generator_service
    app.state.webhooks = webhook_service
    app.state.usage_logger = usage_logger

    # Load the model (unless skipped)
    if settings.model.skip_load:
        logger.warning("Model loading SKIPPED (model.skip_load=True). Generation endpoints will not work.")
    else:
        logger.info(f"Loading model: {settings.model.name}...")
        try:
            generator_service.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Server will start but generation endpoints will not work.")
            # Don't exit - let the server run for API testing

    # Start background services
    await storage_service.start_cleanup_task(interval_minutes=5)
    await webhook_service.start()

    logger.info("Scratchy is ready to serve requests")

    # Setup graceful shutdown
    shutdown_event = None

    def handle_shutdown(signum, frame):
        nonlocal shutdown_event
        logger.info("Shutdown signal received...")
        if shutdown_event:
            shutdown_event.set()

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Stop accepting new requests (handled by uvicorn)

    # Stop background tasks
    await storage_service.stop_cleanup_task()
    await webhook_service.stop()

    # Wait for current generation to complete (max 30s)
    # This is handled by uvicorn's graceful shutdown

    # Unload model
    generator_service.unload_model()

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    # Setup logging
    setup_logging(settings)

    app = FastAPI(
        title="Scratchy Image Generation API",
        description="Production-ready AI image generation API with authentication, credits, and job management.",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    app.add_exception_handler(ProblemDetailException, problem_detail_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Register routers
    app.include_router(health_router)
    app.include_router(generate_router)
    app.include_router(jobs_router)
    app.include_router(account_router)
    app.include_router(admin_router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "scratchy.main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        reload=False,
    )
