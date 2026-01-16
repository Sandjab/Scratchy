"""Admin endpoints for API key and system management."""

import csv
import io
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import StreamingResponse

from scratchy.models.requests import ApiKeyCreateRequest, ApiKeyUpdateRequest
from scratchy.models.responses import (
    ApiKeyResponse,
    ApiKeyCreateResponse,
    ApiKeyListResponse,
    AnalyticsResponse,
    AnalyticsExportResponse,
)
from scratchy.models.database import ApiKey, UsageLog
from scratchy.middleware.auth import require_admin

router = APIRouter(prefix="/v1/admin", tags=["Admin"])


@router.get(
    "/keys",
    response_model=ApiKeyListResponse,
    summary="List API keys",
    description="List all API keys (admin only).",
)
async def list_keys(
    request: Request,
    admin_key: Annotated[ApiKey, Depends(require_admin)],
    include_inactive: bool = Query(False, description="Include inactive keys"),
):
    """List all API keys."""
    auth_service = request.app.state.auth_service
    keys = auth_service.list_keys(include_inactive=include_inactive)

    return ApiKeyListResponse(
        keys=[
            ApiKeyResponse(
                id=key.id,
                name=key.name,
                credits=key.credits,
                rate_limit=key.rate_limit,
                created_at=key.created_at,
                last_used_at=key.last_used_at,
                is_active=key.is_active,
            )
            for key in keys
        ],
        total=len(keys),
    )


@router.post(
    "/keys",
    response_model=ApiKeyCreateResponse,
    summary="Create API key",
    description="Create a new API key (admin only). The key is only shown once!",
)
async def create_key(
    request: Request,
    key_request: ApiKeyCreateRequest,
    admin_key: Annotated[ApiKey, Depends(require_admin)],
):
    """Create a new API key."""
    auth_service = request.app.state.auth_service
    settings = request.app.state.settings

    plaintext_key, api_key = auth_service.create_key(
        name=key_request.name,
        credits=key_request.credits,
        rate_limit=key_request.rate_limit or settings.auth.default_rate_limit,
    )

    return ApiKeyCreateResponse(
        id=api_key.id,
        key=plaintext_key,  # Only shown once!
        name=api_key.name,
        credits=api_key.credits,
        rate_limit=api_key.rate_limit,
        created_at=api_key.created_at,
    )


@router.get(
    "/keys/{key_id}",
    response_model=ApiKeyResponse,
    summary="Get API key details",
    description="Get details of a specific API key (admin only).",
)
async def get_key(
    request: Request,
    key_id: str,
    admin_key: Annotated[ApiKey, Depends(require_admin)],
):
    """Get details of a specific API key."""
    auth_service = request.app.state.auth_service
    api_key = auth_service.get_key_by_id(key_id)

    if not api_key:
        from scratchy.middleware.errors import ProblemDetailException
        raise ProblemDetailException(
            status_code=404,
            problem_type="key-not-found",
            title="API Key Not Found",
            detail=f"No API key found with ID: {key_id}",
        )

    return ApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        credits=api_key.credits,
        rate_limit=api_key.rate_limit,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        is_active=api_key.is_active,
    )


@router.put(
    "/keys/{key_id}",
    response_model=ApiKeyResponse,
    summary="Update API key",
    description="Update an API key's properties (admin only).",
)
async def update_key(
    request: Request,
    key_id: str,
    update: ApiKeyUpdateRequest,
    admin_key: Annotated[ApiKey, Depends(require_admin)],
):
    """Update an API key."""
    auth_service = request.app.state.auth_service

    api_key = auth_service.update_key(
        key_id=key_id,
        name=update.name,
        credits=update.credits,
        rate_limit=update.rate_limit,
        is_active=update.is_active,
    )

    if not api_key:
        from scratchy.middleware.errors import ProblemDetailException
        raise ProblemDetailException(
            status_code=404,
            problem_type="key-not-found",
            title="API Key Not Found",
            detail=f"No API key found with ID: {key_id}",
        )

    return ApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        credits=api_key.credits,
        rate_limit=api_key.rate_limit,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        is_active=api_key.is_active,
    )


@router.delete(
    "/keys/{key_id}",
    summary="Delete API key",
    description="Deactivate an API key (admin only).",
)
async def delete_key(
    request: Request,
    key_id: str,
    admin_key: Annotated[ApiKey, Depends(require_admin)],
):
    """Delete (deactivate) an API key."""
    auth_service = request.app.state.auth_service
    success = auth_service.delete_key(key_id)

    if not success:
        from scratchy.middleware.errors import ProblemDetailException
        raise ProblemDetailException(
            status_code=404,
            problem_type="key-not-found",
            title="API Key Not Found",
            detail=f"No API key found with ID: {key_id}",
        )

    return {"status": "deleted", "key_id": key_id}


@router.get(
    "/analytics",
    summary="Export analytics",
    description="Export usage analytics (admin only).",
)
async def export_analytics(
    request: Request,
    admin_key: Annotated[ApiKey, Depends(require_admin)],
    format: str = Query("json", description="Export format: json or csv"),
    key_id: Optional[str] = Query(None, description="Filter by specific key ID"),
):
    """Export usage analytics."""
    from sqlalchemy import func
    from scratchy.models.database import init_database

    settings = request.app.state.settings
    _, Session = init_database(str(settings.storage.db_path))

    with Session() as session:
        # Build query
        query = session.query(
            ApiKey.id.label("key_id"),
            ApiKey.name,
            func.count(UsageLog.id).label("total_requests"),
            func.sum(
                func.case((UsageLog.status == "success", 1), else_=0)
            ).label("successful"),
            func.sum(
                func.case((UsageLog.status == "failed", 1), else_=0)
            ).label("failed"),
            func.sum(UsageLog.credits_used).label("credits_used"),
            func.max(UsageLog.timestamp).label("last_active"),
        ).outerjoin(UsageLog, ApiKey.id == UsageLog.key_id).group_by(ApiKey.id)

        if key_id:
            query = query.filter(ApiKey.id == key_id)

        results = query.all()

    analytics_data = [
        AnalyticsResponse(
            key_id=r.key_id,
            name=r.name,
            total_requests=r.total_requests or 0,
            successful_generations=r.successful or 0,
            failed_generations=r.failed or 0,
            total_credits_used=r.credits_used or 0,
            total_credits_refunded=0,  # TODO: Track refunds separately
            last_active=r.last_active,
        )
        for r in results
    ]

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "key_id",
                "name",
                "total_requests",
                "successful_generations",
                "failed_generations",
                "total_credits_used",
                "last_active",
            ],
        )
        writer.writeheader()
        for item in analytics_data:
            writer.writerow({
                "key_id": item.key_id,
                "name": item.name,
                "total_requests": item.total_requests,
                "successful_generations": item.successful_generations,
                "failed_generations": item.failed_generations,
                "total_credits_used": item.total_credits_used,
                "last_active": item.last_active.isoformat() if item.last_active else "",
            })

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=scratchy_analytics_{datetime.utcnow().strftime('%Y%m%d')}.csv"
            },
        )

    return AnalyticsExportResponse(
        data=analytics_data,
        exported_at=datetime.utcnow(),
    )


@router.post(
    "/backup",
    summary="Trigger backup",
    description="Trigger a manual database backup (admin only).",
)
async def trigger_backup(
    request: Request,
    admin_key: Annotated[ApiKey, Depends(require_admin)],
):
    """Trigger a manual database backup."""
    import shutil
    from pathlib import Path

    settings = request.app.state.settings
    db_path = Path(settings.storage.db_path)
    backup_dir = Path(settings.storage.backup_dir)

    if not db_path.exists():
        from scratchy.middleware.errors import ProblemDetailException
        raise ProblemDetailException(
            status_code=500,
            problem_type="backup-failed",
            title="Backup Failed",
            detail="Database file not found",
        )

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_name = f"scratchy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.db"
    backup_path = backup_dir / backup_name

    try:
        shutil.copy2(db_path, backup_path)
    except Exception as e:
        from scratchy.middleware.errors import ProblemDetailException
        raise ProblemDetailException(
            status_code=500,
            problem_type="backup-failed",
            title="Backup Failed",
            detail=str(e),
        )

    return {
        "status": "success",
        "backup_path": str(backup_path),
        "backup_size_bytes": backup_path.stat().st_size,
        "created_at": datetime.utcnow().isoformat(),
    }
