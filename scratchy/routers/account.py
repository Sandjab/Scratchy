"""Account management endpoints."""

from typing import Annotated

from fastapi import APIRouter, Request, Depends

from scratchy.models.responses import BalanceResponse
from scratchy.models.database import ApiKey
from scratchy.middleware.auth import get_current_key

router = APIRouter(prefix="/v1/account", tags=["Account"])


@router.get(
    "/balance",
    response_model=BalanceResponse,
    summary="Get credit balance",
    description="Get current credit balance and rate limit information.",
)
async def get_balance(
    request: Request,
    api_key: Annotated[ApiKey, Depends(get_current_key)],
):
    """Get the credit balance for the authenticated API key."""
    auth_service = request.app.state.auth_service

    requests_remaining = auth_service.get_requests_remaining(
        api_key.id,
        api_key.rate_limit,
    )

    return BalanceResponse(
        credits=api_key.credits,
        rate_limit=api_key.rate_limit,
        requests_remaining=requests_remaining,
    )
