"""Authentication middleware."""

from typing import Optional, Annotated

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from scratchy.models.database import ApiKey
from scratchy.services.auth import AuthService


# Security scheme for OpenAPI docs
security_scheme = HTTPBearer(
    scheme_name="API Key",
    description="API key in format: Bearer sk_xxxxx",
    auto_error=False,
)


class AuthMiddleware:
    """Middleware for API key authentication."""

    def __init__(self, auth_service: AuthService):
        """
        Initialize the auth middleware.

        Args:
            auth_service: The authentication service
        """
        self._auth_service = auth_service

    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    ) -> Optional[ApiKey]:
        """
        Authenticate the request and return the API key.

        Args:
            request: The incoming request
            credentials: Bearer token credentials

        Returns:
            ApiKey if authenticated

        Raises:
            HTTPException: If authentication fails
        """
        # Skip auth for health endpoints and docs
        if request.url.path in [
            "/v1/health/live",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics",
        ]:
            return None

        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Include 'Authorization: Bearer sk_xxx' header.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        api_key = self._auth_service.validate_key(credentials.credentials)

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or inactive API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check rate limit
        is_allowed, remaining = self._auth_service.check_rate_limit(
            api_key.id,
            api_key.rate_limit,
        )

        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(api_key.rate_limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Store in request state for later use
        request.state.api_key = api_key
        request.state.rate_limit_remaining = remaining

        return api_key


# Dependency for getting the current authenticated key
async def get_current_key(
    request: Request,
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(security_scheme),
    ] = None,
) -> ApiKey:
    """
    Dependency for getting the current authenticated API key.

    This is used in route handlers that require authentication.
    """
    if hasattr(request.state, "api_key") and request.state.api_key:
        return request.state.api_key

    # If middleware hasn't run yet, we need to authenticate here
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'Authorization: Bearer sk_xxx' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get auth service from app state
    auth_service: AuthService = request.app.state.auth_service
    api_key = auth_service.validate_key(credentials.credentials)

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check rate limit
    is_allowed, remaining = auth_service.check_rate_limit(
        api_key.id,
        api_key.rate_limit,
    )

    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(api_key.rate_limit),
                "X-RateLimit-Remaining": "0",
            },
        )

    request.state.api_key = api_key
    request.state.rate_limit_remaining = remaining

    return api_key


# Dependency for optional authentication (returns None if not authenticated)
async def get_optional_key(
    request: Request,
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials],
        Depends(security_scheme),
    ] = None,
) -> Optional[ApiKey]:
    """Dependency for optional authentication."""
    if not credentials:
        return None

    try:
        return await get_current_key(request, credentials)
    except HTTPException:
        return None


# Dependency for admin-only endpoints
async def require_admin(
    request: Request,
    api_key: Annotated[ApiKey, Depends(get_current_key)],
) -> ApiKey:
    """
    Dependency for admin-only endpoints.

    For now, all authenticated keys have admin access.
    In the future, this could check for specific scopes/permissions.
    """
    # TODO: Implement admin scope checking when key scopes are added
    return api_key
