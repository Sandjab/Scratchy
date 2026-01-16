"""API routers."""

from scratchy.routers.health import router as health_router
from scratchy.routers.generate import router as generate_router
from scratchy.routers.jobs import router as jobs_router
from scratchy.routers.account import router as account_router
from scratchy.routers.admin import router as admin_router

__all__ = [
    "health_router",
    "generate_router",
    "jobs_router",
    "account_router",
    "admin_router",
]
