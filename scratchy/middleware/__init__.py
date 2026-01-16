"""Middleware for Scratchy API."""

from scratchy.middleware.auth import AuthMiddleware, get_current_key
from scratchy.middleware.errors import (
    ProblemDetailException,
    problem_detail_exception_handler,
    validation_exception_handler,
)

__all__ = [
    "AuthMiddleware",
    "get_current_key",
    "ProblemDetailException",
    "problem_detail_exception_handler",
    "validation_exception_handler",
]
