"""Core services for Scratchy."""

from scratchy.services.auth import AuthService
from scratchy.services.credits import CreditService
from scratchy.services.queue import QueueService
from scratchy.services.storage import StorageService
from scratchy.services.generator import GeneratorService

__all__ = [
    "AuthService",
    "CreditService",
    "QueueService",
    "StorageService",
    "GeneratorService",
]
