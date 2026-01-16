"""Webhook notification service."""

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class WebhookService:
    """Service for sending webhook notifications."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        timeout_seconds: int = 10,
        max_retries: int = 3,
    ):
        """
        Initialize the webhook service.

        Args:
            secret_key: Secret key for signing webhooks
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
        """
        self._secret_key = secret_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Start the HTTP client."""
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def stop(self):
        """Stop the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _sign_payload(self, payload: str) -> str:
        """Sign a webhook payload using HMAC-SHA256."""
        if not self._secret_key:
            return ""

        signature = hmac.new(
            self._secret_key.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        return f"sha256={signature}"

    async def send(
        self,
        url: str,
        event: str,
        data: dict,
    ) -> bool:
        """
        Send a webhook notification.

        Args:
            url: The webhook URL
            event: Event type (e.g., job.completed)
            data: Event data

        Returns:
            True if successful, False otherwise
        """
        if not self._client:
            await self.start()

        payload = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        }

        payload_str = json.dumps(payload, sort_keys=True)
        signature = self._sign_payload(payload_str)

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Scratchy-Webhook/1.0",
        }

        if signature:
            headers["X-Scratchy-Signature"] = signature

        for attempt in range(self._max_retries):
            try:
                response = await self._client.post(
                    url,
                    content=payload_str,
                    headers=headers,
                )

                if response.status_code < 300:
                    logger.info(f"Webhook sent successfully to {url}")
                    return True

                logger.warning(
                    f"Webhook to {url} returned {response.status_code} "
                    f"(attempt {attempt + 1}/{self._max_retries})"
                )

            except httpx.RequestError as e:
                logger.warning(
                    f"Webhook to {url} failed: {e} "
                    f"(attempt {attempt + 1}/{self._max_retries})"
                )

            # Exponential backoff
            if attempt < self._max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        logger.error(f"Webhook to {url} failed after {self._max_retries} attempts")
        return False

    async def send_job_completed(
        self,
        url: str,
        job_id: str,
        seed: int,
        generation_time: float,
        retrieval_url: str,
        expires_at: datetime,
    ) -> bool:
        """Send a job.completed webhook."""
        return await self.send(
            url=url,
            event="job.completed",
            data={
                "job_id": job_id,
                "status": "completed",
                "seed": seed,
                "generation_time": generation_time,
                "retrieval_url": retrieval_url,
                "expires_at": expires_at.isoformat(),
            },
        )

    async def send_job_failed(
        self,
        url: str,
        job_id: str,
        error_message: str,
    ) -> bool:
        """Send a job.failed webhook."""
        return await self.send(
            url=url,
            event="job.failed",
            data={
                "job_id": job_id,
                "status": "failed",
                "error_message": error_message,
            },
        )
