"""Load testing with Locust.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8080
"""

import json
import random
from locust import HttpUser, task, between


class ScratchyUser(HttpUser):
    """Simulated Scratchy API user."""

    wait_time = between(1, 5)

    def on_start(self):
        """Setup before starting tasks."""
        # You need to set this to a valid test API key
        self.api_key = "sk_test_key_here"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @task(10)
    def health_check(self):
        """Check health endpoint (frequent)."""
        self.client.get("/v1/health/live")

    @task(5)
    def check_readiness(self):
        """Check readiness endpoint."""
        self.client.get("/v1/health/ready")

    @task(3)
    def check_balance(self):
        """Check credit balance."""
        self.client.get(
            "/v1/account/balance",
            headers=self.headers,
        )

    @task(1)
    def generate_image(self):
        """Generate an image (infrequent due to cost)."""
        prompts = [
            "A beautiful sunset over mountains",
            "A cat sitting on a windowsill",
            "An abstract geometric pattern",
            "A futuristic city skyline",
            "A serene forest path",
        ]

        payload = {
            "prompt": random.choice(prompts),
            "width": 512,  # Smaller for load testing
            "height": 512,
            "steps": 4,  # Minimum steps for speed
            "output_format": "jpeg",
        }

        with self.client.post(
            "/v1/generate",
            headers=self.headers,
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 402:
                # Out of credits - expected in load test
                response.success()
            elif response.status_code == 429:
                # Rate limited - expected in load test
                response.success()
            elif response.status_code == 503:
                # Queue full - expected under load
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class AdminUser(HttpUser):
    """Simulated admin user for admin endpoint testing."""

    wait_time = between(5, 10)
    weight = 1  # Less frequent than regular users

    def on_start(self):
        """Setup admin credentials."""
        self.api_key = "sk_admin_key_here"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @task(5)
    def list_keys(self):
        """List API keys."""
        self.client.get(
            "/v1/admin/keys",
            headers=self.headers,
        )

    @task(1)
    def get_analytics(self):
        """Get usage analytics."""
        self.client.get(
            "/v1/admin/analytics",
            headers=self.headers,
        )
