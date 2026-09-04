"""Shared Server Security Utilities (Rate Limiting, Authentication, CORS & Host Validation).

Provides zero-dependency, thread-safe token bucket rate limiting and uniform security helpers
for both FastAPI (promptdiff serve) and Python HTTP Studio (promptdiff studio).
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time

logger = logging.getLogger("promptdiff.cli.security")


class TokenBucketRateLimiter:
    """Thread-safe Token Bucket Rate Limiter keyed by client IP."""

    def __init__(self, rate_per_minute: int = 60, burst_capacity: int | None = None) -> None:
        self.rate_per_minute = max(1, rate_per_minute)
        self.burst_capacity = float(burst_capacity if burst_capacity is not None else self.rate_per_minute)
        self.fill_rate = self.rate_per_minute / 60.0  # tokens per second
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_update)
        self._lock = threading.Lock()

    def acquire(self, key: str = "default", tokens: float = 1.0) -> bool:
        """Attempt to consume tokens for the given client key (e.g. IP).

        Returns True if tokens were consumed, False if rate limit exceeded.
        """
        now = time.monotonic()
        with self._lock:
            current_tokens, last_update = self._buckets.get(key, (self.burst_capacity, now))
            # Refill tokens based on elapsed time
            elapsed = max(0.0, now - last_update)
            current_tokens = min(self.burst_capacity, current_tokens + (elapsed * self.fill_rate))

            if current_tokens >= tokens:
                self._buckets[key] = (current_tokens - tokens, now)
                return True
            else:
                self._buckets[key] = (current_tokens, now)
                return False

    def reset(self) -> None:
        """Clear all rate limiting state (useful for tests)."""
        with self._lock:
            self._buckets.clear()


def verify_api_key_value(provided_key: str | None, expected_key: str | None = None) -> bool:
    """Safely verify an API key in constant time.

    If expected_key is None or empty, verification passes (unauthenticated mode).
    """
    target = expected_key if expected_key is not None else os.getenv("PROMPTDIFF_API_KEY")
    if not target:
        return True
    if not provided_key:
        return False
    return secrets.compare_digest(provided_key, target)


def validate_bind_host(host: str, api_key: str | None = None) -> str:
    """Validate server bind interface and warn if binding to public IP without auth.

    Returns the host after validation.
    """
    key = api_key if api_key is not None else os.getenv("PROMPTDIFF_API_KEY")
    is_localhost = host in ("127.0.0.1", "localhost", "::1")

    if not is_localhost and not key:
        msg = f"⚠️  GÜVENLİK UYARISI: Sunucu herkese açık IP'ye ({host}) bind ediliyor ama API key tanımlı değil!"
        logger.warning(msg)
        print(f"\n[!] {msg}\n", flush=True)
    return host
