"""Retry helper and transient-failure classifier (issue #90)."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable, TypeVar, Awaitable

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 3
BASE_DELAY = 0.5   # seconds
MAX_DELAY = 10.0   # seconds

_AWS_TRANSIENT_CODES = {
    "RequestThrottled", "Throttling", "ThrottlingException",
    "RequestTimeout", "SlowDown", "ServiceUnavailable",
    "InternalError", "InternalFailure",
}


def is_retryable(exc: BaseException) -> bool:
    """Return True if exc is a transient failure that can safely be retried."""
    # httpx
    try:
        import httpx
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {429, 500, 502, 503, 504}
    except ImportError:
        pass

    # botocore / AWS
    try:
        from botocore.exceptions import (
            ConnectTimeoutError, ReadTimeoutError,
            EndpointConnectionError, ClientError,
        )
        if isinstance(exc, (ConnectTimeoutError, ReadTimeoutError, EndpointConnectionError)):
            return True
        if isinstance(exc, ClientError):
            code = exc.response.get("Error", {}).get("Code", "")
            return code in _AWS_TRANSIENT_CODES
    except ImportError:
        pass

    # google.api_core (Firestore)
    try:
        from google.api_core.exceptions import (
            DeadlineExceeded, ServiceUnavailable, TooManyRequests,
        )
        if isinstance(exc, (DeadlineExceeded, ServiceUnavailable, TooManyRequests)):
            return True
    except ImportError:
        pass

    return False


def _backoff(attempt: int, base: float, ceiling: float) -> float:
    return min(base * (2 ** attempt) + random.uniform(0.0, 0.5), ceiling)


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
) -> T:
    """Retry an async callable on transient failures with exponential backoff + jitter."""
    last_exc: BaseException
    for attempt in range(max_attempts):
        try:
            return await fn()
        except BaseException as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
            if attempt + 1 < max_attempts:
                delay = _backoff(attempt, base_delay, max_delay)
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1, max_attempts, delay, exc,
                )
                await asyncio.sleep(delay)
    raise last_exc


def with_retry_sync(
    fn: Callable[[], T],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
) -> T:
    """Retry a sync callable on transient failures with exponential backoff + jitter."""
    last_exc: BaseException
    for attempt in range(max_attempts):
        try:
            return fn()
        except BaseException as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
            if attempt + 1 < max_attempts:
                delay = _backoff(attempt, base_delay, max_delay)
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1, max_attempts, delay, exc,
                )
                time.sleep(delay)
    raise last_exc
