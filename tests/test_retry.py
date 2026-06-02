"""Tests for issue #90: retry helper and transient/permanent classifier."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.common.retry import is_retryable, with_retry, with_retry_sync


# ── is_retryable: httpx ───────────────────────────────────────────────────────

def test_httpx_timeout_is_retryable():
    import httpx
    assert is_retryable(httpx.TimeoutException("timed out")) is True


def test_httpx_connect_error_is_retryable():
    import httpx
    assert is_retryable(httpx.ConnectError("refused")) is True


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_httpx_retryable_status_is_retryable(status):
    import httpx
    resp = MagicMock()
    resp.status_code = status
    exc = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
    assert is_retryable(exc) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_httpx_client_error_status_is_not_retryable(status):
    import httpx
    resp = MagicMock()
    resp.status_code = status
    exc = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
    assert is_retryable(exc) is False


# ── is_retryable: botocore ────────────────────────────────────────────────────

def test_boto_connect_timeout_is_retryable():
    from botocore.exceptions import ConnectTimeoutError
    assert is_retryable(ConnectTimeoutError(endpoint_url="s3")) is True


def test_boto_read_timeout_is_retryable():
    from botocore.exceptions import ReadTimeoutError
    assert is_retryable(ReadTimeoutError(endpoint_url="s3")) is True


@pytest.mark.parametrize("code", [
    "Throttling", "ThrottlingException", "RequestThrottled",
    "RequestTimeout", "SlowDown", "ServiceUnavailable",
    "InternalError", "InternalFailure",
])
def test_boto_transient_client_error_is_retryable(code):
    from botocore.exceptions import ClientError
    exc = ClientError({"Error": {"Code": code, "Message": "..."}}, "PutObject")
    assert is_retryable(exc) is True


def test_boto_permanent_client_error_is_not_retryable():
    from botocore.exceptions import ClientError
    exc = ClientError({"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject")
    assert is_retryable(exc) is False


# ── is_retryable: google.api_core (Firestore) ─────────────────────────────────

def test_firestore_deadline_exceeded_is_retryable():
    from google.api_core.exceptions import DeadlineExceeded
    assert is_retryable(DeadlineExceeded("deadline exceeded")) is True


def test_firestore_service_unavailable_is_retryable():
    from google.api_core.exceptions import ServiceUnavailable
    assert is_retryable(ServiceUnavailable("unavailable")) is True


def test_firestore_too_many_requests_is_retryable():
    from google.api_core.exceptions import TooManyRequests
    assert is_retryable(TooManyRequests("too many requests")) is True


# ── is_retryable: permanent errors ────────────────────────────────────────────

@pytest.mark.parametrize("exc", [
    ValueError("bad input"),
    TypeError("wrong type"),
    KeyError("missing key"),
    PermissionError("denied"),
])
def test_standard_errors_are_not_retryable(exc):
    assert is_retryable(exc) is False


# ── with_retry_sync ───────────────────────────────────────────────────────────

def test_sync_succeeds_first_attempt():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert with_retry_sync(fn, max_attempts=3) == "ok"
    assert len(calls) == 1


def test_sync_retries_on_transient_error():
    import httpx
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("refused")
        return "ok"

    with patch("time.sleep"):
        result = with_retry_sync(fn, max_attempts=3, base_delay=0.0)

    assert result == "ok"
    assert len(calls) == 3


def test_sync_raises_immediately_on_permanent_error():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        with_retry_sync(fn, max_attempts=3)

    assert len(calls) == 1


def test_sync_exhausts_all_attempts_and_reraises():
    import httpx
    calls = []

    def fn():
        calls.append(1)
        raise httpx.ConnectError("refused")

    with patch("time.sleep"):
        with pytest.raises(httpx.ConnectError):
            with_retry_sync(fn, max_attempts=3, base_delay=0.0)

    assert len(calls) == 3


def test_sync_backoff_grows_each_attempt():
    import httpx
    delays = []

    def fn():
        raise httpx.ConnectError("refused")

    with patch("time.sleep", side_effect=lambda d: delays.append(d)):
        with pytest.raises(httpx.ConnectError):
            with_retry_sync(fn, max_attempts=3, base_delay=1.0, max_delay=100.0)

    assert len(delays) == 2
    assert delays[1] > delays[0]


# ── with_retry (async) ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_succeeds_first_attempt():
    calls = []

    async def fn():
        calls.append(1)
        return "ok"

    assert await with_retry(fn, max_attempts=3) == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_retries_on_transient_error():
    import httpx
    calls = []

    async def fn():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("refused")
        return "ok"

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await with_retry(fn, max_attempts=3, base_delay=0.0)

    assert result == "ok"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_async_raises_immediately_on_permanent_error():
    calls = []

    async def fn():
        calls.append(1)
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await with_retry(fn, max_attempts=3)

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_exhausts_all_attempts_and_reraises():
    import httpx
    calls = []

    async def fn():
        calls.append(1)
        raise httpx.ConnectError("refused")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.ConnectError):
            await with_retry(fn, max_attempts=3, base_delay=0.0)

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_async_backoff_grows_each_attempt():
    import httpx
    delays = []

    async def fn():
        raise httpx.ConnectError("refused")

    async def capture_sleep(d):
        delays.append(d)

    with patch("asyncio.sleep", side_effect=capture_sleep):
        with pytest.raises(httpx.ConnectError):
            await with_retry(fn, max_attempts=3, base_delay=1.0, max_delay=100.0)

    assert len(delays) == 2
    assert delays[1] > delays[0]
