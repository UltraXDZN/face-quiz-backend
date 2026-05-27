"""Tests for the admin SSE stream generator.

The generator (`build_event_stream`) is factored out of the route handler so
we can drive it directly with asyncio instead of trying to coax a long-lived
SSE response out of `TestClient.stream`.
"""

import asyncio
import json

import pytest

from api.live.routes import build_event_stream
from api.live.session import BroadcastEvent, BroadcastHub, LiveSessionRegistry


def _parse_sse(chunk: str) -> tuple[str, dict] | None:
    """Crude SSE parser for tests: return `(event_type, json_data)` or None
    for a keep-alive comment."""
    if chunk.startswith(":"):
        return None
    lines = chunk.strip().split("\n")
    event_type = next(l[len("event: ") :] for l in lines if l.startswith("event: "))
    data_str = next(l[len("data: ") :] for l in lines if l.startswith("data: "))
    return event_type, json.loads(data_str)


async def _stream_take(agen, n: int, timeout: float = 1.0) -> list[str]:
    """Pull up to `n` chunks from an async generator with a per-chunk timeout."""
    out: list[str] = []
    for _ in range(n):
        chunk = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
        out.append(chunk)
    return out


async def test_stream_emits_snapshot_for_each_existing_session():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()

    await registry.get_or_create("e", "p", "a@x")
    await registry.get_or_create("e", "p", "b@x")

    async def never_disconnected() -> bool:
        return False

    agen = build_event_stream(registry, hub, "e", "p", never_disconnected)
    try:
        # Two sessions × (presence + score) = 4 snapshot chunks.
        chunks = await _stream_take(agen, 4)
    finally:
        await agen.aclose()

    events = [_parse_sse(c) for c in chunks]
    emails = {data["email"] for et, data in events}
    types = {et for et, data in events}
    assert emails == {"a@x", "b@x"}
    assert types == {"presence", "score"}


async def test_stream_forwards_live_events_after_snapshot():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()

    async def never_disconnected() -> bool:
        return False

    agen = build_event_stream(registry, hub, "e", "p", never_disconnected)
    try:
        # Drive the generator one step so it subscribes (no sessions → no
        # snapshot → the first __anext__ blocks on the queue). Then publish.
        task = asyncio.create_task(agen.__anext__())
        for _ in range(5):
            await asyncio.sleep(0)
        assert hub.subscriber_count("e", "p") == 1

        await hub.publish(
            "e", "p", BroadcastEvent(type="score", data={"email": "a@x", "percent": 50})
        )
        chunk = await asyncio.wait_for(task, timeout=1.0)
    finally:
        await agen.aclose()

    parsed = _parse_sse(chunk)
    assert parsed == ("score", {"email": "a@x", "percent": 50})


async def test_stream_emits_keepalive_after_timeout():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()

    async def never_disconnected() -> bool:
        return False

    # Tiny keepalive so the test doesn't sleep for 15s.
    agen = build_event_stream(
        registry, hub, "e", "p", never_disconnected, keepalive_seconds=0.05
    )
    try:
        chunk = await asyncio.wait_for(agen.__anext__(), timeout=1.0)
    finally:
        await agen.aclose()

    assert chunk.startswith(":"), "expected a keep-alive comment frame"


async def test_stream_unsubscribes_when_client_disconnects():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()

    disconnected = {"v": False}

    async def is_disconnected() -> bool:
        return disconnected["v"]

    agen = build_event_stream(
        registry, hub, "e", "p", is_disconnected, keepalive_seconds=0.01
    )
    # Drain one keepalive so the generator has actually subscribed.
    await asyncio.wait_for(agen.__anext__(), timeout=1.0)
    assert hub.subscriber_count("e", "p") == 1

    disconnected["v"] = True
    # Generator should exit on the next keepalive tick → StopAsyncIteration.
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(agen.__anext__(), timeout=1.0)

    assert hub.subscriber_count("e", "p") == 0, "must unsubscribe on disconnect"


async def test_multiple_subscribers_each_receive_the_same_event():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()

    async def never_disconnected() -> bool:
        return False

    a = build_event_stream(registry, hub, "e", "p", never_disconnected)
    b = build_event_stream(registry, hub, "e", "p", never_disconnected)
    try:
        # Both generators must subscribe before the publish lands. Pull one
        # chunk from each with a timeout — they'll block until a keep-alive
        # fires, so we instead publish first.
        # To force subscription, start a publish task after a microsleep.
        await asyncio.sleep(0)  # let both generators register
        # Drive each generator one step on its own task so subscription
        # actually happens.
        ta = asyncio.create_task(a.__anext__())
        tb = asyncio.create_task(b.__anext__())
        # Yield repeatedly so the subscribers register before publishing.
        for _ in range(5):
            await asyncio.sleep(0)
        assert hub.subscriber_count("e", "p") == 2

        await hub.publish(
            "e", "p", BroadcastEvent(type="presence", data={"email": "x@x"})
        )

        chunk_a = await asyncio.wait_for(ta, timeout=1.0)
        chunk_b = await asyncio.wait_for(tb, timeout=1.0)
        assert "event: presence" in chunk_a
        assert "event: presence" in chunk_b
    finally:
        await a.aclose()
        await b.aclose()
