"""Tests for the `LiveFlusher` — uses a stub flush_fn so we can verify
batching and retry-on-failure behavior without touching Firestore."""

import pytest

from api.live.persistence import LiveFlusher
from api.live.session import LiveSession, LiveSessionRegistry


async def test_flusher_skips_clean_sessions():
    registry = LiveSessionRegistry()
    flushed: list[str] = []

    async def fake_flush(session: LiveSession) -> None:
        flushed.append(session.email)

    flusher = LiveFlusher(registry, fake_flush)

    await registry.get_or_create("e", "p", "clean@x")  # dirty=False by default
    await flusher._tick()
    assert flushed == []


async def test_flusher_writes_dirty_sessions_and_clears_dirty_flag():
    registry = LiveSessionRegistry()
    flushed: list[str] = []

    async def fake_flush(session: LiveSession) -> None:
        flushed.append(session.email)

    flusher = LiveFlusher(registry, fake_flush)

    s1 = await registry.get_or_create("e", "p", "a@x")
    s1.dirty = True
    s2 = await registry.get_or_create("e", "p", "b@x")
    s2.dirty = True

    await flusher._tick()

    assert set(flushed) == {"a@x", "b@x"}
    assert s1.dirty is False
    assert s2.dirty is False
    assert s1.last_flushed_at > 0
    assert s2.last_flushed_at > 0


async def test_flusher_skips_submitted_sessions():
    registry = LiveSessionRegistry()
    flushed: list[str] = []

    async def fake_flush(session: LiveSession) -> None:
        flushed.append(session.email)

    flusher = LiveFlusher(registry, fake_flush)

    session = await registry.get_or_create("e", "p", "done@x")
    session.dirty = True
    session.status = "submitted"

    await flusher._tick()
    # Submitted sessions belong to the finalize path; the flusher must not
    # double-write.
    assert flushed == []
    # Still dirty — but that's fine because the finalize path will overwrite
    # the doc entirely.
    assert session.dirty is True


async def test_flusher_keeps_session_dirty_on_failure():
    registry = LiveSessionRegistry()
    attempts: list[str] = []

    async def failing_flush(session: LiveSession) -> None:
        attempts.append(session.email)
        raise RuntimeError("boom")

    flusher = LiveFlusher(registry, failing_flush)

    session = await registry.get_or_create("e", "p", "a@x")
    session.dirty = True

    await flusher._tick()
    assert attempts == ["a@x"]
    assert session.dirty is True, "must retry on next tick after failure"

    # Second tick retries.
    await flusher._tick()
    assert attempts == ["a@x", "a@x"]


async def test_flusher_start_stop_lifecycle():
    registry = LiveSessionRegistry()

    async def fake_flush(_: LiveSession) -> None:
        pass

    flusher = LiveFlusher(registry, fake_flush)
    flusher.start()
    assert flusher._task is not None
    assert not flusher._task.done()

    await flusher.stop()
    assert flusher._task is None

    # Idempotent stop.
    await flusher.stop()
