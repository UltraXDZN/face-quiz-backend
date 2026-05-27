"""Unit tests for the in-process primitives in `api.live.session`.

These cover BE-1 acceptance criteria (issue #79):
  - subscribe/publish fanout
  - queue cleanup on consumer drop
  - reaper status transition
  - per-session write-lock contention
"""

import asyncio
import time

import pytest

from api.live.session import (
    BroadcastEvent,
    BroadcastHub,
    HEARTBEAT_TIMEOUT_SECONDS,
    LiveSession,
    LiveSessionRegistry,
    Reaper,
)


async def test_registry_get_or_create_is_idempotent():
    registry = LiveSessionRegistry()

    s1 = await registry.get_or_create("exam1", "pw1", "a@b.c")
    s2 = await registry.get_or_create("exam1", "pw1", "a@b.c")

    assert s1 is s2, "same key must return the same LiveSession instance"
    assert registry.get("exam1", "pw1", "a@b.c") is s1


async def test_registry_list_for_filters_by_exam_password():
    registry = LiveSessionRegistry()

    await registry.get_or_create("exam1", "pw1", "a@b.c")
    await registry.get_or_create("exam1", "pw1", "b@b.c")
    await registry.get_or_create("exam1", "pw2", "a@b.c")
    await registry.get_or_create("exam2", "pw1", "a@b.c")

    sessions = registry.list_for("exam1", "pw1")
    assert {s.email for s in sessions} == {"a@b.c", "b@b.c"}
    assert all(s.exam_id == "exam1" and s.password == "pw1" for s in sessions)


async def test_hub_publish_fans_out_to_all_subscribers():
    hub = BroadcastHub()
    q1 = await hub.subscribe("exam1", "pw1")
    q2 = await hub.subscribe("exam1", "pw1")
    q3 = await hub.subscribe("exam1", "pw2")  # different channel — must not receive

    event = BroadcastEvent(type="presence", data={"email": "a@b.c"})
    await hub.publish("exam1", "pw1", event)

    assert q1.get_nowait() is event
    assert q2.get_nowait() is event
    assert q3.empty(), "subscribers on other channels must not receive the event"


async def test_hub_unsubscribe_drops_subscriber_and_cleans_up_channel():
    hub = BroadcastHub()
    q1 = await hub.subscribe("exam1", "pw1")
    q2 = await hub.subscribe("exam1", "pw1")
    assert hub.subscriber_count("exam1", "pw1") == 2

    await hub.unsubscribe("exam1", "pw1", q1)
    assert hub.subscriber_count("exam1", "pw1") == 1

    await hub.unsubscribe("exam1", "pw1", q2)
    # When the last subscriber leaves, the channel should be removed.
    assert hub.subscriber_count("exam1", "pw1") == 0

    # Publishing on an empty channel must not raise.
    await hub.publish("exam1", "pw1", BroadcastEvent(type="presence", data={}))


async def test_hub_drops_oldest_on_queue_overflow():
    # Build a hub whose queue would overflow on the second put. We can't easily
    # reach 256 events without spamming, so we test the back-pressure path by
    # swapping the maxsize via direct construction.
    hub = BroadcastHub()
    q = await hub.subscribe("exam1", "pw1")
    # Shrink the queue to 1 to trigger the overflow path on the next publish.
    q._maxsize = 1  # type: ignore[attr-defined]

    e1 = BroadcastEvent(type="presence", data={"n": 1})
    e2 = BroadcastEvent(type="presence", data={"n": 2})
    await hub.publish("exam1", "pw1", e1)
    await hub.publish("exam1", "pw1", e2)

    # Oldest dropped; newest kept.
    remaining = q.get_nowait()
    assert remaining is e2
    assert q.empty()


async def test_reaper_flips_stale_solving_session_to_disconnected():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()
    reaper = Reaper(registry, hub)

    session = await registry.get_or_create("exam1", "pw1", "a@b.c")
    # Force the session to look stale.
    session.last_seen = time.time() - (HEARTBEAT_TIMEOUT_SECONDS + 1)

    q = await hub.subscribe("exam1", "pw1")
    await reaper._tick()  # bypass the sleep loop

    assert session.status == "disconnected"
    event = q.get_nowait()
    assert event.type == "presence"
    assert event.data["status"] == "disconnected"
    assert event.data["email"] == "a@b.c"


async def test_reaper_leaves_fresh_session_alone():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()
    reaper = Reaper(registry, hub)

    session = await registry.get_or_create("exam1", "pw1", "a@b.c")
    # Fresh heartbeat → stays solving.
    session.last_seen = time.time()

    q = await hub.subscribe("exam1", "pw1")
    await reaper._tick()

    assert session.status == "solving"
    assert q.empty()


async def test_reaper_does_not_flip_submitted_sessions():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()
    reaper = Reaper(registry, hub)

    session = await registry.get_or_create("exam1", "pw1", "a@b.c")
    session.status = "submitted"
    session.last_seen = time.time() - 9999  # ancient

    await reaper._tick()
    assert session.status == "submitted", "reaper must never touch submitted sessions"


async def test_per_session_lock_serializes_concurrent_mutators():
    """Two coroutines racing on the same session see serialized updates."""
    registry = LiveSessionRegistry()
    session = await registry.get_or_create("exam1", "pw1", "a@b.c")

    counter = {"n": 0}
    observed_inside_lock: list[int] = []

    async def mutator():
        async with session.lock:
            # Read-modify-write that would race without the lock.
            n = counter["n"]
            await asyncio.sleep(0)  # yield to the other task
            counter["n"] = n + 1
            observed_inside_lock.append(counter["n"])

    await asyncio.gather(*(mutator() for _ in range(50)))

    assert counter["n"] == 50
    assert observed_inside_lock == list(range(1, 51))


async def test_score_payload_handles_zero_total_points():
    session = LiveSession(exam_id="e", password="p", email="a@b.c")
    session.total_points = 0
    session.achieved_points = 0

    payload = session.score_payload()
    assert payload["percent"] == 0.0


async def test_reaper_start_stop_is_idempotent():
    registry = LiveSessionRegistry()
    hub = BroadcastHub()
    reaper = Reaper(registry, hub)

    reaper.start()
    reaper.start()  # double-start should be a no-op, not an error
    assert reaper._task is not None
    assert not reaper._task.done()

    await reaper.stop()
    await reaper.stop()  # double-stop should be a no-op
    assert reaper._task is None
