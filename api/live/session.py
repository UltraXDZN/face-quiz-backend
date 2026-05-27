"""In-process primitives for the live exam monitoring feature.

This module deliberately exposes only the data plane:
  - `LiveSession`        — per-(exam, password, email) mutable state
  - `LiveSessionRegistry` — keyed lookup of all sessions
  - `BroadcastHub`       — fan-out queues per (exam, password) channel
  - `Reaper`             — periodic task that marks stale sessions disconnected

HTTP transport lives in `api/live/routes.py` (BE-2, BE-3). Keeping the
primitives transport-agnostic lets us test them without spinning up the app
and lets BE-4 add a per-student channel without touching this file.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Iterable, Literal


LiveStatus = Literal["solving", "submitted", "disconnected", "not_started"]


# A heartbeat older than this flips the session to "disconnected". The reaper
# polls at REAPER_INTERVAL_SECONDS, so the worst-case latency to detect a
# drop is HEARTBEAT_TIMEOUT_SECONDS + REAPER_INTERVAL_SECONDS.
HEARTBEAT_TIMEOUT_SECONDS = 30.0
REAPER_INTERVAL_SECONDS = 5.0

# Per-subscriber queue depth. Generous enough for normal bursts, bounded so
# a slow client can't pin unbounded memory.
SUBSCRIBER_QUEUE_MAXSIZE = 256


@dataclass
class BroadcastEvent:
    """Transport-agnostic event for the broadcast hub.

    The SSE layer (BE-3) is responsible for serializing this to
    `event: <type>\\ndata: <json>\\n\\n`.
    """

    type: str  # "presence" | "score" | "submitted"
    data: dict


@dataclass
class LiveSession:
    exam_id: str
    password: str
    email: str
    answers: dict[str, bool | None] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    status: LiveStatus = "solving"
    achieved_points: float = 0.0
    total_points: float = 0.0
    current_group: int = 1
    answered_count: int = 0
    # Per-session write lock — held by anyone mutating answers/score so the
    # reaper and request handlers don't trample each other.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def presence_payload(self) -> dict:
        return {
            "email": self.email,
            "status": self.status,
            "last_seen": self.last_seen,
            "current_group": self.current_group,
            "answered_count": self.answered_count,
        }

    def score_payload(self) -> dict:
        percent = (
            (self.achieved_points / self.total_points) * 100
            if self.total_points > 0
            else 0.0
        )
        return {
            "email": self.email,
            "achieved_points": self.achieved_points,
            "total_points": self.total_points,
            "percent": percent,
        }


class LiveSessionRegistry:
    """Keyed lookup of all live sessions across all exams.

    Single instance per app, held on `app.state.live_registry`.
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str, str], LiveSession] = {}
        # Guards insert/remove on the dict itself; per-session mutation uses
        # `LiveSession.lock` instead.
        self._lock = asyncio.Lock()

    async def get_or_create(
        self, exam_id: str, password: str, email: str
    ) -> LiveSession:
        key = (exam_id, password, email)
        async with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = LiveSession(exam_id=exam_id, password=password, email=email)
                self._sessions[key] = session
            return session

    def get(self, exam_id: str, password: str, email: str) -> LiveSession | None:
        return self._sessions.get((exam_id, password, email))

    def list_for(self, exam_id: str, password: str) -> list[LiveSession]:
        return [
            s
            for (e_id, pw, _email), s in self._sessions.items()
            if e_id == exam_id and pw == password
        ]

    def iter_all(self) -> Iterable[LiveSession]:
        # Snapshot to a list so the reaper can iterate without holding the
        # registry lock for the duration of the scan.
        return list(self._sessions.values())

    async def remove(self, exam_id: str, password: str, email: str) -> None:
        async with self._lock:
            self._sessions.pop((exam_id, password, email), None)


class BroadcastHub:
    """Per-channel fan-out. A channel is keyed by (exam_id, password).

    Each subscriber gets its own bounded queue. On overflow the oldest event
    is dropped — admin SSE consumers care about the latest state, not every
    historical tick.
    """

    def __init__(self) -> None:
        self._channels: dict[tuple[str, str], set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(
        self, exam_id: str, password: str
    ) -> asyncio.Queue[BroadcastEvent]:
        q: asyncio.Queue[BroadcastEvent] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_MAXSIZE)
        key = (exam_id, password)
        async with self._lock:
            self._channels.setdefault(key, set()).add(q)
        return q

    async def unsubscribe(
        self, exam_id: str, password: str, q: asyncio.Queue[BroadcastEvent]
    ) -> None:
        key = (exam_id, password)
        async with self._lock:
            subs = self._channels.get(key)
            if subs is None:
                return
            subs.discard(q)
            if not subs:
                self._channels.pop(key, None)

    async def publish(
        self, exam_id: str, password: str, event: BroadcastEvent
    ) -> None:
        key = (exam_id, password)
        # Snapshot the subscriber set so we don't hold the channel lock while
        # putting onto individual queues.
        async with self._lock:
            subs = list(self._channels.get(key, ()))
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest, push newest. A slow consumer should fall behind,
                # not block publishers.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    def subscriber_count(self, exam_id: str, password: str) -> int:
        return len(self._channels.get((exam_id, password), ()))


class Reaper:
    """Periodically flips stale `solving` sessions to `disconnected`.

    Owned by the FastAPI lifespan; `start()` schedules the task and `stop()`
    cancels it cleanly on shutdown.
    """

    def __init__(self, registry: LiveSessionRegistry, hub: BroadcastHub) -> None:
        self._registry = registry
        self._hub = hub
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="live-reaper")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(REAPER_INTERVAL_SECONDS)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Never let a transient error kill the reaper; just log and
                # try again on the next tick.
                print(f"[live-reaper] tick error: {e}")

    async def _tick(self) -> None:
        now = time.time()
        for session in self._registry.iter_all():
            if session.status != "solving":
                continue
            if now - session.last_seen <= HEARTBEAT_TIMEOUT_SECONDS:
                continue
            changed = False
            async with session.lock:
                # Re-check under the lock — a heartbeat may have landed in
                # the meantime.
                if (
                    session.status == "solving"
                    and now - session.last_seen > HEARTBEAT_TIMEOUT_SECONDS
                ):
                    session.status = "disconnected"
                    changed = True
            if changed:
                await self._hub.publish(
                    session.exam_id,
                    session.password,
                    BroadcastEvent(type="presence", data=session.presence_payload()),
                )
