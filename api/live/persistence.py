"""Write-through persistence for live exam sessions.

A student's answers and score live in process for hot-path reads (admin SSE),
but every ~5s we flush dirty sessions to Firestore so a deploy or crash
mid-exam doesn't lose progress.

The flusher takes a `flush_fn` callback so tests can stub Firestore without
mocking. Production wires `make_firestore_flush_fn(get_db)` in
`api/main.py`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from api.live.session import LiveSession, LiveSessionRegistry


FLUSH_INTERVAL_SECONDS = 5.0


FlushFn = Callable[[LiveSession], Awaitable[None]]


class LiveFlusher:
    """Periodically writes dirty live sessions through to durable storage."""

    def __init__(self, registry: LiveSessionRegistry, flush_fn: FlushFn) -> None:
        self._registry = registry
        self._flush_fn = flush_fn
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="live-flusher")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        # Best-effort final flush so a graceful shutdown doesn't drop ~5s of
        # state. We tolerate individual flush failures here — the reaper
        # might already be cancelled and we're racing the event loop close.
        try:
            await self._tick()
        except Exception:
            pass

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[live-flusher] tick error: {e}")

    async def _tick(self) -> None:
        for session in self._registry.iter_all():
            if not session.dirty:
                continue
            # `submitted` is owned by the finalize path (BE-5); don't double-
            # write a doc that the upload route is about to fully overwrite.
            if session.status == "submitted":
                continue
            try:
                await self._flush_fn(session)
            except Exception as e:
                # Stay dirty so we retry on the next tick. Don't let one
                # failed write bring down the loop.
                print(f"[live-flusher] flush failed for {session.email}: {e}")
                continue
            async with session.lock:
                session.dirty = False
                session.last_flushed_at = time.time()


def make_firestore_flush_fn(get_db: Callable[[], object]) -> FlushFn:
    """Build a flush callable that writes to Firestore via `asyncio.to_thread`.

    Uses `merge=True` so a partial live write doesn't clobber any fields the
    final finalize POST writes later. The finalize path (BE-5) will replace
    the doc entirely.
    """

    async def flush(session: LiveSession) -> None:
        # Snapshot under the lock so we don't race a request-handler mutation
        # mid-write.
        async with session.lock:
            payload = {
                "solutions": [
                    {"id": task_id, "state": state}
                    for task_id, state in session.answers.items()
                ],
                "status": session.status,
                "last_updated": int(time.time() * 1000),
            }
            exam_id, password, email = session.exam_id, session.password, session.email

        def write() -> None:
            db = get_db()
            doc = (
                db.collection("solutions")  # type: ignore[attr-defined]
                .document(exam_id)
                .collection(password)
                .document(email)
            )
            doc.set(payload, merge=True)

        await asyncio.to_thread(write)

    return flush
