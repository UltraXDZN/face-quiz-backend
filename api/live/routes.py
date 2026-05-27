"""HTTP routes for live exam monitoring (student write side).

Student endpoints:
  - POST /live/{exam_id}/{password}/heartbeat
  - POST /live/{exam_id}/{password}/answers

Admin read endpoints (the SSE stream) land in a separate file in BE-3.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.live.scoring import compute_live_score
from api.live.session import BroadcastEvent


router = APIRouter(prefix="/live", tags=["live"])


HEARTBEAT_MIN_INTERVAL_SECONDS = 1.0
# How long to wait on the broadcast queue before emitting a keep-alive
# comment. Has to be shorter than any proxy idle timeout in front of us
# (nginx default is 60s; we keep a comfortable margin).
SSE_KEEPALIVE_SECONDS = 15.0


class HeartbeatRequest(BaseModel):
    email: str
    current_group: int = 1
    answered_count: int = 0


class AnswerEntry(BaseModel):
    id: str
    state: Optional[bool] = None


class AnswerUpdateRequest(BaseModel):
    email: str
    solutions: list[AnswerEntry]


async def _load_exam_data(request: Request, exam_id: str) -> dict:
    """Return the exam doc, lazy-loading once per app lifetime.

    Cache lives on `app.state.live_exam_cache`; first miss issues a synchronous
    Firestore read via `asyncio.to_thread`.
    """
    cache: dict[str, dict] = request.app.state.live_exam_cache
    cached = cache.get(exam_id)
    if cached is not None:
        return cached

    # Imported here to avoid pulling the solutions module (and its Firestore
    # client init) at import time for unit tests that don't need it.
    from api.solutions.routes import get_db

    def fetch() -> dict | None:
        db = get_db()
        doc = db.collection("exams").document(exam_id).get()
        return doc.to_dict() if doc.exists else None

    exam_data = await asyncio.to_thread(fetch)
    if exam_data is None:
        raise HTTPException(status_code=404, detail="Exam not found")
    cache[exam_id] = exam_data
    return exam_data


@router.post("/{exam_id}/{password}/heartbeat", status_code=status.HTTP_204_NO_CONTENT)
async def heartbeat(
    exam_id: str, password: str, body: HeartbeatRequest, request: Request
) -> None:
    """Mark a student as currently solving and refresh their `last_seen`.

    Idempotent and rate-limited (~1/sec/email) against client bugs. A
    heartbeat on a `disconnected` session revives it back to `solving`.
    """
    registry = request.app.state.live_registry
    hub = request.app.state.live_hub
    session = await registry.get_or_create(exam_id, password, body.email)

    publish_presence = False
    async with session.lock:
        now = time.time()
        # Rate-limit: drop heartbeats that arrive before the minimum
        # interval has elapsed. We still 204 the client; this is a noop, not
        # an error. Uses `last_heartbeat` (not `last_seen`) so a freshly
        # created session's default last_seen doesn't spuriously trip this.
        if (
            session.last_heartbeat > 0
            and now - session.last_heartbeat < HEARTBEAT_MIN_INTERVAL_SECONDS
        ):
            return
        session.last_seen = now
        session.last_heartbeat = now
        session.status = "solving"
        session.current_group = body.current_group
        session.answered_count = body.answered_count
        publish_presence = True

    if publish_presence:
        await hub.publish(
            exam_id,
            password,
            BroadcastEvent(type="presence", data=session.presence_payload()),
        )


@router.post("/{exam_id}/{password}/answers", status_code=status.HTTP_204_NO_CONTENT)
async def update_answers(
    exam_id: str, password: str, body: AnswerUpdateRequest, request: Request
) -> None:
    """Merge a diff of answer changes into the student's live session.

    Scoring uses the same points-based formula as `_calculate_user_results`
    via `compute_live_score`, so admin live scores match the final saved
    score. The actual Firestore write is debounced — the flusher background
    task picks up dirty sessions every ~5s.
    """
    registry = request.app.state.live_registry
    hub = request.app.state.live_hub

    exam_data = await _load_exam_data(request, exam_id)
    session = await registry.get_or_create(exam_id, password, body.email)

    async with session.lock:
        for entry in body.solutions:
            session.answers[entry.id] = entry.state
        achieved, total = compute_live_score(exam_data, session.answers)
        session.achieved_points = achieved
        session.total_points = total
        session.last_seen = time.time()
        # Revive a session the reaper had flipped to disconnected — answers
        # imply the student is still here.
        session.status = "solving"
        session.answered_count = sum(
            1 for v in session.answers.values() if v is not None
        )
        session.dirty = True
        score_payload = session.score_payload()
        presence_payload = session.presence_payload()

    await hub.publish(
        exam_id, password, BroadcastEvent(type="score", data=score_payload)
    )
    await hub.publish(
        exam_id, password, BroadcastEvent(type="presence", data=presence_payload)
    )


def _format_sse(event_type: str, data: dict) -> str:
    """Encode an event for the SSE wire format."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def build_event_stream(
    registry,
    hub,
    exam_id: str,
    password: str,
    is_disconnected,
    keepalive_seconds: float = SSE_KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """Yield SSE-formatted lines for the (exam, password) channel.

    Factored out of the route handler so it can be driven directly from
    tests without going through `TestClient.stream` (which doesn't tear
    down a long-lived SSE response cleanly).

    `is_disconnected` is an async callable returning `True` when the
    client has gone away. In production this is `request.is_disconnected`;
    tests pass a fake.
    """
    queue = await hub.subscribe(exam_id, password)
    try:
        # Snapshot — send the current state of every existing session so a
        # late-joining admin doesn't have to wait for the next mutation.
        for session in registry.list_for(exam_id, password):
            yield _format_sse("presence", session.presence_payload())
            yield _format_sse("score", session.score_payload())

        while True:
            if await is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
                yield _format_sse(event.type, event.data)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
    finally:
        await hub.unsubscribe(exam_id, password, queue)


@router.get("/{exam_id}/{password}/stream")
async def admin_stream(exam_id: str, password: str, request: Request):
    """SSE stream of live presence + score events for the (exam, password) channel.

    Admin-only by intent; like the rest of this backend (see
    `api/proctoring/routes.py`) the gate is enforced by the frontend admin
    middleware. Worth tightening server-side eventually, but out of scope
    here.
    """
    return StreamingResponse(
        build_event_stream(
            request.app.state.live_registry,
            request.app.state.live_hub,
            exam_id,
            password,
            request.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable nginx response buffering so events are flushed
            # immediately on this stream.
            "X-Accel-Buffering": "no",
        },
    )
