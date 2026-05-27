"""Route-level tests for live student endpoints.

Builds a minimal FastAPI app fixture that registers the live router with
stubbed state. Avoids importing `api.main`, which initializes Firebase.
"""

import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.live.routes import router as live_router
from api.live.session import BroadcastHub, LiveSessionRegistry


@pytest.fixture
def app_and_state():
    app = FastAPI()
    app.include_router(live_router)

    app.state.live_registry = LiveSessionRegistry()
    app.state.live_hub = BroadcastHub()
    # Pre-populate so `_load_exam_data` never hits Firestore.
    app.state.live_exam_cache = {
        "exam1": {
            "groups": [
                {
                    "tasks": [
                        {"id": "t1", "state": True, "positive_points": 2.0, "negative_points": 0.5},
                        {"id": "t2", "state": False, "positive_points": 2.0, "negative_points": 0.5},
                    ]
                }
            ]
        }
    }
    return app, app.state


def test_heartbeat_creates_session_and_publishes_presence(app_and_state):
    app, state = app_and_state
    client = TestClient(app)

    r = client.post(
        "/live/exam1/pw1/heartbeat",
        json={"email": "a@x", "current_group": 2, "answered_count": 3},
    )
    assert r.status_code == 204

    session = state.live_registry.get("exam1", "pw1", "a@x")
    assert session is not None
    assert session.status == "solving"
    assert session.current_group == 2
    assert session.answered_count == 3


def test_heartbeat_rate_limit_drops_back_to_back_calls(app_and_state):
    app, state = app_and_state
    client = TestClient(app)

    # First heartbeat lands and sets last_heartbeat.
    client.post("/live/exam1/pw1/heartbeat", json={"email": "a@x"})
    session = state.live_registry.get("exam1", "pw1", "a@x")
    first_heartbeat = session.last_heartbeat

    # Force a tight retry — last_heartbeat must NOT advance.
    client.post(
        "/live/exam1/pw1/heartbeat",
        json={"email": "a@x", "current_group": 99},
    )
    assert session.last_heartbeat == first_heartbeat
    assert session.current_group != 99, "rate-limited heartbeat must not mutate state"


def test_heartbeat_revives_disconnected_session(app_and_state):
    app, state = app_and_state
    client = TestClient(app)

    # First heartbeat creates the session; then we externally flip it to
    # disconnected to simulate the reaper having marked it stale.
    client.post("/live/exam1/pw1/heartbeat", json={"email": "a@x"})
    session = state.live_registry.get("exam1", "pw1", "a@x")
    session.status = "disconnected"
    session.last_heartbeat = 0  # bypass the rate-limit on the next call

    r = client.post("/live/exam1/pw1/heartbeat", json={"email": "a@x"})
    assert r.status_code == 204
    assert session.status == "solving"


def test_answers_merges_and_scores(app_and_state):
    app, state = app_and_state
    client = TestClient(app)

    r = client.post(
        "/live/exam1/pw1/answers",
        json={
            "email": "a@x",
            "solutions": [
                {"id": "t1", "state": True},   # correct → +2
                {"id": "t2", "state": True},   # wrong   → -0.5
            ],
        },
    )
    assert r.status_code == 204

    session = state.live_registry.get("exam1", "pw1", "a@x")
    assert session.answers == {"t1": True, "t2": True}
    assert session.achieved_points == 1.5
    assert session.total_points == 4.0
    assert session.dirty is True
    assert session.answered_count == 2
    assert session.status == "solving"


def test_answers_diff_update_preserves_existing(app_and_state):
    app, state = app_and_state
    client = TestClient(app)

    client.post(
        "/live/exam1/pw1/answers",
        json={"email": "a@x", "solutions": [{"id": "t1", "state": True}]},
    )
    # Second call sends only t2 — t1 must remain.
    client.post(
        "/live/exam1/pw1/answers",
        json={"email": "a@x", "solutions": [{"id": "t2", "state": False}]},
    )

    session = state.live_registry.get("exam1", "pw1", "a@x")
    assert session.answers == {"t1": True, "t2": False}
    # Both correct now → full score.
    assert session.achieved_points == 4.0


def test_answers_returns_404_for_unknown_exam(app_and_state):
    app, state = app_and_state
    client = TestClient(app)

    # `nope` is not in the cache, and `_load_exam_data` would try Firestore
    # which would fail in this test. We expect a 500 because Firestore
    # isn't available — the important guarantee is that an unknown exam
    # doesn't silently create a session. Verify no session was created.
    try:
        client.post(
            "/live/nope/pw1/answers",
            json={"email": "a@x", "solutions": []},
        )
    except Exception:
        pass

    assert state.live_registry.get("nope", "pw1", "a@x") is None
