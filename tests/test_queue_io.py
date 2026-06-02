"""Tests for issue #95: event-driven enqueue and batch queue writes."""

from __future__ import annotations

import json
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _entry(exam_id="e1", email="a@b.com", ts=100, camera_key="k"):
    return {
        "exam_id": exam_id, "email": email, "timestamp": ts,
        "camera_key": camera_key, "status": "PENDING",
        "uploadedToFirebase": False, "analyzed_at": None,
    }


# ── queue_io.enqueue ──────────────────────────────────────────────────────────

def test_enqueue_creates_inbox(tmp_path, monkeypatch):
    """enqueue() creates queue-inbox.json with a PENDING entry."""
    from workers import queue_io
    monkeypatch.setattr(queue_io, "INBOX_PATH", tmp_path / "queue-inbox.json")
    monkeypatch.setattr(queue_io, "_LOCK_PATH", tmp_path / "queue-inbox.lock")

    queue_io.enqueue("e1", "a@b.com", 100, "k")

    inbox = json.loads((tmp_path / "queue-inbox.json").read_text())
    assert len(inbox) == 1
    assert inbox[0]["exam_id"] == "e1"
    assert inbox[0]["status"] == "PENDING"
    assert inbox[0]["uploadedToFirebase"] is False


def test_enqueue_appends_to_existing_inbox(tmp_path, monkeypatch):
    """enqueue() appends to an existing inbox without losing prior entries."""
    from workers import queue_io
    monkeypatch.setattr(queue_io, "INBOX_PATH", tmp_path / "queue-inbox.json")
    monkeypatch.setattr(queue_io, "_LOCK_PATH", tmp_path / "queue-inbox.lock")

    queue_io.enqueue("e1", "a@b.com", 100, "k1")
    queue_io.enqueue("e1", "a@b.com", 200, "k2")

    inbox = json.loads((tmp_path / "queue-inbox.json").read_text())
    assert len(inbox) == 2


def test_enqueue_deduplicates(tmp_path, monkeypatch):
    """enqueue() silently skips a duplicate (same exam/email/timestamp)."""
    from workers import queue_io
    monkeypatch.setattr(queue_io, "INBOX_PATH", tmp_path / "queue-inbox.json")
    monkeypatch.setattr(queue_io, "_LOCK_PATH", tmp_path / "queue-inbox.lock")

    queue_io.enqueue("e1", "a@b.com", 100, "k")
    queue_io.enqueue("e1", "a@b.com", 100, "k")  # duplicate

    inbox = json.loads((tmp_path / "queue-inbox.json").read_text())
    assert len(inbox) == 1


# ── queue_io.drain_inbox ──────────────────────────────────────────────────────

def test_drain_inbox_moves_entries_to_queue(tmp_path, monkeypatch):
    """drain_inbox() transfers inbox items into the queue."""
    from workers import queue_io
    monkeypatch.setattr(queue_io, "INBOX_PATH", tmp_path / "queue-inbox.json")
    monkeypatch.setattr(queue_io, "_LOCK_PATH", tmp_path / "queue-inbox.lock")

    (tmp_path / "queue-inbox.json").write_text(json.dumps([_entry()]))

    queue: list = []
    added = queue_io.drain_inbox(queue)

    assert added == 1
    assert len(queue) == 1
    assert queue[0]["exam_id"] == "e1"


def test_drain_inbox_clears_inbox_after_merge(tmp_path, monkeypatch):
    """drain_inbox() empties the inbox file after merging."""
    from workers import queue_io
    monkeypatch.setattr(queue_io, "INBOX_PATH", tmp_path / "queue-inbox.json")
    monkeypatch.setattr(queue_io, "_LOCK_PATH", tmp_path / "queue-inbox.lock")

    (tmp_path / "queue-inbox.json").write_text(json.dumps([_entry()]))
    queue_io.drain_inbox([])

    remaining = json.loads((tmp_path / "queue-inbox.json").read_text())
    assert remaining == []


def test_drain_inbox_deduplicates_against_queue(tmp_path, monkeypatch):
    """drain_inbox() does not add items already present in the queue."""
    from workers import queue_io
    monkeypatch.setattr(queue_io, "INBOX_PATH", tmp_path / "queue-inbox.json")
    monkeypatch.setattr(queue_io, "_LOCK_PATH", tmp_path / "queue-inbox.lock")

    (tmp_path / "queue-inbox.json").write_text(json.dumps([_entry()]))
    queue = [_entry()]  # same item already in queue

    added = queue_io.drain_inbox(queue)

    assert added == 0
    assert len(queue) == 1  # no duplicate


def test_drain_inbox_noop_when_missing(tmp_path, monkeypatch):
    """drain_inbox() returns 0 when the inbox file does not exist."""
    from workers import queue_io
    monkeypatch.setattr(queue_io, "INBOX_PATH", tmp_path / "no-inbox.json")
    monkeypatch.setattr(queue_io, "_LOCK_PATH", tmp_path / "queue-inbox.lock")

    queue: list = []
    assert queue_io.drain_inbox(queue) == 0
    assert queue == []


# ── _analyze_one: no per-item queue write ────────────────────────────────────

def test_analyze_one_does_not_write_queue(monkeypatch):
    """_analyze_one must not call _atomic_write_json (batch write only)."""
    import workers.photo_analysis_worker as w

    write_calls: list = []
    monkeypatch.setattr(w, "_atomic_write_json", lambda path, data: write_calls.append(path))

    # Stub: S3 returns empty bytes → imdecode returns None → STATUS_NONE
    class _FakeBody:
        def read(self): return b""

    class _FakeS3:
        def get_object(self, **_): return {"Body": _FakeBody()}

    import cv2
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **kw: None)

    entry = _entry()
    w._analyze_one(entry, _FakeS3())

    assert not write_calls, "_analyze_one must not write queue.json"
    assert entry["status"] == w.STATUS_NONE  # in-memory update still happens
