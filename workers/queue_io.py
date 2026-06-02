"""Shared queue I/O for the API upload handler and the photo-analysis worker.

Design (issue #95):

  Primary path  — event-driven enqueue on upload:
    The API appends to queue-inbox.json immediately after every successful S3
    camera upload.  The worker drains the inbox at the top of each poll cycle.
    This eliminates the need for a full-bucket S3 scan on the hot path.

  Fallback path — prefix-scoped rescan:
    The worker still rescans S3 periodically, but only iterates the exam_id
    prefixes already tracked in the queue (not the whole bucket).  This
    recovers frames that were uploaded while the worker was down or when the
    API enqueue failed.

Both the API and the worker import from this module so the inbox file path and
wire format are always consistent.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
from typing import Any

WORKERS_DIR = Path(__file__).resolve().parent
INBOX_PATH = WORKERS_DIR / "queue-inbox.json"
_LOCK_PATH = WORKERS_DIR / "queue-inbox.lock"

STATUS_PENDING = "PENDING"


@contextlib.contextmanager
def _inbox_lock():
    """Exclusive flock on the inbox lock file (cross-process safe)."""
    with open(_LOCK_PATH, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def enqueue(exam_id: str, email: str, timestamp: int, camera_key: str) -> None:
    """Append a PENDING item to queue-inbox.json after a successful S3 upload.

    Best-effort: if the write fails the worker's prefix-scoped rescan will
    recover the missed frame on the next cycle.
    """
    entry: dict[str, Any] = {
        "exam_id": exam_id,
        "email": email,
        "timestamp": timestamp,
        "camera_key": camera_key,
        "status": STATUS_PENDING,
        "uploadedToFirebase": False,
        "analyzed_at": None,
    }
    try:
        with _inbox_lock():
            try:
                inbox: list = json.loads(INBOX_PATH.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                inbox = []

            key = (exam_id, email, timestamp)
            if not any(
                (e.get("exam_id"), e.get("email"), e.get("timestamp")) == key
                for e in inbox
            ):
                inbox.append(entry)

            tmp = INBOX_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(inbox, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, INBOX_PATH)
    except Exception:
        pass  # non-fatal — rescan recovers


def drain_inbox(queue: list) -> int:
    """Move all inbox entries into queue (deduplicating). Return count added.

    Called by the worker at the top of every poll cycle.  Clears the inbox
    file after merging so each item is processed exactly once.
    """
    if not INBOX_PATH.exists():
        return 0
    try:
        with _inbox_lock():
            try:
                inbox: list = json.loads(INBOX_PATH.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                inbox = []

            seen = {
                (e.get("exam_id"), e.get("email"), e.get("timestamp"))
                for e in queue
            }
            added = 0
            for entry in inbox:
                key = (entry.get("exam_id"), entry.get("email"), entry.get("timestamp"))
                if key not in seen:
                    queue.append(entry)
                    seen.add(key)
                    added += 1

            INBOX_PATH.write_text("[]", encoding="utf-8")
        return added
    except Exception:
        return 0
