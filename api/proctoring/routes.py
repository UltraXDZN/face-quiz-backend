"""Proctoring photo-analysis API.

Reads written by the standalone photo-analysis worker
(workers/photo_analysis_worker.py) into Firestore at:

    exams/{exam_id}/photoAnalysis/{email}_{timestamp}

All endpoints are admin-only (enforced by the frontend `admin` middleware
on every page that calls them — there is no per-route auth dependency
elsewhere in this codebase, so we follow the existing pattern).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import boto3
from botocore.config import Config
from fastapi import APIRouter, HTTPException, Query
from google.auth.credentials import AnonymousCredentials
from google.cloud import firestore

router = APIRouter(prefix="/proctoring", tags=["proctoring"])

_BOTO_CONFIG = Config(connect_timeout=3, read_timeout=10, retries={"max_attempts": 1})
_s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
    config=_BOTO_CONFIG,
)
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "face-quiz-media")
PHOTO_ANALYSIS_COLLECTION = "photoAnalysis"

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
WORKERS_DIR = BACKEND_ROOT / "workers"
QUEUE_PATH = WORKERS_DIR / "queue.json"
STATE_PATH = WORKERS_DIR / "worker-state.json"
LOG_PATH = WORKERS_DIR / "worker.log"
LOG_TAIL_BYTES = 4096


def _get_db():
    if os.getenv("ENVIRONMENT") != "production":
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials(),
        )
    from firebase_admin import firestore as admin_firestore
    return admin_firestore.client()


def _presigned_url(key: str) -> str | None:
    try:
        return _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=3600,
        )
    except Exception:
        return None


@router.get("/photo-analysis")
def get_photo_analysis(exam_id: str, email: str | None = Query(None)):
    """Get analyzed photos for an exam, optionally filtered by student email.

    Returns one document per analyzed camera frame. The frontend reads this
    once per (exam, user) load and keys results by `${email}_${timestamp}`.
    """
    db = _get_db()
    coll = db.collection("exams").document(exam_id).collection(PHOTO_ANALYSIS_COLLECTION)
    query = coll.where("email", "==", email) if email else coll

    results = []
    for doc in query.stream():
        d = doc.to_dict() or {}
        d["id"] = doc.id
        camera_key = d.get("camera_key")
        if not camera_key:
            ts = d.get("timestamp")
            doc_email = d.get("email")
            if ts is not None and doc_email:
                safe_email = doc_email.replace("@", "_at_").replace(".", "_")
                camera_key = f"{exam_id}/{safe_email}/{ts}/camera.png"
        d["camera_key"] = camera_key
        d["camera_url"] = _presigned_url(camera_key) if camera_key else None
        results.append(d)
    return results


ALLOWED_OVERRIDE_STATUSES = {"FACE_DETECTED", "MULTIPLE_FACES_DETECTED", "NO_FACE_DETECTED"}


@router.put("/photo-analysis/{exam_id}/{doc_id}/override")
def override_photo_analysis(
    exam_id: str,
    doc_id: str,
    status: str = Query(..., description="One of FACE_DETECTED|MULTIPLE_FACES_DETECTED|NO_FACE_DETECTED"),
    note: str | None = Query(None),
):
    """Admin override of the AI prediction. Stored as overriddenStatus so the
    original AI status remains for audit."""
    if status not in ALLOWED_OVERRIDE_STATUSES:
        raise HTTPException(status_code=400, detail=f"status must be one of {sorted(ALLOWED_OVERRIDE_STATUSES)}")
    db = _get_db()
    ref = db.collection("exams").document(exam_id).collection(PHOTO_ANALYSIS_COLLECTION).document(doc_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Analysis not found")
    ref.update({
        "overridden": True,
        "overriddenStatus": status,
        "manual_note": note or "",
        "overriddenAt": int(time.time() * 1000),
    })
    return {"status": "overridden", "overriddenStatus": status}


@router.delete("/photo-analysis/{exam_id}/{doc_id}/override")
def clear_photo_analysis_override(exam_id: str, doc_id: str):
    """Remove the admin override; effective status falls back to the AI prediction."""
    db = _get_db()
    ref = db.collection("exams").document(exam_id).collection(PHOTO_ANALYSIS_COLLECTION).document(doc_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Analysis not found")
    ref.update({
        "overridden": False,
        "overriddenStatus": firestore.DELETE_FIELD,
        "manual_note": "",
        "overriddenAt": firestore.DELETE_FIELD,
    })
    return {"status": "cleared"}


@router.get("/worker-status")
def get_worker_status():
    """Snapshot of the background worker for the admin dashboard.

    Returns queue depth, last-batch info, request counter, and a small
    tail of worker.log so admins can spot crashes without shelling in.
    """
    queue: list = []
    if QUEUE_PATH.exists():
        try:
            with open(QUEUE_PATH, "r", encoding="utf-8") as f:
                queue = json.load(f)
        except (json.JSONDecodeError, OSError):
            queue = []

    state: dict = {}
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            state = {}

    status_counts: dict[str, int] = {}
    last_analyzed_ts: int | None = None
    pending_uploads = 0
    for entry in queue:
        status = entry.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
        if entry.get("status") != "PENDING" and not entry.get("uploadedToFirebase"):
            pending_uploads += 1
        analyzed_at = entry.get("analyzed_at")
        if analyzed_at and (last_analyzed_ts is None or analyzed_at > last_analyzed_ts):
            last_analyzed_ts = analyzed_at

    log_tail = ""
    if LOG_PATH.exists():
        try:
            size = LOG_PATH.stat().st_size
            with open(LOG_PATH, "rb") as f:
                f.seek(max(0, size - LOG_TAIL_BYTES))
                log_tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            log_tail = ""

    return {
        "queueDepth": len(queue),
        "statusCounts": status_counts,
        "pendingUploads": pending_uploads,
        "lastAnalyzedAt": last_analyzed_ts,
        "requestCount": int(state.get("request_count", 0)),
        "logTail": log_tail,
        "queueFileExists": QUEUE_PATH.exists(),
    }
