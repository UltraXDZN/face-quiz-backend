"""Standalone photo-analysis worker.

Runs as a separate OS process (subprocess.Popen target). Polls system load
via psutil + an API request-rate counter shared through worker-state.json,
and only analyzes pending camera frames when the host is near-idle.

Queue lives locally in queue.json. Firestore is written in batched commits
when the worker is about to stop processing (idle → busy transition or
no PENDING items left).

Run standalone:
    python workers/photo_analysis_worker.py
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Make sibling packages importable when run standalone from any cwd.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

# In production, scrub any emulator env vars BEFORE importing google.cloud.firestore.
# The Firestore SDK auto-detects FIRESTORE_EMULATOR_HOST and routes there no matter
# how you build the client, so leaving it set (because .env defines it) silently
# bypasses production credentials. Mirrors api/main.py's startup logic.
if os.getenv("ENVIRONMENT") == "production":
    for var in ("FIRESTORE_EMULATOR_HOST", "FIREBASE_AUTH_EMULATOR_HOST"):
        os.environ.pop(var, None)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import boto3  # noqa: E402
from google.cloud import firestore  # noqa: E402
from google.auth.credentials import AnonymousCredentials  # noqa: E402
from workers.queue_io import drain_inbox  # noqa: E402

# --------------------------------------------------------------------------- #
# Tunables — all named constants, no magic numbers.
# --------------------------------------------------------------------------- #
CPU_IDLE_THRESHOLD = 10.0  # percent (1-second average)
MEMORY_INCREASE_THRESHOLD = 5.0  # percent above baseline captured at startup
REQUEST_IDLE_THRESHOLD = 1  # requests per minute

POLL_INTERVAL_SECONDS = 10
CPU_SAMPLE_SECONDS = 1.0  # psutil.cpu_percent() blocking interval
BASELINE_SETTLE_SECONDS = 2.0  # let process settle before sampling baseline

S3_RESCAN_EVERY_N_POLLS = 6  # ~ every minute when idle
FIRESTORE_BATCH_LIMIT = 500  # Firestore hard cap is 500 writes/batch
SHUTDOWN_GRACE_SECONDS = 30

WORKERS_DIR = BACKEND_ROOT / "workers"
QUEUE_PATH = WORKERS_DIR / "queue.json"
STATE_PATH = WORKERS_DIR / "worker-state.json"
LOG_PATH = WORKERS_DIR / "worker.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "face-quiz-media")
PHOTO_ANALYSIS_COLLECTION = "photoAnalysis"

STATUS_PENDING = "PENDING"
STATUS_FACE = "FACE_DETECTED"
STATUS_MULTIPLE = "MULTIPLE_FACES_DETECTED"
STATUS_NONE = "NO_FACE_DETECTED"
TERMINAL_STATUSES = {STATUS_FACE, STATUS_MULTIPLE, STATUS_NONE}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def _build_logger() -> logging.Logger:
    WORKERS_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("photo_analysis_worker")
    log.setLevel(logging.INFO)
    log.propagate = False
    if log.handlers:
        return log
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = RotatingFileHandler(
        LOG_PATH, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


logger = _build_logger()


# --------------------------------------------------------------------------- #
# Atomic JSON I/O
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s (%s); renaming and using default", path, e)
        try:
            path.rename(path.with_suffix(path.suffix + f".broken-{int(time.time())}"))
        except OSError:
            pass
        return default


# --------------------------------------------------------------------------- #
# External clients (S3 + Firestore)
# --------------------------------------------------------------------------- #
def _get_s3():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )


def _get_db():
    if os.getenv("ENVIRONMENT") != "production":
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials(),
        )
    import firebase_admin
    from firebase_admin import credentials, firestore as admin_firestore

    if not firebase_admin._apps:
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    return admin_firestore.client()


# --------------------------------------------------------------------------- #
# Vision wrapper — uses the existing detector, do not duplicate logic.
# --------------------------------------------------------------------------- #
def _classify(frame: np.ndarray) -> str:
    from api.media.face_detection import detect_faces

    count, _ = detect_faces(frame)
    if count == 1:
        return STATUS_FACE
    if count > 1:
        return STATUS_MULTIPLE
    return STATUS_NONE


# --------------------------------------------------------------------------- #
# Load monitoring
# --------------------------------------------------------------------------- #
class LoadMonitor:
    def __init__(self) -> None:
        time.sleep(BASELINE_SETTLE_SECONDS)
        self.baseline_mem_pct = psutil.virtual_memory().percent
        # Prime the cpu_percent counter (first call returns 0.0).
        psutil.cpu_percent(interval=None)
        self._last_count: int | None = None
        self._last_time: float | None = None

    def _request_rate(self) -> float:
        state = _read_json(STATE_PATH, {"request_count": 0})
        count = int(state.get("request_count", 0))
        now = time.time()
        if self._last_count is None or self._last_time is None:
            self._last_count, self._last_time = count, now
            return 0.0
        delta = max(0, count - self._last_count)
        elapsed = max(1e-3, now - self._last_time)
        self._last_count, self._last_time = count, now
        return (delta / elapsed) * 60.0

    def sample(self) -> tuple[bool, float, float, float]:
        cpu_pct = psutil.cpu_percent(interval=CPU_SAMPLE_SECONDS)
        mem_pct_delta = psutil.virtual_memory().percent - self.baseline_mem_pct
        req_rate = self._request_rate()
        idle = (
            cpu_pct < CPU_IDLE_THRESHOLD
            and mem_pct_delta < MEMORY_INCREASE_THRESHOLD
            and req_rate < REQUEST_IDLE_THRESHOLD
        )
        return idle, cpu_pct, mem_pct_delta, req_rate


# --------------------------------------------------------------------------- #
# Queue management
# --------------------------------------------------------------------------- #
def _entry_key(exam_id: str, email: str, timestamp: int) -> str:
    return f"{exam_id}|{email}|{timestamp}"


def _safe_email(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_")


def _email_from_safe(safe_email: str) -> str:
    return safe_email.replace("_at_", "@").replace("_", ".")


def _list_s3_camera_keys(s3) -> list[dict]:
    """List every {exam_id}/{safe_email}/{timestamp}/camera.png in the bucket."""
    out: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")
            if len(parts) != 4 or parts[-1] != "camera.png":
                continue
            try:
                timestamp = int(parts[2])
            except ValueError:
                continue
            out.append(
                {
                    "exam_id": parts[0],
                    "email": _email_from_safe(parts[1]),
                    "timestamp": timestamp,
                    "camera_key": key,
                }
            )
    return out


def _fetch_existing_analysis(db, exam_id: str) -> dict[str, dict]:
    """Return {doc_id: doc_dict} for every analyzed frame already in Firestore for this exam."""
    out: dict[str, dict] = {}
    coll = (
        db.collection("exams").document(exam_id).collection(PHOTO_ANALYSIS_COLLECTION)
    )
    for doc in coll.stream():
        data = doc.to_dict() or {}
        if data.get("status") in TERMINAL_STATUSES:
            out[doc.id] = data
    return out


def _bootstrap_queue(s3, db) -> list[dict]:
    """First-run only: scan S3 + Firestore to build the initial queue.

    Optimised: instead of one .get() per frame (which was 4000+ sequential
    RPCs for a year's worth of exams), we do one .stream() per distinct exam
    and check set membership in memory.
    """
    logger.info("BOOTSTRAP starting (no queue.json found)")
    discovered = _list_s3_camera_keys(s3)
    logger.info("BOOTSTRAP discovered %d camera frames in S3", len(discovered))

    by_exam: dict[str, list[dict]] = {}
    for item in discovered:
        by_exam.setdefault(item["exam_id"], []).append(item)

    queue: list[dict] = []
    for exam_id, items in by_exam.items():
        try:
            existing = _fetch_existing_analysis(db, exam_id)
        except Exception as e:
            logger.warning(
                "BOOTSTRAP exam=%s failed to fetch existing analysis (%s); treating all as pending",
                exam_id,
                e,
            )
            existing = {}
        for item in items:
            doc_id = f"{item['email']}_{item['timestamp']}"
            prior = existing.get(doc_id)
            if prior:
                queue.append(
                    {
                        **item,
                        "status": prior["status"],
                        "uploadedToFirebase": True,
                        "analyzed_at": prior.get("analyzedAt"),
                    }
                )
            else:
                queue.append(
                    {
                        **item,
                        "status": STATUS_PENDING,
                        "uploadedToFirebase": False,
                        "analyzed_at": None,
                    }
                )
        logger.info(
            "BOOTSTRAP exam=%s frames=%d already_analyzed=%d",
            exam_id,
            len(items),
            sum(1 for i in items if f"{i['email']}_{i['timestamp']}" in existing),
        )

    logger.info(
        "BOOTSTRAP queue built: %d entries (%d pending) across %d exam(s)",
        len(queue),
        sum(1 for e in queue if e["status"] == STATUS_PENDING),
        len(by_exam),
    )
    return queue


def _load_or_bootstrap_queue(s3, db) -> list[dict]:
    if QUEUE_PATH.exists():
        data = _read_json(QUEUE_PATH, [])
        if isinstance(data, list):
            return data
    queue = _bootstrap_queue(s3, db)
    _atomic_write_json(QUEUE_PATH, queue)
    return queue


def _rescan_by_known_exams(queue: list[dict], s3) -> int:
    """Fallback rescan: scan S3 only for exam_ids already tracked in the queue.

    This is a safety net for frames uploaded while the worker was down or when
    the API's event-driven enqueue failed.  It is NOT a full-bucket scan — it
    issues one prefix-scoped list call per known exam_id rather than listing the
    whole bucket.
    """
    exam_ids = {e["exam_id"] for e in queue}
    if not exam_ids:
        return 0
    seen = {(e["exam_id"], e["email"], e["timestamp"]) for e in queue}
    added = 0
    paginator = s3.get_paginator("list_objects_v2")
    for exam_id in exam_ids:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{exam_id}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                parts = key.split("/")
                if len(parts) != 4 or parts[-1] != "camera.png":
                    continue
                try:
                    timestamp = int(parts[2])
                except ValueError:
                    continue
                item_key = (parts[0], _email_from_safe(parts[1]), timestamp)
                if item_key in seen:
                    continue
                queue.append(
                    {
                        "exam_id": parts[0],
                        "email": _email_from_safe(parts[1]),
                        "timestamp": timestamp,
                        "camera_key": key,
                        "status": STATUS_PENDING,
                        "uploadedToFirebase": False,
                        "analyzed_at": None,
                    }
                )
                seen.add(item_key)
                added += 1
    if added:
        _atomic_write_json(QUEUE_PATH, queue)
        logger.info("RESCAN added %d new pending entries", added)
    return added


# --------------------------------------------------------------------------- #
# Analysis + batch flush
# --------------------------------------------------------------------------- #
def _analyze_one(entry: dict, s3) -> None:
    """Classify one frame and update the entry in-memory.

    The queue is NOT written here; the caller writes it once after the full
    batch completes (issue #95: avoid rewriting queue per item).
    """
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=entry["camera_key"])
        body = resp["Body"].read()
        frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            status = STATUS_NONE
        else:
            status = _classify(frame)
    except Exception as e:
        logger.exception(
            "ANALYZE_ERROR exam=%s email=%s ts=%s err=%s",
            entry["exam_id"],
            entry["email"],
            entry["timestamp"],
            e,
        )
        return
    entry["status"] = status
    entry["analyzed_at"] = int(time.time() * 1000)
    entry["uploadedToFirebase"] = False
    logger.info(
        "ANALYZED exam=%s email=%s ts=%s status=%s",
        entry["exam_id"],
        entry["email"],
        entry["timestamp"],
        status,
    )


def _flush_batch(queue: list[dict], db) -> int:
    items = [
        e
        for e in queue
        if e["status"] in TERMINAL_STATUSES and not e["uploadedToFirebase"]
    ]
    if not items:
        return 0
    written = 0
    for start in range(0, len(items), FIRESTORE_BATCH_LIMIT):
        chunk = items[start : start + FIRESTORE_BATCH_LIMIT]
        batch = db.batch()
        for item in chunk:
            ref = (
                db.collection("exams")
                .document(item["exam_id"])
                .collection(PHOTO_ANALYSIS_COLLECTION)
                .document(f"{item['email']}_{item['timestamp']}")
            )
            batch.set(
                ref,
                {
                    "exam_id": item["exam_id"],
                    "email": item["email"],
                    "timestamp": item["timestamp"],
                    "status": item["status"],
                    "analyzedAt": item["analyzed_at"] or int(time.time() * 1000),
                },
            )
        batch.commit()
        for item in chunk:
            item["uploadedToFirebase"] = True
        written += len(chunk)
    _atomic_write_json(QUEUE_PATH, queue)
    logger.info("BATCH_WRITE items=%d", written)
    return written


# --------------------------------------------------------------------------- #
# Main loop + signal handling
# --------------------------------------------------------------------------- #
_running = True


def _handle_sigterm(signum, _frame):
    global _running
    logger.info("SIGNAL received signal=%s; will exit after current cycle", signum)
    _running = False


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    logger.info(
        "STARTUP pid=%d cwd=%s bucket=%s env=%s",
        os.getpid(),
        os.getcwd(),
        S3_BUCKET,
        os.getenv("ENVIRONMENT", "development"),
    )

    s3 = _get_s3()
    db = _get_db()
    monitor = LoadMonitor()
    logger.info("BASELINE mem=%.1f%%", monitor.baseline_mem_pct)

    queue = _load_or_bootstrap_queue(s3, db)
    poll_count = 0

    while _running:
        try:
            poll_count += 1

            # Primary path: drain frames enqueued by the API on upload.
            added = drain_inbox(queue)
            if added:
                logger.info("INBOX_DRAIN added=%d", added)

            # Fallback: prefix-scoped rescan for frames missed by the event path.
            if poll_count % S3_RESCAN_EVERY_N_POLLS == 0:
                try:
                    _rescan_by_known_exams(queue, s3)
                except Exception as e:
                    logger.warning("RESCAN_ERROR err=%s", e)

            idle, cpu, mem_d, req_rate = monitor.sample()

            if poll_count % 60 == 0:
                logger.info(
                    "LOAD cpu=%.1f%% mem=+%.1f%% req_rate=%.2f/min idle=%s",
                    cpu,
                    mem_d,
                    req_rate,
                    idle,
                )

            if not idle:
                _flush_batch(queue, db)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            pending = [e for e in queue if e["status"] == STATUS_PENDING]
            if not pending:
                _flush_batch(queue, db)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info("ANALYZE_BEGIN n_pending=%d", len(pending))
            processed = 0
            stop_reason = "no_pending"
            for entry in pending:
                if not _running:
                    stop_reason = "shutdown"
                    break
                idle_now, *_ = monitor.sample()
                if not idle_now:
                    stop_reason = "busy"
                    break
                _analyze_one(entry, s3)
                processed += 1
            logger.info("ANALYZE_STOP reason=%s processed=%d", stop_reason, processed)

            # Write queue once per batch, not once per analyzed item (issue #95).
            if processed:
                _atomic_write_json(QUEUE_PATH, queue)
            _flush_batch(queue, db)
            time.sleep(POLL_INTERVAL_SECONDS)

        except Exception as e:
            logger.exception("LOOP_ERROR err=%s", e)
            time.sleep(POLL_INTERVAL_SECONDS)

    flushed = _flush_batch(queue, db)
    logger.info("SHUTDOWN flushed=%d", flushed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
