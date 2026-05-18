"""Background worker that processes pending screenshot analysis during idle periods."""
import asyncio
import os
import time
import io
import cv2
import numpy as np
from typing import Optional
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials

from api.media.face_detection import detect_faces

# S3
import boto3

_s3_client = None

def s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
    return _s3_client


def get_db():
    if os.getenv("ENVIRONMENT") != "production":
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials(),
        )
    from firebase_admin import firestore as admin_firestore
    return admin_firestore.client()


BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "face-quiz-media")

# Load threshold
CHECK_INTERVAL = float(os.getenv("PROCTORING_CHECK_INTERVAL", "15"))
MAX_LOAD = float(os.getenv("PROCTORING_MAX_LOAD", "1.0"))
BATCH_SIZE = int(os.getenv("PROCTORING_BATCH_SIZE", "10"))

_running = False


def _is_idle() -> bool:
    """Return True if server load is below threshold."""
    try:
        load1 = os.getloadavg()[0]
        return load1 < MAX_LOAD
    except Exception:
        return True


async def _process_one(doc_ref, doc_data: dict):
    try:
        camera_key = doc_data["camera_key"]
        response = s3().get_object(Bucket=BUCKET_NAME, Key=camera_key)
        image_bytes = response["Body"].read()
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            doc_ref.update({
                "status": "analyzed",
                "prediction": "error",
                "face_count": 0,
                "analyzed_at": int(time.time() * 1000),
            })
            return

        face_count, _bboxes = detect_faces(frame)
        if face_count == 1:
            pred = "single"
        elif face_count > 1:
            pred = "multiple"
        else:
            pred = "none"

        doc_ref.update({
            "status": "analyzed",
            "prediction": pred,
            "face_count": face_count,
            "analyzed_at": int(time.time() * 1000),
        })
        print(f"[PROCTORING] Analyzed {camera_key}: {pred}")
    except Exception as e:
        print(f"[PROCTORING] Error processing {doc_data.get('camera_key')}: {e}")


async def worker_loop():
    global _running
    _running = True
    print("[PROCTORING] Worker started")
    db = get_db()

    while _running:
        try:
            if not _is_idle():
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            coll = db.collection("exams")
            pending = []
            for exam_doc in coll.stream():
                analysis_coll = exam_doc.reference.collection("pending_analysis")
                query = analysis_coll.where("status", "==", "pending").order_by("timestamp").limit(BATCH_SIZE)
                for qdoc in query.stream():
                    pending.append((exam_doc.id, qdoc.reference, qdoc.to_dict()))

            if not pending:
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            print(f"[PROCTORING] Processing {min(len(pending), BATCH_SIZE)} pending screenshots")
            for exam_id, doc_ref, doc_data in pending[:BATCH_SIZE]:
                await _process_one(doc_ref, doc_data)

        except Exception as e:
            print(f"[PROCTORING] Worker error: {e}")

        await asyncio.sleep(CHECK_INTERVAL)

    print("[PROCTORING] Worker stopped")
