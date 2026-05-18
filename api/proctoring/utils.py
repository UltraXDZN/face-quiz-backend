"""Shared helper: enqueue a camera screenshot for deferred proctoring analysis."""
import os
import time
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials


def _get_db():
    if os.getenv("ENVIRONMENT") != "production":
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials(),
        )
    from firebase_admin import firestore as admin_firestore
    return admin_firestore.client()


def enqueue_screenshot(exam_id: str, email: str, timestamp: int, camera_key: str):
    """Enqueue a camera screenshot doc in Firestore. Non-blocking, best-effort."""
    try:
        db = _get_db()
        doc_ref = db.collection("exams").document(exam_id).collection("pending_analysis").document(
            f"{email}_{timestamp}"
        )
        doc_ref.set({
            "exam_id": exam_id,
            "email": email,
            "camera_key": camera_key,
            "timestamp": timestamp,
            "status": "pending",
            "prediction": None,
            "face_count": None,
            "analyzed_at": None,
            "overridden": False,
            "manual_note": "",
        })
    except Exception as e:
        print(f"[PROCTORING] Enqueue failed (non-fatal): {e}")
