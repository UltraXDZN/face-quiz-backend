"""Proctoring analysis API — enqueue, fetch, override."""
import os
import time
import boto3
from fastapi import APIRouter, HTTPException, Query
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials

router = APIRouter(prefix="/proctoring", tags=["proctoring"])

_s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "face-quiz-media")


def get_db():
    if os.getenv("ENVIRONMENT") != "production":
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials(),
        )
    from firebase_admin import firestore as admin_firestore
    return admin_firestore.client()


@router.post("/enqueue")
async def enqueue_screenshot(exam_id: str, email: str, timestamp: int, camera_key: str):
    """Enqueue a camera screenshot for deferred analysis."""
    db = get_db()
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
    return {"status": "enqueued", "doc_id": doc_ref.id}


@router.get("/analysis")
async def get_analysis(exam_id: str, email: str = Query(None)):
    """Get analyzed (or pending) screenshots for an exam."""
    db = get_db()
    docs = []
    coll = db.collection("exams").document(exam_id).collection("pending_analysis")
    if email:
        query = coll.where("email", "==", email).order_by("timestamp")
    else:
        query = coll.order_by("timestamp")
    for doc in query.stream():
        d = doc.to_dict()
        d["id"] = doc.id
        # Generate presigned URL for camera image
        try:
            url = _s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": d["camera_key"]},
                ExpiresIn=3600,
            )
            d["camera_url"] = url
        except Exception:
            d["camera_url"] = None
        docs.append(d)
    return docs


@router.put("/analysis/{exam_id}/{doc_id}/override")
async def override_analysis(exam_id: str, doc_id: str, note: str = Query(None)):
    """Mark a specific analysis as overridden (manual OK)."""
    db = get_db()
    doc_ref = db.collection("exams").document(exam_id).collection("pending_analysis").document(doc_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Analysis not found")
    doc_ref.update({
        "overridden": True,
        "manual_note": note or "",
        "effective_status": "ok",
        "overridden_at": int(time.time() * 1000),
    })
    return {"status": "overridden"}
