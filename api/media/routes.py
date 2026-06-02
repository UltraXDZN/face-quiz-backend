import asyncio
import os
import time
import boto3
from botocore.config import Config
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from io import BytesIO
from starlette.concurrency import run_in_threadpool
import cv2
import numpy as np

from api.common.retry import with_retry

try:
    from api.media.face_detection import detect_faces
except ModuleNotFoundError:
    detect_faces = None  # dev stub, face detection unavailable

try:
    from workers.queue_io import enqueue as _enqueue_capture
except Exception:
    _enqueue_capture = None  # worker package unavailable in this environment

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "face-quiz-media")

# Timeouts bound S3 calls so threads don't hang indefinitely.
# botocore retries disabled here; application-level retries via with_retry().
_BOTO_CONFIG = Config(
    connect_timeout=3,
    read_timeout=10,
    retries={"max_attempts": 1},
)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
    config=_BOTO_CONFIG,
)

router = APIRouter(prefix="/media", tags=["media"])


@router.post("/capture")
async def upload_capture(
    screenshot_file: UploadFile = File(...),
    camera_file: UploadFile = File(...),
    exam_id: str = Form(...),
    email: str = Form(...),
    timestamp_str: str = Form(None),
):
    """Upload a screenshot and camera capture into a common timestamp folder."""
    try:
        screenshot = await screenshot_file.read()
        camera = await camera_file.read()
        if timestamp_str:
            try:
                timestamp = int(timestamp_str)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid timestamp")
        else:
            timestamp = int(time.time() * 1000)

        safe_email = email.replace("@", "_at_").replace(".", "_")
        screenshot_key = f"{exam_id}/{safe_email}/{timestamp}/screenshot.png"
        camera_key = f"{exam_id}/{safe_email}/{timestamp}/camera.png"

        await asyncio.gather(
            with_retry(lambda: run_in_threadpool(
                s3_client.put_object,
                Bucket=BUCKET_NAME, Key=screenshot_key, Body=screenshot,
                ContentType=(screenshot_file.content_type or "image/png"),
            )),
            with_retry(lambda: run_in_threadpool(
                s3_client.put_object,
                Bucket=BUCKET_NAME, Key=camera_key, Body=camera,
                ContentType=(camera_file.content_type or "image/png"),
            )),
        )

        if _enqueue_capture is not None:
            await run_in_threadpool(_enqueue_capture, exam_id, email, timestamp, camera_key)

        return {"message": "Upload successful", "timestamp": timestamp}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshots")
async def list_screenshots(request: Request, exam_id: str, email: str = Query(None)):
    """List screenshots for an exam with timestamps."""
    try:
        base_url = str(request.base_url).rstrip("/")
        prefix = f"{exam_id}/"
        if email:
            safe_email = email.replace("@", "_at_").replace(".", "_")
            prefix = f"{exam_id}/{safe_email}/"

        def _list():
            paginator = s3_client.get_paginator("list_objects_v2")
            items = {}
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    parts = key.split("/")
                    if len(parts) >= 3:
                        ts = parts[-2]
                        base_path = "/".join(parts[:-1])
                        entry = items.setdefault(ts, {
                            "timestamp": ts,
                            "screenshot_url": None,
                            "camera_url": None,
                        })
                        filename = parts[-1].lower()
                        if filename.endswith("screenshot.png"):
                            entry["screenshot_url"] = f"{base_url}/media/screenshot/{base_path}/screenshot.png"
                        elif filename.endswith("camera.png"):
                            entry["camera_url"] = f"{base_url}/media/camera/{base_path}/camera.png"
            return list(items.values())

        screenshots = await with_retry(lambda: run_in_threadpool(_list))
        return {"screenshots": screenshots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshot/{exam_id}/{safe_email}/{timestamp}/screenshot.png")
async def get_screenshot(exam_id: str, safe_email: str, timestamp: str):
    """Retrieve a screenshot image from S3."""
    try:
        key = f"{exam_id}/{safe_email}/{timestamp}/screenshot.png"

        def _get():
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            return response["Body"].read()

        image_data = await with_retry(lambda: run_in_threadpool(_get))
        return StreamingResponse(BytesIO(image_data), media_type="image/png")
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Screenshot not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/camera/{exam_id}/{safe_email}/{timestamp}/camera.png")
async def get_camera(exam_id: str, safe_email: str, timestamp: str):
    """Retrieve a camera image from S3."""
    try:
        key = f"{exam_id}/{safe_email}/{timestamp}/camera.png"

        def _get():
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            return response["Body"].read()

        image_data = await with_retry(lambda: run_in_threadpool(_get))
        return StreamingResponse(BytesIO(image_data), media_type="image/png")
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Camera image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/face-check")
async def face_check(image: UploadFile = File(...), confidence: float = Form(0.5)):
    """Detect face(s) from a single camera frame."""
    try:
        if detect_faces is None:
            raise HTTPException(status_code=500, detail="Face detection not configured")
        image_bytes = await image.read()

        def _detect():
            np_buffer = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid image data")
            return detect_faces(frame, float(confidence))

        face_count, _ = await run_in_threadpool(_detect)

        if face_count == 1:
            status = "single"
        elif face_count > 1:
            status = "multiple"
        else:
            status = "none"

        return {
            "faceDetected": face_count == 1,
            "faceCount": face_count,
            "status": status,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
