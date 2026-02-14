import os
import time
import boto3
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse
from io import BytesIO

# Load config from environment
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Create S3 client
s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

router = APIRouter(prefix="/api/media", tags=["media"])

@router.post("/capture/")
async def upload_capture(screenshot_file: UploadFile = File(...),
    camera_file: UploadFile = File(...),
    exam_id: str = Form(...),
    email: str = Form(...),
    timestamp_str: str = Form(None),
):
    """Upload a screenshot and camera capture into a common timestamp folder.
    If timestamp_str is provided, the file is stored in that timestamp folder;
    otherwise a new timestamp (ms) is generated and returned so clients can reuse it."""
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

        # Sanitise email for use as a path segment
        safe_email = email.replace("@", "_at_").replace(".", "_")

        s3_client.put_object(Bucket=BUCKET_NAME, Key=f"{exam_id}/{safe_email}/{timestamp}/screenshot.png", Body=screenshot, ContentType=(screenshot_file.content_type or "image/png"))
        s3_client.put_object(Bucket=BUCKET_NAME, Key=f"{exam_id}/{safe_email}/{timestamp}/camera.png", Body=camera, ContentType=(camera_file.content_type or "image/png"))
        
        return {"message": "Upload successful", "timestamp": timestamp}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshots/")
async def list_screenshots(request: Request, exam_id: str, email: str = Query(None)):
    """List screenshots for an exam with timestamps, optionally filtered by student email.
    Each entry contains timestamp, screenshot_url and camera_url (when available)."""
    try:
        # Get base URL from request
        base_url = str(request.base_url).rstrip("/")
        
        prefix = f"{exam_id}/"
        if email:
            safe_email = email.replace("@", "_at_").replace(".", "_")
            prefix = f"{exam_id}/{safe_email}/"

        paginator = s3_client.get_paginator("list_objects_v2")
        items = {}
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                parts = key.split("/")
                if len(parts) >= 3:
                    timestamp = parts[-2]
                    base_path = "/".join(parts[:-1])  # exam_id/safe_email/timestamp
                    entry = items.setdefault(timestamp, {
                        "timestamp": timestamp,
                        "screenshot_url": None,
                        "camera_url": None,
                    })
                    filename = parts[-1].lower()
                    if filename.endswith("screenshot.png"):
                        entry["screenshot_url"] = f"{base_url}/media/screenshot/{base_path}/screenshot.png"
                    elif filename.endswith("camera.png"):
                        entry["camera_url"] = f"{base_url}/media/camera/{base_path}/camera.png"

        screenshots = list(items.values())
        return {"screenshots": screenshots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshot/{exam_id}/{safe_email}/{timestamp}/screenshot.png")
async def get_screenshot(exam_id: str, safe_email: str, timestamp: str):
    """Retrieve a screenshot image from S3."""
    try:
        key = f"{exam_id}/{safe_email}/{timestamp}/screenshot.png"
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        image_data = response["Body"].read()
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
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        image_data = response["Body"].read()
        return StreamingResponse(BytesIO(image_data), media_type="image/png")
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Camera image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))