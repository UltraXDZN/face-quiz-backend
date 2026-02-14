import os
import time
import boto3
from fastapi import APIRouter, HTTPException, Query
from models.exams import ExamCaptureMetadata

# Load config from environment
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Create S3 client
s3_client = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

router = APIRouter(prefix="/api/media", tags=["media"])

@router.post("/capture/")
async def upload_capture(metadata: ExamCaptureMetadata):
    """Upload a screenshot and camera capture into a common timestamp folder.
    If timestamp_str is provided, the file is stored in that timestamp folder;
    otherwise a new timestamp (ms) is generated and returned so clients can reuse it."""
    try:
        screenshot = await metadata.screenshot_file.read()
        camera = await metadata.camera_file.read()
        if metadata.timestamp_str:
            try:
                timestamp = int(metadata.timestamp_str)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid timestamp")
        else:
            timestamp = int(time.time() * 1000)

        # Sanitise email for use as a path segment
        safe_email = metadata.email.replace("@", "_at_").replace(".", "_")

        s3_client.put_object(Bucket=BUCKET_NAME, Key=f"{metadata.exam_id}/{safe_email}/{timestamp}/screenshot.png", Body=screenshot, ContentType=(metadata.screenshot_file.content_type or "image/png"))
        s3_client.put_object(Bucket=BUCKET_NAME, Key=f"{metadata.exam_id}/{safe_email}/{timestamp}/camera.png", Body=camera, ContentType=(metadata.camera_file.content_type or "image/png"))
        
        return {"message": "Upload successful", "timestamp": timestamp}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshots/{exam_id}")
async def list_screenshots(exam_id: str, email: str = Query(None)):
    """List screenshots for an exam with timestamps, optionally filtered by student email."""
    try:
        prefix = f"{exam_id}/"
        if email:
            safe_email = email.replace("@", "_at_").replace(".", "_")
            prefix = f"{exam_id}/{safe_email}/"

        paginator = s3_client.get_paginator("list_objects_v2")
        screenshots = []
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Extract timestamp from key path: exam_id/safe_email/timestamp/screenshot.png
                parts = key.split("/")
                if len(parts) >= 3:
                    timestamp = parts[-2]  # timestamp is second-to-last part
                    screenshots.append({
                        "timestamp": timestamp,
                        "url": f"/api/media/screenshot/{key}"
                    })
        return {"screenshots": screenshots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))