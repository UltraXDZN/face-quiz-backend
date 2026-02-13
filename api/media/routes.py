import os
import io
import time
import boto3
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import StreamingResponse
from typing import List

# Load config from environment
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Create S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

router = APIRouter(prefix="/api/media", tags=["media"])


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read file content
        contents = await file.read()
        
        # Upload to S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=file.filename,  # Use original filename; in prod, generate unique names
            Body=contents
        )
        return {"message": f"File '{file.filename}' uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screenshot/")
async def upload_screenshot(
    file: UploadFile = File(...),
    exam_id: str = Form(...),
    email: str = Form(...),
):
    """Upload a fullscreen screenshot captured during an exam."""
    try:
        contents = await file.read()
        timestamp = int(time.time() * 1000)
        # Sanitise email for use as a path segment
        safe_email = email.replace("@", "_at_").replace(".", "_")
        key = f"screenshots/{exam_id}/{safe_email}/{timestamp}.png"

        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=contents,
            ContentType="image/png",
        )
        return {"message": "Screenshot uploaded", "key": key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshots/{exam_id}")
async def list_screenshots(exam_id: str, email: str = Query(None)):
    """List screenshot keys for an exam, optionally filtered by student email."""
    try:
        prefix = f"screenshots/{exam_id}/"
        if email:
            safe_email = email.replace("@", "_at_").replace(".", "_")
            prefix = f"screenshots/{exam_id}/{safe_email}/"

        paginator = s3_client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return {"screenshots": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshot/{file_key:path}")
async def get_screenshot(file_key: str):
    """Download a single screenshot by its full S3 key."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        file_content = response["Body"].read()
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={file_key.split('/')[-1]}"},
        )
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Screenshot not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{file_name}")
async def download_file(file_name: str):
    try:
        # Get object from S3
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_name)
        file_content = response["Body"].read()
        
        # Stream back to client
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={file_name}"}
        )
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))