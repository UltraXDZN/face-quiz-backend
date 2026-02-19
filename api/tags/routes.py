import os
import uuid
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
from models.tags import Tag, TagCreate
from typing import List

router = APIRouter(prefix="/api/tags", tags=["tags"])


def get_db():
    """Get Firestore client"""
    if os.getenv("ENVIRONMENT") != "production":
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials()
        )
    else:
        from firebase_admin import firestore as admin_firestore
        return admin_firestore.client()


@router.get("/", response_model=List[Tag])
async def get_tags():
    """Get all tags"""
    try:
        db = get_db()
        tags_ref = db.collection("tags").stream()
        return [Tag(**tag.to_dict()) for tag in tags_ref]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=Tag)
async def create_tag(tag_data: TagCreate):
    """Create a new tag"""
    try:
        db = get_db()

        existing = list(db.collection("tags").where("name", "==", tag_data.name).limit(1).stream())
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Tag with name '{tag_data.name}' already exists"
            )

        tag_id = uuid.uuid4().hex[:20]
        tag = Tag(id=tag_id, **tag_data.dict())
        db.collection("tags").document(tag_id).set(tag.dict())
        return tag
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{tag_id}", response_model=Tag)
async def update_tag(tag_id: str, tag: Tag):
    """Update a tag"""
    try:
        db = get_db()
        tag_ref = db.collection("tags").document(tag_id)

        if not tag_ref.get().exists:
            raise HTTPException(status_code=404, detail="Tag not found")

        tag_ref.set(tag.dict())
        return tag
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tag(tag_id: str):
    """Delete a tag"""
    try:
        db = get_db()
        tag_ref = db.collection("tags").document(tag_id)

        if not tag_ref.get().exists:
            raise HTTPException(status_code=404, detail="Tag not found")

        tag_ref.delete()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
