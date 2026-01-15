import os
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
from models.users import User, UserCreate, UserUpdate
from typing import List
# Import leaderboard update helper
from api.leaderboard.routes import update_user_leaderboard_entry
router = APIRouter(prefix="/api/users", tags=["users"])


def get_db():
    """Get Firestore client"""
    # For emulator, use AnonymousCredentials
    if os.getenv("FIRESTORE_EMULATOR_HOST"):
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials()
        )
    else:
        # For production, use firebase_admin
        from firebase_admin import firestore as admin_firestore
        return admin_firestore.client()


@router.get("/", response_model=List[User])
async def get_all_users():
    """Get all users"""
    try:
        db = get_db()
        users_ref = db.collection("users").stream()
        users = [user.to_dict() for user in users_ref]
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{username}", response_model=User)
async def get_user(username: str):
    """Get user by username"""
    try:
        db = get_db()
        user_query = db.collection("users").where("username", "==", username).stream()
        users = list(user_query)
        
        if not users:
            raise HTTPException(status_code=404, detail="User not found")
        
        return users[0].to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    """Create a new user"""
    try:
        db = get_db()
        # Check if user already exists
        existing = db.collection("users").where("username", "==", user.username).stream()
        if any(existing):
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Create new user document
        user_data = {
            **user.dict(),
            "creationYear": 2025,
            "admin": False,
            "tags": [],
            "acessedExams": []
        }
        
        db.collection("users").document(user.username).set(user_data)
        return user_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{username}", response_model=User)
async def update_user(username: str, user_update: UserUpdate):
    """Update user information"""
    try:
        db = get_db()
        # Check if user exists
        user_query = db.collection("users").where("username", "==", username).stream()
        users = list(user_query)
        
        if not users:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_doc = users[0]
        # Update only provided fields
        update_data = {k: v for k, v in user_update.dict().items() if v is not None}
        db.collection("users").document(user_doc.id).update(update_data)
        
        # Return updated user
        updated_user = db.collection("users").document(user_doc.id).get()
        return updated_user.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(username: str):
    """Delete a user"""
    try:
        db = get_db()
        # Check if user exists
        user_doc = db.collection("users").document(username).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        db.collection("users").document(username).delete()
        return "User deleted successfully"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{username}/exams", response_model=List[dict])
async def get_user_exams(username: str):
    """Get user's accessed exams"""
    try:
        db = get_db()
        user_query = db.collection("users").where("username", "==", username).stream()
        users = list(user_query)
        
        if not users:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = users[0].to_dict()
        return user_data.get("acessedExams", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{username}/exams")
async def update_user_accessed_exam(username: str, accessed_exam: dict, old_exam_id: str = None):
    """Add or update an accessed exam for a user"""
    try:
        db = get_db()
        user_query = db.collection("users").where("username", "==", username).stream()
        users = list(user_query)
        
        if not users:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_doc = users[0]
        user_data = user_doc.to_dict()
        accessed_exams = user_data.get("acessedExams", [])
        
        # Remove old exam if provided
        if old_exam_id:
            accessed_exams = [exam for exam in accessed_exams if exam.get("id") != old_exam_id]
        
        # Add new exam
        accessed_exams.append(accessed_exam)
        
        # Update document
        db.collection("users").document(user_doc.id).update({"acessedExams": accessed_exams})
        
        # Automatically update leaderboard if exam is finished
        if accessed_exam.get("status") == "ZAVRŠEN":
            await update_user_leaderboard_entry(user_doc.id)
        
        return {"message": "Accessed exam updated successfully", "acessedExams": accessed_exams}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{username}/exams/{exam_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_accessed_exam(username: str, exam_id: str):
    """Remove an accessed exam from a user"""
    try:
        db = get_db()
        user_query = db.collection("users").where("username", "==", username).stream()
        users = list(user_query)
        
        if not users:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_doc = users[0]
        user_data = user_doc.to_dict()
        accessed_exams = user_data.get("acessedExams", [])
        
        # Remove exam
        accessed_exams = [exam for exam in accessed_exams if exam.get("id") != exam_id]
        
        # Update document
        db.collection("users").document(user_doc.id).update({"acessedExams": accessed_exams})
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tracking/created-users")
async def get_created_users_tracking():
    """Get created users tracking data"""
    try:
        db = get_db()
        data_doc = db.collection("data").document("users").get()
        
        if not data_doc.exists:
            return {"createdUsers": {}}
        
        return data_doc.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tracking/created-users/{email}")
async def update_created_user_tracking(email: str, version: int = 0):
    """Update created user tracking version"""
    try:
        db = get_db()
        data_ref = db.collection("data").document("users")
        
        # Use set with merge to create if doesn't exist
        data_ref.set({
            "createdUsers": {
                email: version
            }
        }, merge=True)
        
        return {"message": "User tracking updated successfully", "email": email, "version": version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
