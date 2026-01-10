from fastapi import APIRouter, HTTPException, status
from firebase_admin import firestore, auth
from models.users import User, UserCreate, UserUpdate
from typing import List

router = APIRouter(prefix="/api/users", tags=["users"])


def get_db():
    """Get Firestore client"""
    return firestore.client()


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
