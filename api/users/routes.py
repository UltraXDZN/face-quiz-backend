import os
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
from models.users import User, UserCreate, UserUpdate, UserUISettings
from typing import List
# Import leaderboard update helper
from api.leaderboard.routes import update_user_leaderboard_entry
router = APIRouter(prefix="/users", tags=["users"])


def get_db():
    """Get Firestore client"""
    # For emulator, use AnonymousCredentials
    if os.getenv("ENVIRONMENT") != "production":
        print("🔥 Using Firestore Emulator Client for users")
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials()
        )
    else:
        # For production, use firebase_admin
        from firebase_admin import firestore as admin_firestore
        print("🔥 Using Firestore Admin Client for users")
        return admin_firestore.client()


@router.get("/", response_model=List[User])
async def get_all_users():
    """Get all users"""
    try:
        db = get_db()
        users_ref = db.collection("users").stream()
        users = [User.model_validate(user.to_dict()) for user in users_ref]
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{email}", response_model=User)
async def get_user(email: str):
    """Get user by email"""
    try:
        db = get_db()
        print(f"🔍 Looking for user: {email}")
        
        # Try to get document directly by email first (if email is used as doc ID)
        try:
            user_doc = db.collection("users").document(email).get()
            if user_doc.exists:
                print(f"✅ Found user by document ID: {email}")
                return User.model_validate(user_doc.to_dict())
        except Exception as doc_error:
            print(f"⚠️ Could not get by document ID: {doc_error}")
        
        # Fall back to query by email field
        user_query = db.collection("users").where("email", "==", email).stream()
        users = list(user_query)
        
        if not users:
            print(f"❌ User not found: {email}")
            raise HTTPException(status_code=404, detail="User not found")
        
        print(f"✅ Found user by query: {email}")
        return User.model_validate(users[0].to_dict())
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting user '{email}': {e}")
        import traceback
        traceback.print_exc()
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
        return User.model_validate(user_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{email}", response_model=User)
async def update_user(email: str, user_update: UserUpdate):
    """Update user information"""
    try:
        db = get_db()
        # Check if user exists - try document ID first, then query
        user_doc = db.collection("users").document(email).get()
        
        if not user_doc.exists:
            # Try querying by email field
            user_query = db.collection("users").where("email", "==", email).stream()
            users = list(user_query)
            if not users:
                raise HTTPException(status_code=404, detail="User not found")
            user_doc = users[0]
        
        # Update only provided fields
        update_data = {k: v for k, v in user_update.dict().items() if v is not None}
        db.collection("users").document(user_doc.id).update(update_data)
        
        # Return updated user
        updated_user = db.collection("users").document(user_doc.id).get()
        user_data = updated_user.to_dict()
        
        # If photoURL was updated, update leaderboard entry
        if user_update.photoURL is not None:
            try:
                await update_user_leaderboard_entry(email)
            except Exception as e:
                print(f"Failed to update leaderboard for user {email}: {e}")
        
        return User.model_validate(user_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{email}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(email: str):
    """Delete a user and remove them from created-users tracking"""
    try:
        db = get_db()
        # Try document by email (used as doc ID) first
        user_doc_ref = db.collection("users").document(email)
        user_doc = user_doc_ref.get()

        if not user_doc.exists:
            # Fall back to querying by email field
            user_query = db.collection("users").where("email", "==", email).stream()
            users = list(user_query)
            if not users:
                raise HTTPException(status_code=404, detail="User not found")
            user_doc_ref = db.collection("users").document(users[0].id)

        # Delete the user document
        user_doc_ref.delete()

        # Remove the user from the created-users tracking map
        data_ref = db.collection("data").document("users")
        data_doc = data_ref.get()
        if data_doc.exists:
            data = data_doc.to_dict() or {}
            if email in data.get("createdUsers", {}):
                data_ref.update({f"createdUsers.{email}": firestore.DELETE_FIELD})

        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{email}", response_model=User)
async def full_update_user(email: str, user_data: User):
    """Full replacement of a user document (equivalent to Firestore setDoc)"""
    try:
        db = get_db()
        # Try document by email (used as doc ID) first
        user_doc_ref = db.collection("users").document(email)
        user_doc = user_doc_ref.get()

        if not user_doc.exists:
            # Fall back to querying by email field
            user_query = db.collection("users").where("email", "==", email).stream()
            users = list(user_query)
            if not users:
                raise HTTPException(status_code=404, detail="User not found")
            user_doc_ref = db.collection("users").document(users[0].id)

        # Full replacement of the document
        user_doc_ref.set(user_data.dict())

        return user_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{email}/ui-settings")
async def update_user_ui_settings(email: str, ui_settings: UserUISettings):
    """Update user UI settings"""
    try:
        db = get_db()
        # Try document by email first
        user_doc_ref = db.collection("users").document(email)
        user_doc = user_doc_ref.get()

        if not user_doc.exists:
            # Fall back to query
            user_query = db.collection("users").where("email", "==", email).stream()
            users = list(user_query)
            if not users:
                raise HTTPException(status_code=404, detail="User not found")
            user_doc = users[0]
            user_doc_ref = db.collection("users").document(user_doc.id)

        # Update UISettings field
        user_doc_ref.update({"UISettings": ui_settings.dict()})

        return {"message": "UI settings updated successfully", "UISettings": ui_settings.dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{email}/exams", response_model=List[dict])
async def get_user_exams(email: str):
    """Get user's accessed exams"""
    try:
        db = get_db()
        # Try document by email first
        user_doc = db.collection("users").document(email).get()
        
        if not user_doc.exists:
            # Fall back to query
            user_query = db.collection("users").where("email", "==", email).stream()
            users = list(user_query)
            if not users:
                raise HTTPException(status_code=404, detail="User not found")
            user_data = users[0].to_dict()
        else:
            user_data = user_doc.to_dict()
        
        return user_data.get("acessedExams", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{email}/exams")
async def update_user_accessed_exam(email: str, accessed_exam: dict, old_exam_id: str = None):
    """Add or update an accessed exam for a user"""
    try:
        db = get_db()
        # Try document by email first
        user_doc_ref = db.collection("users").document(email)
        user_doc = user_doc_ref.get()
        
        if not user_doc.exists:
            # Fall back to query
            user_query = db.collection("users").where("email", "==", email).stream()
            users = list(user_query)
            if not users:
                raise HTTPException(status_code=404, detail="User not found")
            user_doc = users[0]
            user_doc_ref = db.collection("users").document(user_doc.id)
        
        user_data = user_doc.to_dict()
        accessed_exams = user_data.get("acessedExams", [])
        
        # Remove old exam if provided
        if old_exam_id:
            accessed_exams = [exam for exam in accessed_exams if exam.get("id") != old_exam_id]
        
        # Add new exam
        accessed_exams.append(accessed_exam)
        
        # Update document
        user_doc_ref.update({"acessedExams": accessed_exams})
        
        # Automatically update leaderboard if exam is finished
        if accessed_exam.get("status") == "ZAVRŠEN":
            await update_user_leaderboard_entry(user_doc_ref.id)
        
        return {"message": "Accessed exam updated successfully", "acessedExams": accessed_exams}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{email}/exams/{exam_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_accessed_exam(email: str, exam_id: str):
    """Remove an accessed exam from a user"""
    try:
        db = get_db()
        # Try document by email first
        user_doc_ref = db.collection("users").document(email)
        user_doc = user_doc_ref.get()
        
        if not user_doc.exists:
            # Fall back to query
            user_query = db.collection("users").where("email", "==", email).stream()
            users = list(user_query)
            if not users:
                raise HTTPException(status_code=404, detail="User not found")
            user_doc = users[0]
            user_doc_ref = db.collection("users").document(user_doc.id)
        
        user_data = user_doc.to_dict()
        accessed_exams = user_data.get("acessedExams", [])
        
        # Remove exam
        accessed_exams = [exam for exam in accessed_exams if exam.get("id") != exam_id]
        
        # Update document
        user_doc_ref.update({"acessedExams": accessed_exams})
        
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
