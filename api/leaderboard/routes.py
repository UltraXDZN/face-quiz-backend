import os
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
from models.leaderboard import LeaderboardData, LeaderboardUpdateRequest, LeaderboardEntry
from typing import Dict

router = APIRouter(prefix="/api/leaderboard", tags=["leaderboard"])


def get_db():
    """Get Firestore client"""
    if os.getenv("FIRESTORE_EMULATOR_HOST"):
        return firestore.Client(
            project=os.getenv("FIREBASE_TESTING_PROJECT_ID", "demo-test"),
            credentials=AnonymousCredentials()
        )
    else:
        from firebase_admin import firestore as admin_firestore
        return admin_firestore.client()


@router.get("/", response_model=LeaderboardData)
async def get_leaderboard():
    """Get the current leaderboard"""
    try:
        db = get_db()
        leaderboard_doc = db.collection("data").document("leaderboard").get()
        
        if not leaderboard_doc.exists:
            return LeaderboardData(leaderboard={})
        
        data = leaderboard_doc.to_dict()
        return LeaderboardData(leaderboard=data.get("leaderboard", {}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=LeaderboardData)
async def generate_leaderboard():
    """Generate leaderboard from all users' exam data"""
    try:
        db = get_db()
        users_collection = db.collection("users")
        
        # Get all users
        users = users_collection.stream()
        leaderboard_map: Dict[str, LeaderboardEntry] = {}
        
        for user_doc in users:
            user_data = user_doc.to_dict()
            accessed_exams = user_data.get("acessedExams", [])
            
            # Calculate total points
            total_points = sum(exam.get("pointsEarned", 0) for exam in accessed_exams)
            
            jmbag = user_data.get("jmbag")
            if jmbag:
                leaderboard_map[jmbag] = LeaderboardEntry(
                    totalPoints=total_points,
                    img=user_data.get("photoURL")
                )
        
        # Save to Firestore
        leaderboard_doc = db.collection("data").document("leaderboard")
        leaderboard_doc.set({"leaderboard": {
            k: v.dict() for k, v in leaderboard_map.items()
        }})
        
        return LeaderboardData(leaderboard=leaderboard_map)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/", response_model=LeaderboardData)
async def update_leaderboard(request: LeaderboardUpdateRequest):
    """Update a specific user's leaderboard entry"""
    try:
        db = get_db()
        leaderboard_doc = db.collection("data").document("leaderboard")
        leaderboard_snapshot = leaderboard_doc.get()
        
        # Get existing data or create new
        if leaderboard_snapshot.exists:
            data = leaderboard_snapshot.to_dict()
            leaderboard_map = data.get("leaderboard", {})
        else:
            leaderboard_map = {}
        
        # Update specific user
        leaderboard_map[request.userId] = {
            "totalPoints": request.newExamPoints,
            "img": request.img
        }
        
        # Save back to Firestore
        leaderboard_doc.set({"leaderboard": leaderboard_map})
        
        # Convert to proper format for response
        leaderboard_entries = {
            k: LeaderboardEntry(**v) for k, v in leaderboard_map.items()
        }
        
        return LeaderboardData(leaderboard=leaderboard_entries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def update_user_leaderboard_entry(user_email: str):
    """
    Helper function to recalculate and update a user's leaderboard entry.
    Called automatically after exam completion.
    """
    try:
        db = get_db()
        
        # Get user data
        user_doc = db.collection("users").document(user_email).get()
        if not user_doc.exists:
            return
        
        user_data = user_doc.to_dict()
        accessed_exams = user_data.get("acessedExams", [])
        
        # Calculate total points from all completed exams
        total_points = sum(
            exam.get("pointsEarned", 0) 
            for exam in accessed_exams 
            if exam.get("status") == "ZAVRŠEN"
        )
        
        jmbag = user_data.get("jmbag")
        photo_url = user_data.get("photoURL")
        
        if not jmbag:
            return
        
        # Update leaderboard
        leaderboard_doc = db.collection("data").document("leaderboard")
        leaderboard_snapshot = leaderboard_doc.get()
        
        if leaderboard_snapshot.exists:
            data = leaderboard_snapshot.to_dict()
            leaderboard_map = data.get("leaderboard", {})
        else:
            leaderboard_map = {}
        
        leaderboard_map[jmbag] = {
            "totalPoints": total_points,
            "img": photo_url
        }
        
        leaderboard_doc.set({"leaderboard": leaderboard_map})
        
    except Exception as e:
        print(f"Error updating leaderboard for user {user_email}: {str(e)}")
