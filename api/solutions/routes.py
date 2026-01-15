import os
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
from models.solutions import Solution, SolutionSubmission, UserSolutionResponse, ExamInfo, Task, Group
from typing import List

router = APIRouter(prefix="/api/solutions", tags=["solutions"])


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


@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_solution(submission: SolutionSubmission):
    """Upload user's exam solutions"""
    try:
        db = get_db()
        solution_data = {
            "solutions": [s.dict() for s in submission.solutions]
        }
        
        solution_ref = db.collection("solutions").document(submission.exam_id)\
                        .collection(submission.password).document(submission.email)
        solution_ref.set(solution_data)
        
        return {"message": "Solution uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{exam_id}/{password}/users", response_model=List[UserSolutionResponse])
async def get_users_with_scores(exam_id: str, password: str, exam_info: ExamInfo):
    """Get all users who solved this exam with their scores"""
    try:
        db = get_db()
        
        # Get all solutions for this exam
        solutions_ref = db.collection("solutions").document(exam_id)\
                         .collection(password).stream()
        
        solutions_list = list(solutions_ref)
        print(f"DEBUG: Found {len(solutions_list)} solutions for exam {exam_id}/{password}")
        
        # Calculate total tasks from exam_info
        total_tasks = sum(len(group.tasks) for group in exam_info.groups)
        flat_tasks = [task for group in exam_info.groups for task in group.tasks]
        
        print(f"DEBUG: Total tasks: {total_tasks}, flat_tasks: {len(flat_tasks)}")
        print(f"DEBUG: Exam info groups: {len(exam_info.groups)}")
        
        users_with_scores = []
        
        for solution_doc in solutions_list:
            email = solution_doc.id
            solution_data = solution_doc.to_dict()
            solutions = solution_data.get("solutions", [])
            
            print(f"DEBUG: Processing email {email} with {len(solutions)} solutions")
            
            # Calculate correct answers
            correct_count = 0
            for i, sol in enumerate(solutions):
                if i < len(flat_tasks) and sol.get("state") == flat_tasks[i].state:
                    correct_count += 1
            
            percent = round((correct_count / total_tasks) * 100) if total_tasks > 0 else 0
            
            # Get user data
            user_doc = db.collection("users").document(email).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                users_with_scores.append({
                    "email": email,
                    "username": user_data.get("username", ""),
                    "firstName": user_data.get("firstName", ""),
                    "lastName": user_data.get("lastName", ""),
                    "percent": percent,
                    "creationYear": user_data.get("creationYear", 0),
                    "admin": user_data.get("admin", False)
                })
            else:
                print(f"DEBUG: User {email} not found in users collection")
        
        print(f"DEBUG: Returning {len(users_with_scores)} users with scores")
        return users_with_scores
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{exam_id}/{password}/users/{email}")
async def get_user_answers(exam_id: str, password: str, email: str):
    """Get specific user's answers for an exam"""
    try:
        db = get_db()
        solution_ref = db.collection("solutions").document(exam_id)\
                        .collection(password).document(email)
        solution_doc = solution_ref.get()
        
        if not solution_doc.exists:
            raise HTTPException(status_code=404, detail="Solutions not found")
        
        solution_data = solution_doc.to_dict()
        return {"solutions": solution_data.get("solutions", [])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
