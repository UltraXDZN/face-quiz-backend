import os
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
from models.solutions import (
    Solution, SolutionSubmission, UserSolutionResponse, 
    ExamInfo, Task, Group, ExamResultResponse, TaskResult, GroupResult
)
from typing import List, Dict
from collections import defaultdict

router = APIRouter(prefix="/solutions", tags=["solutions"])

# In-memory cache for exam results: {exam_id: {password: [ExamResultResponse]}}
results_cache: Dict[str, Dict[str, List[ExamResultResponse]]] = defaultdict(dict)


def get_db():
    """Get Firestore client"""
    # For emulator, use AnonymousCredentials
    if os.getenv("ENVIRONMENT") != "production":
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
        
        # Update in-memory cache after successful DB write
        try:
            print(f"[CACHE] Updating cache for {submission.exam_id}/{submission.password}/{submission.email}")
            
            # Get exam data
            exam_doc = db.collection("exams").document(submission.exam_id).get()
            if exam_doc.exists:
                exam_data = exam_doc.to_dict()
                user_solutions = solution_data.get("solutions", [])
                
                # Calculate results for this user
                result = _calculate_user_results(
                    submission.exam_id, 
                    submission.password, 
                    submission.email, 
                    exam_data, 
                    user_solutions, 
                    db
                )
                
                # Update cache
                if submission.exam_id not in results_cache:
                    results_cache[submission.exam_id] = {}
                
                if submission.password not in results_cache[submission.exam_id]:
                    results_cache[submission.exam_id][submission.password] = []
                
                # Remove old result for this user if exists, then add new one
                results_cache[submission.exam_id][submission.password] = [
                    r for r in results_cache[submission.exam_id][submission.password] 
                    if r.user_email != submission.email
                ]
                results_cache[submission.exam_id][submission.password].append(result)
                
                print(f"[CACHE] Cache updated. Total users in cache: {len(results_cache[submission.exam_id][submission.password])}")
        except Exception as cache_error:
            print(f"[CACHE] Error updating cache: {str(cache_error)}")
            # Don't fail the request if cache update fails
        
        return {"message": "Solution uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{exam_id}/{password}/users", response_model=List[UserSolutionResponse])
async def get_users_with_scores(exam_id: str, password: str):
    """Get all users who solved this exam with their scores"""
    try:
        db = get_db()
        
        # Get the exam data from backend to calculate scores
        exam_doc = db.collection("exams").document(exam_id).get()
        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam_data = exam_doc.to_dict()
        
        # Get all solutions for this exam
        solutions_ref = db.collection("solutions").document(exam_id).collection(password).stream()
        
        solutions_list = list(solutions_ref)
        print(f"DEBUG: Found {len(solutions_list)} solutions for exam {exam_id}/{password}")
        
        # Calculate total tasks from exam data
        groups = exam_data.get("groups", [])
        total_tasks = sum(len(group.get("tasks", [])) for group in groups)
        flat_tasks = [task for group in groups for task in group.get("tasks", [])]
        
        print(f"DEBUG: Total tasks: {total_tasks}, flat_tasks: {len(flat_tasks)}")
        print(f"DEBUG: Exam groups: {len(groups)}")
        
        users_with_scores = []
        
        for solution_doc in solutions_list:
            email = solution_doc.id
            solution_data = solution_doc.to_dict()
            solutions = solution_data.get("solutions", [])
            
            print(f"DEBUG: Processing email {email} with {len(solutions)} solutions")
            
            # Calculate correct answers
            correct_count = 0
            for i, sol in enumerate(solutions):
                if i < len(flat_tasks) and sol.get("state") == flat_tasks[i].get("state"):
                    correct_count += 1
            
            percent = round((correct_count / total_tasks) * 100) if total_tasks > 0 else 0
            
            # Get user data by querying email field (doc ID is username, not email)
            user_query = db.collection("users").where("email", "==", email).limit(1).stream()
            user_docs = list(user_query)
            if user_docs:
                user_data = user_docs[0].to_dict()
            else:
                user_data = {}
                print(f"DEBUG: User {email} not found in users collection")
            users_with_scores.append({
                "email": email,
                "username": user_data.get("username", ""),
                "name": user_data.get("name", ""),
                "surname": user_data.get("surname", ""),
                "percent": percent,
                "creationYear": user_data.get("creationYear", 0),
                "admin": user_data.get("admin", False)
            })
        
        print(f"DEBUG: Returning {len(users_with_scores)} users with scores")
        return users_with_scores
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{exam_id}/{password}/users/{email}/results", response_model=ExamResultResponse)
async def get_user_exam_results(exam_id: str, password: str, email: str):
    """Get user's exam results with correct answers and calculated score"""
    try:
        db = get_db()
        
        # Get the full exam with correct answers
        exam_doc = db.collection("exams").document(exam_id).get()
        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam_data = exam_doc.to_dict()
        
        # Get user's solutions
        solution_ref = db.collection("solutions").document(exam_id)\
                        .collection(password).document(email)
        solution_doc = solution_ref.get()
        
        if not solution_doc.exists:
            raise HTTPException(status_code=404, detail="User solutions not found")
        
        user_solutions = solution_doc.to_dict().get("solutions", [])
        
        return _calculate_user_results(exam_id, password, email, exam_data, user_solutions, db)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{exam_id}/{password}/results/all", response_model=List[ExamResultResponse])
async def get_all_users_exam_results(exam_id: str, password: str):
    """Get exam results for ALL users at once (optimized for admin view with in-memory cache)"""
    try:
        # Check in-memory cache first
        if exam_id in results_cache and password in results_cache[exam_id]:
            cached_results = results_cache[exam_id][password]
            if cached_results:
                print(f"[CACHE] Returning cached results for {exam_id}/{password}: {len(cached_results)} users")
                return cached_results
        
        print(f"[BATCH] Cache miss. Fetching all results from DB for exam {exam_id}/{password}")
        db = get_db()
        
        # Get the full exam with correct answers once
        exam_doc = db.collection("exams").document(exam_id).get()
        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam_data = exam_doc.to_dict()
        print(f"[BATCH] Exam loaded: {exam_data.get('title', 'N/A')}")
        
        # Get all solutions for this exam
        solutions_ref = db.collection("solutions").document(exam_id).collection(password).stream()
        
        all_results = []
        user_count = 0
        for solution_doc in solutions_ref:
            user_count += 1
            email = solution_doc.id
            user_solutions = solution_doc.to_dict().get("solutions", [])
            
            try:
                result = _calculate_user_results(exam_id, password, email, exam_data, user_solutions, db)
                all_results.append(result)
            except Exception as e:
                print(f"[BATCH] Error calculating results for {email}: {str(e)}")
                continue
        
        # Update cache after successful DB fetch
        if exam_id not in results_cache:
            results_cache[exam_id] = {}
        results_cache[exam_id][password] = all_results
        
        print(f"[BATCH] Returning results for {len(all_results)} users (processed {user_count} total)")
        print(f"[CACHE] Cache populated with {len(all_results)} results")
        return all_results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_user_results(exam_id: str, password: str, email: str, exam_data: dict, user_solutions: list, db) -> ExamResultResponse:
    """Helper function to calculate results for a single user"""
    # Create a map of user solutions for quick lookup
    solution_map = {sol["id"]: sol.get("state") for sol in user_solutions}
    
    # Calculate results for each task
    group_results = []
    total_points = 0.0
    achieved_points = 0.0
    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0
    total_tasks = 0
    
    for group in exam_data.get("groups", []):
        task_results = []
        
        for task in group.get("tasks", []):
            task_id = task.get("id")
            correct_answer = task.get("state")
            user_answer = solution_map.get(task_id)
            positive_points = task.get("positive_points", 1.0)
            negative_points = task.get("negative_points", 0.3)
            
            total_points += positive_points
            total_tasks += 1
            
            # Calculate if answer is correct and points earned
            is_correct = user_answer == correct_answer
            points_earned = 0.0
            
            if user_answer is None:
                unanswered_count += 1
            elif is_correct:
                points_earned = positive_points
                achieved_points += positive_points
                correct_count += 1
            else:
                points_earned = -negative_points
                achieved_points -= negative_points
                incorrect_count += 1
            
            task_result = TaskResult(
                id=task_id,
                text=task.get("text", ""),
                user_answer=user_answer,
                correct_answer=correct_answer,
                is_correct=is_correct,
                points_earned=points_earned,
                image=task.get("image")
            )
            task_results.append(task_result)
        
        group_result = GroupResult(
            id=group.get("id"),
            text=group.get("text", ""),
            tasks=task_results
        )
        group_results.append(group_result)
    
    # Ensure achieved_points is not negative
    achieved_points = max(0.0, achieved_points)
    
    # Update leaderboard with new exam result
    leaderboard_message = "Leaderboard updated"
    try:
        # Get user data to find jmbag (doc ID is username, not email)
        user_query = db.collection("users").where("email", "==", email).limit(1).stream()
        user_docs = list(user_query)
        if user_docs:
            user_data = user_docs[0].to_dict()
            jmbag = user_data.get("jmbag")
            photo_url = user_data.get("photoURL")
            
            if jmbag:
                # Calculate total points from all completed exams
                accessed_exams = user_data.get("acessedExams", [])
                total_user_points = sum(
                    exam.get("pointsEarned", 0) 
                    for exam in accessed_exams 
                    if exam.get("status") == "ZAVRŠEN"
                )
                
                # Update leaderboard
                leaderboard_doc = db.collection("data").document("leaderboard")
                leaderboard_snapshot = leaderboard_doc.get()
                
                if leaderboard_snapshot.exists:
                    data = leaderboard_snapshot.to_dict()
                    leaderboard_map = data.get("leaderboard", {})
                else:
                    leaderboard_map = {}
                
                leaderboard_map[jmbag] = {
                    "totalPoints": float(total_user_points),
                    "img": photo_url
                }
                
                leaderboard_doc.set({"leaderboard": leaderboard_map})
                print(f"Successfully updated leaderboard for {jmbag} with {total_user_points} points")
            else:
                leaderboard_message = "Leaderboard update skipped: No JMBAG found"
                print(f"User {email} has no jmbag")
        else:
            leaderboard_message = "Leaderboard update skipped: User not found"
            print(f"User {email} not found in users collection")
    except Exception as leaderboard_error:
        leaderboard_message = f"Leaderboard failed to update: {str(leaderboard_error)}"
        print(f"Error updating leaderboard: {str(leaderboard_error)}")
    
    return ExamResultResponse(
        exam_id=exam_id,
        exam_title=exam_data.get("title", ""),
        exam_password=password,
        user_email=email,
        groups=group_results,
        total_points=total_points,
        achieved_points=round(achieved_points, 1),
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        unanswered_count=unanswered_count,
        total_tasks=total_tasks,
        leaderboard_message=leaderboard_message
    )


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


@router.delete("/{exam_id}/{password}/users/{email}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_solution(exam_id: str, password: str, email: str):
    """Delete a specific user's solution for an exam"""
    try:
        db = get_db()
        solution_ref = db.collection("solutions").document(exam_id)\
                        .collection(password).document(email)
        solution_doc = solution_ref.get()

        if not solution_doc.exists:
            raise HTTPException(status_code=404, detail="Solutions not found")

        solution_ref.delete()

        # Invalidate in-memory cache for this exam/password
        if exam_id in results_cache and password in results_cache[exam_id]:
            results_cache[exam_id][password] = [
                r for r in results_cache[exam_id][password]
                if r.user_email != email
            ]

        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
