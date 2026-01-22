import os
from fastapi import APIRouter, HTTPException, status
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
from models.exams import (
    Exam, ShortExam, ExamPasswordsUpdate, ExamsDataResponse, ExamMetadata,
    ExamForStudent, GroupWithoutAnswers, TaskWithoutAnswer
)
from typing import List

router = APIRouter(prefix="/api/exams", tags=["exams"])


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
async def create_exam(exam: Exam):
    """Create a new exam"""
    try:
        db = get_db()
        
        # Check if password already exists
        data_doc = db.collection("data").document("exams").get()
        if data_doc.exists:
            data = data_doc.to_dict()
            exam_passwords = data.get("examPasswords", {})
            
            # Check if password is already taken
            for exam_id, passwords in exam_passwords.items():
                if exam.password in passwords:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Password '{exam.password}' is already taken"
                    )
            
            # Check if title is already taken
            created_exams = data.get("createdExams", [])
            for created_exam in created_exams:
                if created_exam.get("title") == exam.title:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Title '{exam.title}' is already taken"
                    )
        
        # Create exam document
        exam_ref = db.collection("exams").document(exam.id)
        exam_ref.set(exam.dict())
        
        # Create short exam for the list
        short_exam = ShortExam(
            id=exam.id,
            title=exam.title,
            creator=exam.creator,
            timestamp=exam.timestamp,
            password=exam.password,
            accessLimit=exam.accessLimit,
            timeLimit=exam.timeLimit,
            startLimit=exam.startLimit,
            endLimit=exam.endLimit,
            version=0
        )
        
        # Update data/exams document
        data_ref = db.collection("data").document("exams")
        
        # Add to createdExams array
        if data_doc.exists:
            data_ref.update({
                "createdExams": firestore.ArrayUnion([short_exam.dict()])
            })
        else:
            data_ref.set({
                "createdExams": [short_exam.dict()],
                "examPasswords": {}
            })
        
        # Set exam password
        data_ref.set({
            "examPasswords": {
                exam.id: {
                    exam.password: firestore.SERVER_TIMESTAMP
                }
            }
        }, merge=True)
        
        return {"message": "Exam created successfully", "exam": exam}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data", response_model=ExamsDataResponse)
async def get_exams_data():
    """Get all exams metadata (created exams list and passwords)"""
    try:
        db = get_db()
        data_doc = db.collection("data").document("exams").get()
        
        if not data_doc.exists:
            return ExamsDataResponse(createdExams=[], examPasswords={})
        
        data = data_doc.to_dict()
        
        # Convert DatetimeWithNanoseconds to integer timestamps
        exam_passwords = data.get("examPasswords", {})
        converted_passwords = {}
        for exam_id, passwords in exam_passwords.items():
            converted_passwords[exam_id] = {}
            for password, timestamp in passwords.items():
                # Convert Firestore timestamp to milliseconds
                if hasattr(timestamp, 'timestamp'):
                    converted_passwords[exam_id][password] = int(timestamp.timestamp() * 1000)
                else:
                    converted_passwords[exam_id][password] = timestamp
        
        return ExamsDataResponse(
            createdExams=data.get("createdExams", []),
            examPasswords=converted_passwords
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{exam_id}/metadata", response_model=ExamMetadata)
async def get_exam_metadata(exam_id: str):
    """Get exam metadata without questions/answers"""
    try:
        db = get_db()
        exam_doc = db.collection("exams").document(exam_id).get()
        
        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam_data = exam_doc.to_dict()
        
        # Return only metadata fields
        return ExamMetadata(
            id=exam_data.get("id"),
            title=exam_data.get("title"),
            password=exam_data.get("password"),
            description=exam_data.get("description"),
            creator=exam_data.get("creator"),
            timeLimit=exam_data.get("timeLimit"),
            activeExam=exam_data.get("activeExam"),
            shuffleQuestions=exam_data.get("shuffleQuestions"),
            numberOfDisplayedQuestions=exam_data.get("numberOfDisplayedQuestions"),
            accessLimit=exam_data.get("accessLimit"),
            startLimit=exam_data.get("startLimit"),
            endLimit=exam_data.get("endLimit")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{exam_id}/full", response_model=ExamForStudent)
async def get_exam_full(exam_id: str):
    """Get full exam with questions but WITHOUT correct answers (for started exams)"""
    try:
        db = get_db()
        exam_doc = db.collection("exams").document(exam_id).get()
        
        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam_data = exam_doc.to_dict()
        
        # Strip out the 'state' field from all tasks
        groups_without_answers = []
        for group in exam_data.get("groups", []):
            tasks_without_answers = []
            for task in group.get("tasks", []):
                # Create task dict without state field
                task_without_answer = {
                    "id": task.get("id"),
                    "text": task.get("text"),
                    "image": task.get("image")
                }
                tasks_without_answers.append(TaskWithoutAnswer(**task_without_answer))
            
            group_without_answers = GroupWithoutAnswers(
                id=group.get("id"),
                text=group.get("text"),
                tasks=tasks_without_answers
            )
            groups_without_answers.append(group_without_answers)
        
        # Return exam without correct answers
        return ExamForStudent(
            id=exam_data.get("id"),
            title=exam_data.get("title"),
            password=exam_data.get("password"),
            description=exam_data.get("description"),
            creator=exam_data.get("creator"),
            groups=groups_without_answers,
            timestamp=exam_data.get("timestamp"),
            accessLimit=exam_data.get("accessLimit"),
            timeLimit=exam_data.get("timeLimit"),
            startLimit=exam_data.get("startLimit"),
            endLimit=exam_data.get("endLimit"),
            version=exam_data.get("version"),
            lastUpdatedTimestamp=exam_data.get("lastUpdatedTimestamp"),
            createdTimestamp=exam_data.get("createdTimestamp"),
            lastUpdatedBy=exam_data.get("lastUpdatedBy"),
            createdBy=exam_data.get("createdBy"),
            numberOfDisplayedQuestions=exam_data.get("numberOfDisplayedQuestions"),
            activeExam=exam_data.get("activeExam"),
            shuffleQuestions=exam_data.get("shuffleQuestions")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{exam_id}")
async def update_exam(exam_id: str, exam: Exam):
    """Update an existing exam"""
    try:
        db = get_db()
        
        # Verify exam exists
        exam_ref = db.collection("exams").document(exam_id)
        exam_doc = exam_ref.get()
        
        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        old_exam = exam_doc.to_dict()
        
        # Get current version from data/exams
        data_doc = db.collection("data").document("exams").get()
        if not data_doc.exists:
            raise HTTPException(status_code=404, detail="Exam data not found")
        
        data = data_doc.to_dict()
        created_exams = data.get("createdExams", [])
        
        # Find current short exam
        old_short_exam = None
        for ce in created_exams:
            if ce.get("id") == exam_id:
                old_short_exam = ce
                break
        
        if not old_short_exam:
            raise HTTPException(status_code=404, detail="Exam not found in created exams")
        
        # Update exam document
        exam_ref.set(exam.dict())
        
        # Create new short exam with incremented version
        new_short_exam = ShortExam(
            id=exam.id,
            title=exam.title,
            creator=exam.creator,
            timestamp=exam.timestamp,
            password=exam.password,
            accessLimit=exam.accessLimit,
            timeLimit=exam.timeLimit,
            startLimit=exam.startLimit,
            endLimit=exam.endLimit,
            version=old_short_exam.get("version", 0) + 1
        )
        
        # Update createdExams array
        data_ref = db.collection("data").document("exams")
        data_ref.update({
            "createdExams": firestore.ArrayRemove([old_short_exam])
        })
        data_ref.update({
            "createdExams": firestore.ArrayUnion([new_short_exam.dict()])
        })
        
        # Update password if changed
        if old_exam.get("password") != exam.password:
            data_ref.set({
                "examPasswords": {
                    exam.id: {
                        exam.password: firestore.SERVER_TIMESTAMP
                    }
                }
            }, merge=True)
        
        return {"message": "Exam updated successfully", "exam": exam}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{exam_id}")
async def delete_exam(exam_id: str):
    """Delete an exam"""
    try:
        db = get_db()
        
        # Get exam data to find in created exams
        data_doc = db.collection("data").document("exams").get()
        if not data_doc.exists:
            raise HTTPException(status_code=404, detail="Exam data not found")
        
        data = data_doc.to_dict()
        created_exams = data.get("createdExams", [])
        
        # Find the short exam to remove
        short_exam_to_remove = None
        for ce in created_exams:
            if ce.get("id") == exam_id:
                short_exam_to_remove = ce
                break
        
        if not short_exam_to_remove:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        data_ref = db.collection("data").document("exams")
        
        # Remove from createdExams
        data_ref.update({
            "createdExams": firestore.ArrayRemove([short_exam_to_remove])
        })
        
        # Remove password
        data_ref.update({
            f"examPasswords.{exam_id}": firestore.DELETE_FIELD
        })
        
        # Delete exam document
        db.collection("exams").document(exam_id).delete()
        
        return {"message": "Exam deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))