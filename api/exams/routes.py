import os
from fastapi import APIRouter, HTTPException, status, Query
from api.shuffle_utils import shuffle_with_seed
from fastapi.responses import JSONResponse, StreamingResponse
from google.cloud import firestore
from google.auth.credentials import AnonymousCredentials
from io import BytesIO
import zipfile
import json
import re
import uuid
from models.exams import (
    Exam, ShortExam, ExamPasswordsUpdate, ExamsDataResponse, ExamMetadata,
    ExamForStudent, GroupWithoutAnswers, TaskWithoutAnswer
)
from typing import List

router = APIRouter(prefix="/exams", tags=["exams"])


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
def create_exam(exam: Exam):
    """Create a new exam"""
    try:
        db = get_db()
        
        # Auto-generate missing IDs
        for group in exam.groups:
            if not group.id:
                group.id = str(uuid.uuid4())
            for task in group.tasks:
                if not task.id:
                    task.id = str(uuid.uuid4())
        
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
def get_exams_data():
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
def get_exam_metadata(exam_id: str):
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
            shuffleTasks=exam_data.get("shuffleTasks"),
            numberOfDisplayedQuestions=exam_data.get("numberOfDisplayedQuestions"),
            numberOfDisplayedTasks=exam_data.get("numberOfDisplayedTasks"),
            accessLimit=exam_data.get("accessLimit"),
            startLimit=exam_data.get("startLimit"),
            endLimit=exam_data.get("endLimit")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{exam_id}/full", response_model=ExamForStudent)
def get_exam_full(exam_id: str, email: str = Query(default="")):
    """Get full exam with questions but WITHOUT correct answers (for started exams).
    If shuffleQuestions is enabled, groups and tasks are shuffled deterministically per student email."""
    try:
        db = get_db()
        exam_doc = db.collection("exams").document(exam_id).get()
        
        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam_data = exam_doc.to_dict()
        raw_groups = exam_data.get("groups", [])
        shuffle_groups = exam_data.get("shuffleQuestions", False)
        shuffle_tasks = exam_data.get("shuffleTasks", False)
        num_displayed = exam_data.get("numberOfDisplayedQuestions")
        num_tasks_displayed = exam_data.get("numberOfDisplayedTasks")
        
        # Prepare groups (shuffle if enabled)
        prepared_groups = list(raw_groups)
        if shuffle_groups and email:
            prepared_groups = shuffle_with_seed(prepared_groups, email, exam_id)
            if num_displayed is not None:
                try:
                    num_int = int(num_displayed)
                    if num_int > 0:
                        prepared_groups = prepared_groups[:num_int]
                except (TypeError, ValueError):
                    pass

        # Strip out the 'state' field from all tasks
        groups_without_answers = []
        for group in prepared_groups:
            raw_tasks = group.get("tasks", [])

            # Shuffle tasks within group if enabled (independent of group shuffle)
            if shuffle_tasks and email:
                raw_tasks = shuffle_with_seed(list(raw_tasks), email, exam_id, group.get("id", ""))
            
            # Limit tasks per group if specified
            if num_tasks_displayed is not None:
                try:
                    num_tasks_int = int(num_tasks_displayed)
                    if num_tasks_int > 0:
                        raw_tasks = raw_tasks[:num_tasks_int]
                except (TypeError, ValueError):
                    pass
            
            tasks_without_answers = []
            for task in raw_tasks:
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
        
        # Return exam
        return ExamForStudent(
            id=exam_data.get("id"),
            title=exam_data.get("title"),
            password=exam_data.get("password"),
            description=exam_data.get("description"),
            creator=exam_data.get("creator"),
            timeLimit=exam_data.get("timeLimit"),
            activeExam=exam_data.get("activeExam"),
            shuffleQuestions=exam_data.get("shuffleQuestions"),
            shuffleTasks=exam_data.get("shuffleTasks"),
            numberOfDisplayedQuestions=exam_data.get("numberOfDisplayedQuestions"),
            numberOfDisplayedTasks=exam_data.get("numberOfDisplayedTasks"),
            accessLimit=exam_data.get("accessLimit"),
            startLimit=exam_data.get("startLimit"),
            endLimit=exam_data.get("endLimit"),
            groups=groups_without_answers
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{exam_id}/admin", response_model=Exam)
def get_exam_admin(exam_id: str):
    """Get complete exam with ALL data including correct answers (for admin/editing)"""
    try:
        db = get_db()
        exam_doc = db.collection("exams").document(exam_id).get()
        
        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")
        
        exam_data = exam_doc.to_dict()
        
        # Return complete exam data as-is with all fields
        return Exam(**exam_data)
    except HTTPException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{exam_id}")
def update_exam(exam_id: str, exam: Exam):
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


@router.post("/{exam_id}/duplicate", status_code=status.HTTP_201_CREATED)
def duplicate_exam(exam_id: str):
    """Duplicate an existing exam with a new ID"""
    try:
        db = get_db()

        # Get original exam
        exam_ref = db.collection("exams").document(exam_id)
        exam_doc = exam_ref.get()

        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")

        exam_data = exam_doc.to_dict()

        # Generate new ID
        new_id = str(uuid.uuid4())

        # Create new exam document with new ID
        new_exam_data = {**exam_data, "id": new_id}
        db.collection("exams").document(new_id).set(new_exam_data)

        # Create new ShortExam and add to createdExams
        new_short_exam = {
            "id": new_id,
            "title": exam_data.get("title", "") + " (kopija)",
            "creator": exam_data.get("creator"),
            "timestamp": exam_data.get("timestamp"),
            "password": exam_data.get("password", ""),
            "accessLimit": exam_data.get("accessLimit", False),
            "timeLimit": exam_data.get("timeLimit", 1),
            "startLimit": exam_data.get("startLimit", None),
            "endLimit": exam_data.get("endLimit", None),
            "version": 0
        }

        data_ref = db.collection("data").document("exams")
        data_ref.update({
            "createdExams": firestore.ArrayUnion([new_short_exam])
        })

        # Also copy the password entry
        data_doc = db.collection("data").document("exams").get()
        if data_doc.exists:
            data = data_doc.to_dict()
            exam_passwords = data.get("examPasswords", {})
            if exam_id in exam_passwords:
                data_ref.set({
                    "examPasswords": {
                        new_id: exam_passwords[exam_id]
                    }
                }, merge=True)

        return {"message": "Exam duplicated successfully", "newExamId": new_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{exam_id}")
def delete_exam(exam_id: str):
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


def _sanitize_filename(value: str) -> str:
    value = re.sub(r"[^\w\-.]+", "_", value.strip(), flags=re.UNICODE)
    return value.strip("._") or "group"


def _split_task_text(task_text: str):
    marker = "(Pojašnjenje"
    if marker not in task_text:
        return task_text.strip(), ""
    statement, explanation = task_text.split(marker, 1)
    return statement.strip(), f"{marker}{explanation}".strip()


def _build_group_markdown(group: dict) -> str:
    lines = []
    group_text = str(group.get("text", "") or "").strip()
    if group_text:
        lines.append(group_text)
        lines.append("")

    for task in group.get("tasks", []) or []:
        statement, explanation = _split_task_text(str(task.get("text", "") or ""))
        correctness = "Točno" if task.get("state") else "Netočno"
        explanation_text = f" {explanation}" if explanation else ""
        lines.append(f"1. {statement} [{correctness}] {explanation_text}".rstrip())

    return "\n".join(lines).strip()


@router.get("/{exam_id}/export")
def export_exam(exam_id: str, format: str = Query(default="json")):
    """Export an exam as JSON or as a zip of Markdown group files"""
    try:
        db = get_db()
        exam_doc = db.collection("exams").document(exam_id).get()

        if not exam_doc.exists:
            raise HTTPException(status_code=404, detail="Exam not found")

        exam_data = exam_doc.to_dict()

        if format.lower() == "md":
            buffer = BytesIO()
            exam_title = _sanitize_filename(str(exam_data.get("title", "exam")))
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                groups = exam_data.get("groups", []) or []
                for index, group in enumerate(groups, start=1):
                    group_title = _sanitize_filename(str(group.get("text", f"group_{index}"))[:60])
                    filename = f"{index:02d}_{group_title or 'group'}.md"
                    zf.writestr(filename, _build_group_markdown(group))

            buffer.seek(0)
            return StreamingResponse(
                buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename={exam_title}_{exam_id}_md.zip",
                },
            )

        return JSONResponse(
            content=exam_data,
            headers={
                "Content-Disposition": f"attachment; filename=exam_{exam_id}.json",
                "Content-Type": "application/json",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))