from pydantic import BaseModel
from typing import List, Dict, Optional


class Task(BaseModel):
    id: str
    text: str  # Changed from title to text
    state: Optional[bool] = None
    image: Optional[str] = None
    positive_points: Optional[float] = None
    negative_points: Optional[float] = None


class Group(BaseModel):
    id: str
    text: str  # Changed from name to text
    tasks: List[Task]


class Exam(BaseModel):
    id: str
    title: str
    password: str
    description: Optional[str] = ""
    creator: Optional[str] = None  # Made optional
    groups: List[Group]
    timestamp: Optional[int] = None  # Made optional
    # Additional fields from Firestore
    accessLimit: Optional[bool] = None
    timeLimit: Optional[int] = None
    startLimit: Optional[int] = None
    endLimit: Optional[int] = None
    version: Optional[int] = None
    lastUpdatedTimestamp: Optional[int] = None
    createdTimestamp: Optional[int] = None
    lastUpdatedBy: Optional[str] = None
    createdBy: Optional[str] = None
    numberOfDisplayedQuestions: Optional[int] = None
    activeExam: Optional[bool] = None
    shuffleQuestions: Optional[bool] = None


class ShortExam(BaseModel):
    id: str
    title: str
    creator: Optional[str] = None
    timestamp: Optional[int] = None
    password: Optional[str] = None
    accessLimit: Optional[bool] = None
    timeLimit: Optional[int] = None
    startLimit: Optional[int] = None
    endLimit: Optional[int] = None
    version: int


class ExamPasswordsUpdate(BaseModel):
    password: str


class ExamsDataResponse(BaseModel):
    createdExams: List[ShortExam]
    examPasswords: Dict[str, Dict[str, int]]
