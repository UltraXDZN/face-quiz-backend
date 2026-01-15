from pydantic import BaseModel
from typing import List, Dict, Optional


class Task(BaseModel):
    id: str
    title: str
    state: Optional[bool] = None
    image: Optional[str] = None


class Group(BaseModel):
    id: str
    name: str
    tasks: List[Task]


class Exam(BaseModel):
    id: str
    title: str
    password: str
    description: Optional[str] = ""
    creator: str
    groups: List[Group]
    timestamp: int


class ShortExam(BaseModel):
    id: str
    title: str
    creator: str
    timestamp: int
    version: int


class ExamPasswordsUpdate(BaseModel):
    password: str


class ExamsDataResponse(BaseModel):
    createdExams: List[ShortExam]
    examPasswords: Dict[str, Dict[str, int]]
