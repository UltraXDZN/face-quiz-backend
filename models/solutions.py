from pydantic import BaseModel
from typing import List, Optional


class Solution(BaseModel):
    id: str
    state: Optional[bool] = None


class SolutionSubmission(BaseModel):
    exam_id: str
    password: str
    email: str
    solutions: List[Solution]


class UserSolutionResponse(BaseModel):
    email: str
    username: str
    firstName: str
    lastName: str
    percent: int
    creationYear: int
    admin: bool

class Solution(BaseModel):
    id: str
    state: Optional[bool] = None


class Task(BaseModel):
    id: str
    state: Optional[bool] = None


class Group(BaseModel):
    tasks: List[Task]


class ExamInfo(BaseModel):
    groups: List[Group]
