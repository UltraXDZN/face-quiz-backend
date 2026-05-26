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
    name: str
    surname: str
    percent: str
    creationYear: int
    admin: bool
    jmbag: str = ""

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


class TaskResult(BaseModel):
    """Result for a single task"""
    id: str
    text: str
    user_answer: Optional[bool] = None
    correct_answer: Optional[bool] = None
    is_correct: bool
    points_earned: float
    image: Optional[str] = None


class GroupResult(BaseModel):
    """Results for a group"""
    id: str
    text: str
    tasks: List[TaskResult]


class ExamResultResponse(BaseModel):
    """Complete exam result with scores and correct answers"""
    exam_id: str
    exam_title: str
    exam_password: str
    user_email: str
    groups: List[GroupResult]
    total_points: float
    achieved_points: float
    correct_count: int
    incorrect_count: int
    unanswered_count: int
    total_tasks: int
    leaderboard_message: str
