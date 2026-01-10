from pydantic import BaseModel, Field
from typing import List, Optional


class AccessedExam(BaseModel):
    id: str
    status: str
    password: str
    pointsEarned: float
    totalPoints: float
    lastAccessed: int
    lastFinished: int


class User(BaseModel):
    username: str
    email: str
    name: str
    surname: str
    jmbag: str
    photoURL: Optional[str] = None
    creationYear: int
    year: str
    type: str
    admin: bool = False
    tags: List[str] = Field(default_factory=list)
    acessedExams: List[AccessedExam] = Field(default_factory=list)


class UserCreate(BaseModel):
    username: str
    email: str
    name: str
    surname: str
    jmbag: str
    year: str
    type: str
    photoURL: Optional[str] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    surname: Optional[str] = None
    photoURL: Optional[str] = None
    year: Optional[str] = None
    tags: Optional[List[str]] = None
