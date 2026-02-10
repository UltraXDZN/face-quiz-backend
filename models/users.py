from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class AccessedExam(BaseModel):
    id: str
    status: str
    password: str
    pointsEarned: float
    totalPoints: float
    lastAccessed: int
    lastFinished: int


class UserUISettings(BaseModel):
    selectedTheme: str = "default"
    colorThemes: List[Dict[str, Any]] = Field(default_factory=list)
    fixedSidebar: bool = False
    showSidebarOnHover: bool = False
    showSettingsOnHover: bool = False
    showQuickActionsOnHover: bool = False
    showThemeButton: bool = True
    showPixelartButton: bool = True
    glassEffect: bool = False
    blurStrength: float = 0
    cardOpacity: float = 1
    radiusSize: float = 1


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
    UISettings: Optional[UserUISettings] = None


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
