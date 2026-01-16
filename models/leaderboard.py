from pydantic import BaseModel
from typing import Dict, List, Optional


class LeaderboardEntry(BaseModel):
    totalPoints: float
    img: Optional[str] = None


class LeaderboardData(BaseModel):
    leaderboard: Dict[str, LeaderboardEntry]


class LeaderboardUpdateRequest(BaseModel):
    userId: str
    img: Optional[str] = None
    newExamPoints: float
