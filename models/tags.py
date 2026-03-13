from pydantic import BaseModel, Field
from typing import List, Optional


class Tag(BaseModel):
    id: str
    name: str
    task_IDs: List[str] = Field(default_factory=list)
    misc: str = "-"
    typeVersion: int = 1


class TagCreate(BaseModel):
    name: str
    task_IDs: List[str] = Field(default_factory=list)
    misc: str = "-"
    typeVersion: int = 1
