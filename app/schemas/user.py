# app/schemas/user.py

from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from uuid import UUID
from typing import List

class UserCreate(BaseModel):
    email: str
    full_name: Optional[str] = None
    password: str


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None


class UserTierUpdate(BaseModel):
    tier: str


class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: Optional[str]
    tier: str
    role: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class UserListResponse(BaseModel):
    items: List[UserResponse]
    total: int
    limit: int
    offset: int
