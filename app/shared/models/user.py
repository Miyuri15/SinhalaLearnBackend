# app/shared/models/user.py

import uuid
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.pricing_plans import DEFAULT_TIER
from app.core.database import Base


USER_ROLE = "user"
ADMIN_ROLE = "admin"
DEFAULT_ROLE = USER_ROLE


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    tier = Column(String, default=DEFAULT_TIER, nullable=False)
    role = Column(String, default=DEFAULT_ROLE, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
