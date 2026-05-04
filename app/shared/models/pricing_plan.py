# app/shared/models/pricing_plan.py

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class PricingPlanModel(Base):
    __tablename__ = "pricing_plans"

    tier_key = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    price_label = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    badge = Column(String, nullable=False)
    features = Column(JSON, nullable=False)
    cta = Column(String, nullable=False)
    note = Column(Text, nullable=False)
    learning_requests_per_hour = Column(Integer, nullable=False)
    evaluation_sessions_per_day = Column(Integer, nullable=False)
    evaluations_per_session = Column(Integer, nullable=True)
    allow_evaluation_overage = Column(Boolean, default=False, nullable=False)
    is_popular = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
