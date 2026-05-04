# app/schemas/usage.py

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.pricing import PlanLimitsResponse


class UsageWindowResponse(BaseModel):
    used: int
    limit: int
    remaining: int
    reset_at: datetime | None = None


class UsageSummaryResponse(BaseModel):
    tier: str
    plan_name: str
    limits: PlanLimitsResponse
    learning_requests: UsageWindowResponse
    evaluation_sessions: UsageWindowResponse
    evaluations_per_session_limit: Optional[int]
    allow_evaluation_overage: bool
