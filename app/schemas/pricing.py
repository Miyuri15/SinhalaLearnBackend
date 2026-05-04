# app/schemas/pricing.py

from typing import List, Optional

from pydantic import BaseModel, Field


class PlanLimitsResponse(BaseModel):
    learning_requests_per_hour: int
    evaluation_sessions_per_day: int
    evaluations_per_session: Optional[int]
    allow_evaluation_overage: bool


class PricingPlanResponse(BaseModel):
    tier: str
    name: str
    price_label: str
    description: str
    badge: str
    features: List[str]
    cta: str
    note: str
    limits: PlanLimitsResponse
    is_popular: bool


class PricingPlansResponse(BaseModel):
    plans: List[PricingPlanResponse]


class PlanLimitsUpdate(BaseModel):
    learning_requests_per_hour: Optional[int] = Field(default=None, ge=0)
    evaluation_sessions_per_day: Optional[int] = Field(default=None, ge=0)
    evaluations_per_session: Optional[int] = Field(default=None, ge=0)
    allow_evaluation_overage: Optional[bool] = None


class PricingPlanUpdate(BaseModel):
    name: Optional[str] = None
    price_label: Optional[str] = None
    description: Optional[str] = None
    badge: Optional[str] = None
    features: Optional[List[str]] = None
    cta: Optional[str] = None
    note: Optional[str] = None
    limits: Optional[PlanLimitsUpdate] = None
    is_popular: Optional[bool] = None
    is_active: Optional[bool] = None
