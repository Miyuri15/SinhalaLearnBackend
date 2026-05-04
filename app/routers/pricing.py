# app/routers/pricing.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.pricing_plans import PRICING_PLANS, normalize_tier
from app.core.security import require_admin_user
from app.schemas.pricing import PricingPlanResponse, PricingPlansResponse, PricingPlanUpdate
from app.services.pricing_plan_service import PricingPlanService

router = APIRouter()


@router.get("/plans", response_model=PricingPlansResponse)
def get_pricing_plans(db: Session = Depends(get_db)):
    service = PricingPlanService(db)
    return {"plans": [service.to_response(plan) for plan in service.list_plans()]}


@router.get("/admin/plans", response_model=PricingPlansResponse)
def get_admin_pricing_plans(
    _admin_user=Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    service = PricingPlanService(db)
    return {"plans": [service.to_response(plan) for plan in service.list_plans(include_inactive=True)]}


@router.patch("/admin/plans/{tier}", response_model=PricingPlanResponse)
def update_pricing_plan(
    tier: str,
    payload: PricingPlanUpdate,
    admin_user=Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    normalized_tier = normalize_tier(tier)
    if tier.strip().lower() not in PRICING_PLANS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tier. Use one of: basic, intermediate, enterprise",
        )

    plan = PricingPlanService(db).update_plan(normalized_tier, payload, admin_user.id)
    return PricingPlanService.to_response(plan)
