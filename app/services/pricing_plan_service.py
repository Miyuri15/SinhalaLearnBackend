# app/services/pricing_plan_service.py

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.pricing_plans import (
    PRICING_PLANS,
    PricingPlan,
    PlanLimits,
    list_pricing_plans,
    normalize_tier,
)
from app.shared.models.pricing_plan import PricingPlanModel

logger = logging.getLogger(__name__)


class PricingPlanService:
    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def to_response(plan: PricingPlan) -> dict:
        return {
            "tier": plan.tier,
            "name": plan.name,
            "price_label": plan.price_label,
            "description": plan.description,
            "badge": plan.badge,
            "features": list(plan.features),
            "cta": plan.cta,
            "note": plan.note,
            "limits": {
                "learning_requests_per_hour": plan.limits.learning_requests_per_hour,
                "evaluation_sessions_per_day": plan.limits.evaluation_sessions_per_day,
                "evaluations_per_session": plan.limits.evaluations_per_session,
                "allow_evaluation_overage": plan.limits.allow_evaluation_overage,
            },
            "is_popular": plan.is_popular,
        }

    @staticmethod
    def _model_to_plan(model: PricingPlanModel) -> PricingPlan:
        return PricingPlan(
            tier=model.tier_key,
            name=model.name,
            price_label=model.price_label,
            description=model.description,
            badge=model.badge,
            features=tuple(model.features or ()),
            cta=model.cta,
            note=model.note,
            limits=PlanLimits(
                learning_requests_per_hour=model.learning_requests_per_hour,
                evaluation_sessions_per_day=model.evaluation_sessions_per_day,
                evaluations_per_session=model.evaluations_per_session,
                allow_evaluation_overage=model.allow_evaluation_overage,
            ),
            is_popular=model.is_popular,
        )

    @staticmethod
    def _defaults_to_model(plan: PricingPlan) -> PricingPlanModel:
        return PricingPlanModel(
            tier_key=plan.tier,
            name=plan.name,
            price_label=plan.price_label,
            description=plan.description,
            badge=plan.badge,
            features=list(plan.features),
            cta=plan.cta,
            note=plan.note,
            learning_requests_per_hour=plan.limits.learning_requests_per_hour,
            evaluation_sessions_per_day=plan.limits.evaluation_sessions_per_day,
            evaluations_per_session=plan.limits.evaluations_per_session,
            allow_evaluation_overage=plan.limits.allow_evaluation_overage,
            is_popular=plan.is_popular,
            is_active=True,
        )

    def ensure_default_plans(self) -> None:
        for plan in list_pricing_plans():
            existing = self.db.query(PricingPlanModel).filter(PricingPlanModel.tier_key == plan.tier).first()
            if existing:
                continue
            self.db.add(self._defaults_to_model(plan))
        self.db.commit()

    def list_plans(self, include_inactive: bool = False) -> list[PricingPlan]:
        try:
            query = self.db.query(PricingPlanModel)
            if not include_inactive:
                query = query.filter(PricingPlanModel.is_active.is_(True))
            models = query.all()
            if not models:
                self.ensure_default_plans()
                query = self.db.query(PricingPlanModel)
                if not include_inactive:
                    query = query.filter(PricingPlanModel.is_active.is_(True))
                models = query.all()
        except SQLAlchemyError as exc:
            logger.warning("Falling back to code-managed pricing plans: %s", exc)
            self.db.rollback()
            return list(list_pricing_plans())

        by_tier = {model.tier_key: self._model_to_plan(model) for model in models}
        ordered_tiers = [plan.tier for plan in list_pricing_plans()]
        ordered = [by_tier[tier] for tier in ordered_tiers if tier in by_tier]
        ordered.extend(plan for tier, plan in by_tier.items() if tier not in ordered_tiers)
        return ordered

    def get_plan(self, tier: Optional[str]) -> PricingPlan:
        normalized_tier = normalize_tier(tier)
        try:
            model = (
                self.db.query(PricingPlanModel)
                .filter(PricingPlanModel.tier_key == normalized_tier)
                .first()
            )
            if not model:
                self.ensure_default_plans()
                model = (
                    self.db.query(PricingPlanModel)
                    .filter(PricingPlanModel.tier_key == normalized_tier)
                    .first()
                )
        except SQLAlchemyError as exc:
            logger.warning("Falling back to code-managed pricing plan for %s: %s", normalized_tier, exc)
            self.db.rollback()
            return PRICING_PLANS[normalized_tier]

        if not model:
            return PRICING_PLANS[normalized_tier]
        return self._model_to_plan(model)

    def update_plan(self, tier: str, payload, admin_user_id: UUID) -> PricingPlan:
        normalized_tier = normalize_tier(tier)
        if normalized_tier not in PRICING_PLANS:
            raise ValueError("Invalid tier")

        self.ensure_default_plans()
        model = (
            self.db.query(PricingPlanModel)
            .filter(PricingPlanModel.tier_key == normalized_tier)
            .first()
        )
        if not model:
            model = self._defaults_to_model(PRICING_PLANS[normalized_tier])
            self.db.add(model)

        update_data = payload.model_dump(exclude_unset=True)
        limits = update_data.pop("limits", None)
        for field, value in update_data.items():
            setattr(model, field, value)
        if limits is not None:
            for field, value in limits.items():
                setattr(model, field, value)
        model.updated_by = admin_user_id

        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._model_to_plan(model)
