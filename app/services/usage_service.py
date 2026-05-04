# app/services/usage_service.py

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.pricing_plans import PricingPlan
from app.services.pricing_plan_service import PricingPlanService
from app.shared.models.chat_session import ChatSession
from app.shared.models.evaluation_session import EvaluationSession
from app.shared.models.message import Message
from app.shared.models.user import User

logger = logging.getLogger(__name__)

APP_TIMEZONE = ZoneInfo("Asia/Colombo")


class UsageService:
    def __init__(self, db: Session):
        self.db = db

    def _get_user(self, user_id: UUID) -> User:
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user

    def _get_user_plan(self, user_id: UUID) -> tuple[User, PricingPlan]:
        user = self._get_user(user_id)
        return user, PricingPlanService(self.db).get_plan(user.tier)

    def _count_learning_requests_since(self, user_id: UUID, threshold: datetime) -> int:
        return (
            self.db.query(func.count(Message.id))
            .join(ChatSession, Message.session_id == ChatSession.id)
            .filter(ChatSession.user_id == user_id)
            .filter(ChatSession.mode == "learning")
            .filter(Message.role == "user")
            .filter(Message.created_at >= threshold)
            .scalar()
            or 0
        )

    def _count_evaluation_sessions_since(self, user_id: UUID, threshold: datetime) -> int:
        return (
            self.db.query(func.count(EvaluationSession.id))
            .join(ChatSession, EvaluationSession.session_id == ChatSession.id)
            .filter(ChatSession.user_id == user_id)
            .filter(EvaluationSession.created_at >= threshold)
            .scalar()
            or 0
        )

    def _today_start_utc(self, now: Optional[datetime] = None) -> datetime:
        local_now = now or datetime.now(APP_TIMEZONE)
        if local_now.tzinfo is None:
            local_now = local_now.replace(tzinfo=APP_TIMEZONE)
        local_now = local_now.astimezone(APP_TIMEZONE)
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        return local_start.astimezone(timezone.utc)

    def _tomorrow_start_utc(self, now: Optional[datetime] = None) -> datetime:
        return self._today_start_utc(now) + timedelta(days=1)

    def _current_hour_start_utc(self, now: Optional[datetime] = None) -> datetime:
        local_now = now or datetime.now(APP_TIMEZONE)

        if local_now.tzinfo is None:
            local_now = local_now.replace(tzinfo=APP_TIMEZONE)

        local_now = local_now.astimezone(APP_TIMEZONE)

        local_hour_start = local_now.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

        return local_hour_start.astimezone(timezone.utc)


    def _next_hour_start_utc(self, now: Optional[datetime] = None) -> datetime:
        return self._current_hour_start_utc(now) + timedelta(hours=1)

    def get_user_tier_limit(self, tier: str) -> int:
        """Backward-compatible helper returning daily evaluation-session limit."""
        return PricingPlanService(self.db).get_plan(tier).limits.evaluation_sessions_per_day

    def check_learning_request_limit(self, user_id: UUID) -> bool:
        user, plan = self._get_user_plan(user_id)
        limit = plan.limits.learning_requests_per_hour
        now = datetime.now(timezone.utc)
        threshold = self._current_hour_start_utc(now)
        count = self._count_learning_requests_since(user_id, threshold)

        if count >= limit:
            logger.warning(
                "User %s (Tier: %s) reached learning request limit: %s/%s",
                user_id,
                user.tier,
                count,
                limit,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Learning request limit reached for your {plan.name} "
                    f"({limit} requests per hour). Please try again later or upgrade your plan."
                ),
            )

        return True

    def check_evaluation_session_limit(self, user_id: UUID) -> bool:
        user, plan = self._get_user_plan(user_id)
        limit = plan.limits.evaluation_sessions_per_day
        threshold = self._today_start_utc()
        count = self._count_evaluation_sessions_since(user_id, threshold)

        if count >= limit:
            logger.warning(
                "User %s (Tier: %s) reached evaluation session limit: %s/%s",
                user_id,
                user.tier,
                count,
                limit,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Evaluation session limit reached for your {plan.name} "
                    f"({limit} sessions per day). Please try again tomorrow or upgrade your plan."
                ),
            )

        return True

    def check_evaluations_per_session_limit(
        self,
        user_id: UUID,
        answer_resource_ids: Sequence[UUID],
        existing_answer_count: int = 0,
    ) -> bool:
        user, plan = self._get_user_plan(user_id)
        limit = plan.limits.evaluations_per_session

        if limit is None:
            return True

        requested_count = len(answer_resource_ids)
        total_count = existing_answer_count + requested_count
        if total_count > limit:
            logger.warning(
                "User %s (Tier: %s) exceeded evaluations per session: %s/%s",
                user_id,
                user.tier,
                total_count,
                limit,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Evaluation limit reached for your {plan.name} "
                    f"({limit} evaluations per session). Please reduce answer scripts or upgrade your plan."
                ),
            )

        return True

    def check_evaluation_limit(self, user_id: UUID) -> bool:
        """Backward-compatible alias for the daily evaluation-session limit."""
        return self.check_evaluation_session_limit(user_id)

    def get_usage_summary(self, user_id: UUID) -> dict:
        user, plan = self._get_user_plan(user_id)
        now = datetime.now(timezone.utc)
        learning_threshold = self._current_hour_start_utc(now)
        day_threshold = self._today_start_utc(now)

        learning_used = self._count_learning_requests_since(user_id, learning_threshold)
        evaluation_used = self._count_evaluation_sessions_since(user_id, day_threshold)
        learning_limit = plan.limits.learning_requests_per_hour
        evaluation_limit = plan.limits.evaluation_sessions_per_day

        return {
            "tier": plan.tier,
            "plan_name": plan.name,
            "limits": {
                "learning_requests_per_hour": learning_limit,
                "evaluation_sessions_per_day": evaluation_limit,
                "evaluations_per_session": plan.limits.evaluations_per_session,
                "allow_evaluation_overage": plan.limits.allow_evaluation_overage,
            },
            "learning_requests": {
                "used": learning_used,
                "limit": learning_limit,
                "remaining": max(learning_limit - learning_used, 0),
                "reset_at": self._next_hour_start_utc(now),
            },
            "evaluation_sessions": {
                "used": evaluation_used,
                "limit": evaluation_limit,
                "remaining": max(evaluation_limit - evaluation_used, 0),
                "reset_at": self._tomorrow_start_utc(now),
            },
            "evaluations_per_session_limit": plan.limits.evaluations_per_session,
            "allow_evaluation_overage": plan.limits.allow_evaluation_overage,
        }
