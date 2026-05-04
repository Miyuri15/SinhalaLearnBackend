# app/core/pricing_plans.py

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


BASIC_TIER = "basic"
INTERMEDIATE_TIER = "intermediate"
ENTERPRISE_TIER = "enterprise"

DEFAULT_TIER = BASIC_TIER

TIER_ALIASES: Dict[str, str] = {
    "normal": BASIC_TIER,
    "basic": BASIC_TIER,
    "classroom": INTERMEDIATE_TIER,
    "intermediate": INTERMEDIATE_TIER,
    "institution": ENTERPRISE_TIER,
    "enterprise": ENTERPRISE_TIER,
}


@dataclass(frozen=True)
class PlanLimits:
    learning_requests_per_hour: int
    evaluation_sessions_per_day: int
    evaluations_per_session: Optional[int]
    allow_evaluation_overage: bool = False


@dataclass(frozen=True)
class PricingPlan:
    tier: str
    name: str
    price_label: str
    description: str
    badge: str
    features: Tuple[str, ...]
    cta: str
    note: str
    limits: PlanLimits
    is_popular: bool = False


PRICING_PLANS: Dict[str, PricingPlan] = {
    BASIC_TIER: PricingPlan(
        tier=BASIC_TIER,
        name="Basic Plan",
        price_label="Free / forever",
        description="A lightweight plan for getting started with Learning Mode",
        badge="Starter",
        features=(
            "Learning mode: 5 requests per hour",
            "Evaluation mode: 1 session per day",
            "Up to 10 evaluations per session",
            "Perfect for getting started",
        ),
        cta="Start Free",
        note="No credit card required",
        limits=PlanLimits(
            learning_requests_per_hour=5,
            evaluation_sessions_per_day=1,
            evaluations_per_session=10,
        ),
    ),
    INTERMEDIATE_TIER: PricingPlan(
        tier=INTERMEDIATE_TIER,
        name="Intermediate Plan",
        price_label="5000 LKR / tier",
        description="For regular users who need more daily usage",
        badge="Most Popular",
        features=(
            "Learning mode: 20 requests per hour",
            "Evaluation mode: 5 sessions per day",
            "Built for steady classroom or personal use",
            "Priority access during busy periods",
        ),
        cta="Choose Intermediate",
        note="Usage resets apply",
        limits=PlanLimits(
            learning_requests_per_hour=20,
            evaluation_sessions_per_day=5,
            evaluations_per_session=None,
        ),
        is_popular=True,
    ),
    ENTERPRISE_TIER: PricingPlan(
        tier=ENTERPRISE_TIER,
        name="Enterprise Plan",
        price_label="10000 LKR onwards / tier",
        description="For teams and institutions that need the highest limits",
        badge="Best for Scale",
        features=(
            "Learning mode: 50 requests per hour",
            "Evaluation mode: 10 sessions per day",
            "Next evaluations are charged",
            "Designed for larger deployments",
        ),
        cta="Contact Sales",
        note="Usage resets apply",
        limits=PlanLimits(
            learning_requests_per_hour=50,
            evaluation_sessions_per_day=10,
            evaluations_per_session=None,
            allow_evaluation_overage=True,
        ),
    ),
}


def normalize_tier(tier: Optional[str]) -> str:
    if not tier:
        return DEFAULT_TIER
    return TIER_ALIASES.get(tier.strip().lower(), DEFAULT_TIER)


def get_pricing_plan(tier: Optional[str]) -> PricingPlan:
    return PRICING_PLANS[normalize_tier(tier)]


def list_pricing_plans() -> Tuple[PricingPlan, ...]:
    return tuple(PRICING_PLANS[tier] for tier in (BASIC_TIER, INTERMEDIATE_TIER, ENTERPRISE_TIER))
