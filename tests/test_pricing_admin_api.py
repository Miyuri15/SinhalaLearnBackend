# tests/test_pricing_admin_api.py
import importlib.util
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient


def load_router_module(module_name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pricing = load_router_module("pricing_router_under_test", "app/routers/pricing.py")
users = load_router_module("users_router_under_test", "app/routers/users.py")


def create_test_app():
    app = FastAPI()
    app.include_router(pricing.router, prefix="/api/v1/pricing")
    app.include_router(users.router, prefix="/api/v1/users")
    return app


def test_pricing_plans_endpoint_returns_frontend_metadata():
    client = TestClient(create_test_app())

    response = client.get("/api/v1/pricing/plans")

    assert response.status_code == 200
    data = response.json()
    plans = data["plans"]
    assert [plan["tier"] for plan in plans] == ["basic", "intermediate", "enterprise"]

    basic = plans[0]
    assert basic["name"] == "Basic Plan"
    assert basic["badge"] == "Starter"
    assert basic["cta"] == "Start Free"
    assert basic["note"] == "No credit card required"
    assert basic["limits"]["learning_requests_per_hour"] == 5
    assert basic["limits"]["evaluation_sessions_per_day"] == 1
    assert basic["limits"]["evaluations_per_session"] == 10

    intermediate = plans[1]
    assert intermediate["is_popular"] is True
    assert intermediate["limits"]["learning_requests_per_hour"] == 20

    enterprise = plans[2]
    assert enterprise["limits"]["learning_requests_per_hour"] == 50
    assert enterprise["limits"]["allow_evaluation_overage"] is True


def test_unauthenticated_user_cannot_update_tier():
    client = TestClient(create_test_app())

    response = client.patch(
        f"/api/v1/users/{uuid4()}/tier",
        json={"tier": "intermediate"},
    )

    assert response.status_code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}


def test_authenticated_non_admin_cannot_update_tier():
    app = create_test_app()

    def reject_non_admin():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

    app.dependency_overrides[users.require_admin_user] = reject_non_admin
    client = TestClient(app)

    response = client.patch(
        f"/api/v1/users/{uuid4()}/tier",
        json={"tier": "intermediate"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_admin_can_update_user_tier(monkeypatch):
    app = create_test_app()
    target_user_id = uuid4()

    class FakeUser:
        id = target_user_id
        email = "student@example.com"
        full_name = "Student"
        tier = "basic"
        role = "user"
        created_at = "2026-04-30T00:00:00Z"
        updated_at = None

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def get_user(self, user_id):
            return FakeUser() if user_id == target_user_id else None

        def update_user_tier(self, user, tier):
            user.tier = tier
            return user

    def allow_admin():
        return object()

    def fake_db():
        yield object()

    monkeypatch.setattr(users, "UserService", FakeUserService)
    app.dependency_overrides[users.require_admin_user] = allow_admin
    app.dependency_overrides[users.get_db] = fake_db
    client = TestClient(app)

    response = client.patch(
        f"/api/v1/users/{target_user_id}/tier",
        json={"tier": "intermediate"},
    )

    assert response.status_code == 200
    assert response.json()["tier"] == "intermediate"


def test_admin_tier_update_rejects_invalid_tier(monkeypatch):
    app = create_test_app()

    def allow_admin():
        return object()

    def fake_db():
        yield object()

    app.dependency_overrides[users.require_admin_user] = allow_admin
    app.dependency_overrides[users.get_db] = fake_db
    client = TestClient(app)

    response = client.patch(
        f"/api/v1/users/{uuid4()}/tier",
        json={"tier": "gold"},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_unauthenticated_user_cannot_list_admin_pricing_plans():
    client = TestClient(create_test_app())

    response = client.get("/api/v1/pricing/admin/plans")

    assert response.status_code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}


def test_admin_can_update_pricing_plan_limits(monkeypatch):
    app = create_test_app()
    admin_id = uuid4()

    class FakeAdmin:
        id = admin_id

    class FakePlan:
        tier = "basic"
        name = "Basic Plan"
        price_label = "Free / forever"
        description = "A lightweight plan for getting started with Learning Mode"
        badge = "Starter"
        features = ("Learning mode: 7 requests per hour",)
        cta = "Start Free"
        note = "No credit card required"
        is_popular = False

        class limits:
            learning_requests_per_hour = 7
            evaluation_sessions_per_day = 2
            evaluations_per_session = 12
            allow_evaluation_overage = False

    class FakePricingPlanService:
        def __init__(self, db):
            self.db = db

        def update_plan(self, tier, payload, admin_user_id):
            assert tier == "basic"
            assert admin_user_id == admin_id
            assert payload.limits.learning_requests_per_hour == 7
            assert payload.limits.evaluation_sessions_per_day == 2
            return FakePlan()

        @staticmethod
        def to_response(plan):
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

    def allow_admin():
        return FakeAdmin()

    def fake_db():
        yield object()

    monkeypatch.setattr(pricing, "PricingPlanService", FakePricingPlanService)
    app.dependency_overrides[pricing.require_admin_user] = allow_admin
    app.dependency_overrides[pricing.get_db] = fake_db
    client = TestClient(app)

    response = client.patch(
        "/api/v1/pricing/admin/plans/basic",
        json={
            "limits": {
                "learning_requests_per_hour": 7,
                "evaluation_sessions_per_day": 2,
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["limits"]["learning_requests_per_hour"] == 7
    assert response.json()["limits"]["evaluation_sessions_per_day"] == 2
