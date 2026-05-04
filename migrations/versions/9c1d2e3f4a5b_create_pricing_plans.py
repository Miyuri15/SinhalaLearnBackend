"""Create editable pricing plans

Revision ID: 9c1d2e3f4a5b
Revises: 88837c0554e3
Create Date: 2026-05-01 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "9c1d2e3f4a5b"
down_revision: Union[str, None] = "88837c0554e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if "pricing_plans" not in inspector.get_table_names():
        op.create_table(
            "pricing_plans",
            sa.Column("tier_key", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("price_label", sa.String(), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("badge", sa.String(), nullable=False),
            sa.Column("features", sa.JSON(), nullable=False),
            sa.Column("cta", sa.String(), nullable=False),
            sa.Column("note", sa.Text(), nullable=False),
            sa.Column("learning_requests_per_hour", sa.Integer(), nullable=False),
            sa.Column("evaluation_sessions_per_day", sa.Integer(), nullable=False),
            sa.Column("evaluations_per_session", sa.Integer(), nullable=True),
            sa.Column("allow_evaluation_overage", sa.Boolean(), server_default=sa.false(), nullable=False),
            sa.Column("is_popular", sa.Boolean(), server_default=sa.false(), nullable=False),
            sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
            sa.Column("updated_by", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["updated_by"], ["users.id"]),
            sa.PrimaryKeyConstraint("tier_key"),
        )

    plans = [
        (
            "basic",
            "Basic Plan",
            "Free / forever",
            "A lightweight plan for getting started with Learning Mode",
            "Starter",
            ["Learning mode: 5 requests per hour", "Evaluation mode: 1 session per day", "Up to 10 evaluations per session", "Perfect for getting started"],
            "Start Free",
            "No credit card required",
            5,
            1,
            10,
            False,
            False,
            True,
        ),
        (
            "intermediate",
            "Intermediate Plan",
            "5000 LKR / tier",
            "For regular users who need more daily usage",
            "Most Popular",
            ["Learning mode: 20 requests per hour", "Evaluation mode: 5 sessions per day", "Built for steady classroom or personal use", "Priority access during busy periods"],
            "Choose Intermediate",
            "Usage resets apply",
            20,
            5,
            None,
            False,
            True,
            True,
        ),
        (
            "enterprise",
            "Enterprise Plan",
            "10000 LKR onwards / tier",
            "For teams and institutions that need the highest limits",
            "Best for Scale",
            ["Learning mode: 50 requests per hour", "Evaluation mode: 10 sessions per day", "Next evaluations are charged", "Designed for larger deployments"],
            "Contact Sales",
            "Usage resets apply",
            50,
            10,
            None,
            True,
            False,
            True,
        ),
    ]

    table = sa.table(
        "pricing_plans",
        sa.column("tier_key", sa.String()),
        sa.column("name", sa.String()),
        sa.column("price_label", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("badge", sa.String()),
        sa.column("features", sa.JSON()),
        sa.column("cta", sa.String()),
        sa.column("note", sa.Text()),
        sa.column("learning_requests_per_hour", sa.Integer()),
        sa.column("evaluation_sessions_per_day", sa.Integer()),
        sa.column("evaluations_per_session", sa.Integer()),
        sa.column("allow_evaluation_overage", sa.Boolean()),
        sa.column("is_popular", sa.Boolean()),
        sa.column("is_active", sa.Boolean()),
    )
    op.bulk_insert(
        table,
        [
            {
                "tier_key": tier_key,
                "name": name,
                "price_label": price_label,
                "description": description,
                "badge": badge,
                "features": features,
                "cta": cta,
                "note": note,
                "learning_requests_per_hour": learning_requests_per_hour,
                "evaluation_sessions_per_day": evaluation_sessions_per_day,
                "evaluations_per_session": evaluations_per_session,
                "allow_evaluation_overage": allow_evaluation_overage,
                "is_popular": is_popular,
                "is_active": is_active,
            }
            for (
                tier_key,
                name,
                price_label,
                description,
                badge,
                features,
                cta,
                note,
                learning_requests_per_hour,
                evaluation_sessions_per_day,
                evaluations_per_session,
                allow_evaluation_overage,
                is_popular,
                is_active,
            ) in plans
        ],
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "pricing_plans" in inspector.get_table_names():
        op.drop_table("pricing_plans")
