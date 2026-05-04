"""create api usage logs table

Revision ID: 88837c0554e3
Revises: e1f2a3b4c5d6
Create Date: 2026-04-30 21:10:57.746466
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "88837c0554e3"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if "api_usage_logs" not in inspector.get_table_names():
        op.create_table(
            "api_usage_logs",
            sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("request_id", sa.String(length=100), nullable=False),
            sa.Column("provider", sa.String(length=50), nullable=False),
            sa.Column("service_name", sa.String(length=100), nullable=False),
            sa.Column("model_name", sa.String(length=100), nullable=True),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("prompt_chars", sa.Integer(), nullable=True),
            sa.Column("response_chars", sa.Integer(), nullable=True),
            sa.Column("prompt_tokens", sa.Integer(), nullable=True),
            sa.Column("completion_tokens", sa.Integer(), nullable=True),
            sa.Column("total_tokens", sa.Integer(), nullable=True),
            sa.Column("attempt_number", sa.Integer(), nullable=True),
            sa.Column("max_retries", sa.Integer(), nullable=True),
            sa.Column("is_retry", sa.Boolean(), nullable=True),
            sa.Column("status", sa.String(length=30), nullable=False),
            sa.Column("error_type", sa.String(length=100), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("duration_ms", sa.Float(), nullable=True),
            sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    op.create_index("ix_api_usage_logs_created_at", "api_usage_logs", ["created_at"], unique=False, if_not_exists=True)
    op.create_index("ix_api_usage_logs_message_id", "api_usage_logs", ["message_id"], unique=False, if_not_exists=True)
    op.create_index("ix_api_usage_logs_request_id", "api_usage_logs", ["request_id"], unique=False, if_not_exists=True)
    op.create_index("ix_api_usage_logs_session_id", "api_usage_logs", ["session_id"], unique=False, if_not_exists=True)
    op.create_index("ix_api_usage_logs_user_id", "api_usage_logs", ["user_id"], unique=False, if_not_exists=True)

def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if "api_usage_logs" in inspector.get_table_names():
        op.drop_index(
            "ix_api_usage_logs_user_id",
            table_name="api_usage_logs",
            if_exists=True,
        )
        op.drop_index(
            "ix_api_usage_logs_session_id",
            table_name="api_usage_logs",
            if_exists=True,
        )
        op.drop_index(
            "ix_api_usage_logs_request_id",
            table_name="api_usage_logs",
            if_exists=True,
        )
        op.drop_index(
            "ix_api_usage_logs_message_id",
            table_name="api_usage_logs",
            if_exists=True,
        )
        op.drop_index(
            "ix_api_usage_logs_created_at",
            table_name="api_usage_logs",
            if_exists=True,
        )

        op.drop_table("api_usage_logs")