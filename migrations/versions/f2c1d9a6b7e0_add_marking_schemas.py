"""add marking schemas

Revision ID: f2c1d9a6b7e0
Revises: f1443a5cdaea
Create Date: 2026-03-31 15:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision: str = "f2c1d9a6b7e0"
down_revision: Union[str, None] = "f1443a5cdaea"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)

    existing_tables = inspector.get_table_names()

    if "marking_schemas" not in existing_tables:
        op.create_table(
            "marking_schemas",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("evaluation_session_id", sa.UUID(), nullable=False),
            sa.Column("resource_id", sa.UUID(), nullable=True),
            sa.Column("is_confirmed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.ForeignKeyConstraint(["evaluation_session_id"], ["evaluation_sessions.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["resource_id"], ["resource_files.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("evaluation_session_id", name="uq_marking_schemas_evaluation_session_id"),
        )

        op.create_index(
            op.f("ix_marking_schemas_evaluation_session_id"),
            "marking_schemas",
            ["evaluation_session_id"],
            unique=False,
        )

    existing_tables = inspector.get_table_names()

    if "marking_schema_items" not in existing_tables:
        op.create_table(
            "marking_schema_items",
            sa.Column("id", sa.UUID(), nullable=False),
            sa.Column("marking_schema_id", sa.UUID(), nullable=False),
            sa.Column("question_id", sa.UUID(), nullable=True),
            sa.Column("question_number", sa.String(), nullable=False),
            sa.Column("question_text", sa.Text(), nullable=False),
            sa.Column("reference_text", sa.Text(), nullable=False),
            sa.Column("max_marks", sa.Integer(), nullable=True),
            sa.Column("part_name", sa.String(), nullable=True),
            sa.Column("source_lessons", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("source_pages", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("source_notes", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
            sa.ForeignKeyConstraint(["marking_schema_id"], ["marking_schemas.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

        op.create_index(
            op.f("ix_marking_schema_items_marking_schema_id"),
            "marking_schema_items",
            ["marking_schema_id"],
            unique=False,
        )

        op.create_index(
            op.f("ix_marking_schema_items_question_id"),
            "marking_schema_items",
            ["question_id"],
            unique=False,
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)

    existing_tables = inspector.get_table_names()

    # Drop marking_schema_items if exists
    if "marking_schema_items" in existing_tables:
        op.drop_index(op.f("ix_marking_schema_items_question_id"), table_name="marking_schema_items")
        op.drop_index(op.f("ix_marking_schema_items_marking_schema_id"), table_name="marking_schema_items")
        op.drop_table("marking_schema_items")

    # Drop marking_schemas if exists
    if "marking_schemas" in existing_tables:
        op.drop_index(op.f("ix_marking_schemas_evaluation_session_id"), table_name="marking_schemas")
        op.drop_table("marking_schemas")