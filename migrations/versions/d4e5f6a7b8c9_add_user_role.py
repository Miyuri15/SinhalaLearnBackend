"""Add user role

Revision ID: d4e5f6a7b8c9
Revises: c9d2f1e8a4b3
Create Date: 2026-04-30 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c9d2f1e8a4b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("role", sa.String(), server_default="user", nullable=False),
        if_not_exists=True,
    )
    op.execute("UPDATE users SET role = 'user' WHERE role IS NULL")
    op.alter_column("users", "role", server_default="user", nullable=False)


def downgrade() -> None:
    op.drop_column("users", "role", if_exists=True)
