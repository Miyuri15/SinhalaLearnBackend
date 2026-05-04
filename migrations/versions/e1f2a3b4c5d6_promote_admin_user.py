"""Promote configured admin user

Revision ID: e1f2a3b4c5d6
Revises: d4e5f6a7b8c9
Create Date: 2026-04-30 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ADMIN_EMAIL = "admin@sinhalalearn.com"


def upgrade() -> None:
    op.execute(
        """
        UPDATE users
        SET role = 'admin'
        WHERE lower(email) = lower('admin@sinhalalearn.com')
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE users
        SET role = 'user'
        WHERE lower(email) = lower('admin@sinhalalearn.com')
        """
    )
