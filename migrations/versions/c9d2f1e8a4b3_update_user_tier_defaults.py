"""Update user tier defaults

Revision ID: c9d2f1e8a4b3
Revises: 4ac66e552875, b4d5e6f7a8b9
Create Date: 2026-04-30 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d2f1e8a4b3"
down_revision: Union[str, Sequence[str], None] = ("4ac66e552875", "b4d5e6f7a8b9")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE users
        SET tier = CASE
            WHEN tier IS NULL THEN 'basic'
            WHEN lower(tier) = 'normal' THEN 'basic'
            WHEN lower(tier) = 'basic' THEN 'basic'
            WHEN lower(tier) = 'classroom' THEN 'intermediate'
            WHEN lower(tier) = 'intermediate' THEN 'intermediate'
            WHEN lower(tier) = 'institution' THEN 'enterprise'
            WHEN lower(tier) = 'enterprise' THEN 'enterprise'
            ELSE 'basic'
        END
        """
    )
    op.alter_column("users", "tier", server_default="basic", nullable=False)


def downgrade() -> None:
    op.alter_column("users", "tier", server_default=None, nullable=False)
