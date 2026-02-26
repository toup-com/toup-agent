"""006 — Add is_paired to telegram_user_mappings and new config columns

Revision ID: 006_dm_pairing
Revises: 005_add_workflows_table
Create Date: 2026-02-07
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '006_dm_pairing'
down_revision = '005_add_workflows_table'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'telegram_user_mappings',
        sa.Column('is_paired', sa.Boolean(), server_default='false', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('telegram_user_mappings', 'is_paired')
