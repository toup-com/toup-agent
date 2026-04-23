"""006 — Add is_paired to telegram_user_mappings and new config columns

Revision ID: 006_dm_pairing
Revises: 004_memory_evolution
Create Date: 2026-02-07

Note (Phase 3 fix, 2026-04-22): down_revision was '005_add_workflows_table'
but that migration file was deleted at some point with no trace in the
repo or recent git history. Re-pointed to 004_memory_evolution to restore
the chain. If a production DB happens to be stamped at the phantom
'005_add_workflows_table' revision, operator must manually
`alembic stamp 006_dm_pairing` before running `alembic upgrade head`.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '006_dm_pairing'
down_revision = '004_memory_evolution'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'telegram_user_mappings',
        sa.Column('is_paired', sa.Boolean(), server_default='false', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('telegram_user_mappings', 'is_paired')
