"""Add setup_type to agent_configs and backfill from hosting_mode/setup_step

Revision ID: 019
Revises: 018
Create Date: 2026-04-08

Tracks whether the user chose Quick Setup ('auto') or Advanced Setup ('manual')
so the wizard can restore the correct flow on page reload.
"""
from alembic import op
import sqlalchemy as sa

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "agent_configs",
        sa.Column("setup_type", sa.String(20), nullable=True),
    )
    # Backfill: managed hosting → auto, step > 1 → manual, else null
    op.execute("""
        UPDATE agent_configs
        SET setup_type = CASE
            WHEN hosting_mode = 'managed' THEN 'auto'
            WHEN setup_step > 1 THEN 'manual'
            ELSE NULL
        END
    """)


def downgrade():
    op.drop_column("agent_configs", "setup_type")
