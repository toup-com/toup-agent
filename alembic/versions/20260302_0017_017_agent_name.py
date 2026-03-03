"""Add agent_name to agent_configs

Revision ID: 017
Revises: 016
Create Date: 2026-03-02

Adds agent_name column for storing the agent's display name in config.
"""
from alembic import op
import sqlalchemy as sa

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("agent_configs", sa.Column("agent_name", sa.String(100), nullable=True))


def downgrade():
    op.drop_column("agent_configs", "agent_name")
