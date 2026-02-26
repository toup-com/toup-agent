"""Add password_plain column for admin visibility

Revision ID: 013_password_plain
Revises: 012_beta_access
Create Date: 2026-02-18
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "013_password_plain"
down_revision = "012_beta_access"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("password_plain", sa.String(255), nullable=True))


def downgrade():
    op.drop_column("users", "password_plain")
