"""Add stripe_customer_id to users table

Revision ID: 018
Revises: 017
Create Date: 2026-04-07

Links each user to their Stripe customer for billing portal,
duplicate prevention, and subscription lookups.
"""
from alembic import op
import sqlalchemy as sa

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column("stripe_customer_id", sa.String(255), nullable=True, unique=True),
    )
    op.create_index("ix_users_stripe_customer_id", "users", ["stripe_customer_id"], unique=True)


def downgrade():
    op.drop_index("ix_users_stripe_customer_id", table_name="users")
    op.drop_column("users", "stripe_customer_id")
