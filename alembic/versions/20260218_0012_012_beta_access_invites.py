"""012 — Beta access: add role to users + invites table.

Adds a `role` column to the `users` table (admin | beta_user) and
creates an `invites` table for closed-beta invite management.
Existing users (including the default "hex" user) get role='admin'.

Revision ID: 012_beta_access
Revises: 011_retrieval_feedback
"""

from alembic import op
import sqlalchemy as sa

revision = "012_beta_access"
down_revision = "011_retrieval_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add role column to users (default beta_user for new users)
    op.add_column(
        "users",
        sa.Column("role", sa.String(20), nullable=False, server_default="beta_user"),
    )
    op.create_index("ix_users_role", "users", ["role"])

    # 2. Promote ALL existing users to admin (they are pre-beta users / you)
    op.execute("UPDATE users SET role = 'admin'")

    # 3. Create invites table
    op.create_table(
        "invites",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("token", sa.String(64), unique=True, nullable=False, index=True),
        sa.Column("created_by", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("role", sa.String(20), nullable=False, server_default="beta_user"),
        sa.Column("note", sa.String(500), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("used_by", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("used_at", sa.DateTime, nullable=True),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_invites_status", "invites", ["status"])


def downgrade() -> None:
    op.drop_table("invites")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_column("users", "role")
