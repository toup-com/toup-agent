"""credential_access_log — immutable audit row per plaintext fetch

Revision ID: 024
Revises: 023
Create Date: 2026-04-24

Vault CP1: every call to GET /streaming/credentials/internal/{channel} that
gets past auth writes a row here BEFORE the plaintext is returned. If this
INSERT cannot commit, the endpoint returns 503 and the fetch fails closed —
plaintext never leaves the DB.

Baseline for CP2 (encryption) and CP3 (collapse dual read path): once
deployed, every real Netflix auto-login produces exactly one row here.
A flip in that invariant (zero rows, or two) flags a regression.
"""
from alembic import op
import sqlalchemy as sa


revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "credential_access_log",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("channel", sa.String(50), nullable=False),
        sa.Column("agent_key_fingerprint", sa.String(16), nullable=False),
        sa.Column("source", sa.String(16), nullable=False, server_default="http"),
        sa.Column(
            "fetched_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_credential_access_log_user_id",
        "credential_access_log",
        ["user_id"],
    )
    op.create_index(
        "ix_credential_access_log_fetched_at",
        "credential_access_log",
        ["fetched_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_credential_access_log_fetched_at", table_name="credential_access_log")
    op.drop_index("ix_credential_access_log_user_id", table_name="credential_access_log")
    op.drop_table("credential_access_log")
