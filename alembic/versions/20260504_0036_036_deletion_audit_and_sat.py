"""deletion_audit_events + sensitive_action_redemptions

Revision ID: 036
Revises: 035
Create Date: 2026-05-04

§1.3 of the App Store submission readiness plan. Two new platform-only
tables to make POST /auth/delete-account safe and durable:

  1. deletion_audit_events — INSERT-at-start, UPDATE-at-end receipt
     for every delete-user attempt. Outlives the user (the user_id FK
     is intentionally not enforced because the user row is gone after
     a successful delete). Indefinite retention for non-PII columns;
     7-year retention for user_email / request_ip / request_user_agent
     followed by an irreversible-hash sweep job (TODO §1.3-followup,
     not built in this commit).

  2. sensitive_action_redemptions — single-use jti set for the
     fresh-JWT replay defense on /auth/delete-account. Persisted (vs
     in-memory LRU) so a platform restart mid-deletion doesn't open
     a replay window. Sweep job: DELETE WHERE expires_at < now() -
     interval '1 day' (TODO §1.3-followup).

Both tables are additive — no changes to existing schema, safe to
roll out alongside running pods that haven't been updated. Standard
alembic upgrade path.
"""

from alembic import op
import sqlalchemy as sa


revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "deletion_audit_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        # PII — 7-year retention, then sweep-hashed.
        sa.Column("user_email", sa.Text(), nullable=False),
        sa.Column("request_ip", sa.String(45), nullable=True),
        sa.Column("request_user_agent", sa.Text(), nullable=True),
        # Who triggered.
        sa.Column("actor", sa.String(16), nullable=False),
        sa.Column("actor_user_id", sa.String(36), nullable=True),
        # State machine.
        sa.Column(
            "initiated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default="in_progress",
        ),
        sa.Column("failure_step", sa.String(32), nullable=True),
        sa.Column("failure_detail", sa.Text(), nullable=True),
        # Per-step success flags. NULL = not reached / n/a.
        sa.Column("container_destroyed", sa.Boolean(), nullable=True),
        sa.Column("stripe_cleared", sa.Boolean(), nullable=True),
        sa.Column("openai_archived", sa.Boolean(), nullable=True),
        # Set by the redaction sweep job once PII has been hashed.
        sa.Column("pii_redacted_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "status IN ('in_progress', 'succeeded', 'failed')",
            name="ck_deletion_audit_events_status",
        ),
        sa.CheckConstraint(
            "actor IN ('self', 'admin')",
            name="ck_deletion_audit_events_actor",
        ),
    )
    op.create_index(
        "ix_deletion_audit_events_user_id",
        "deletion_audit_events",
        ["user_id"],
    )
    op.create_index(
        "ix_deletion_audit_events_initiated_at",
        "deletion_audit_events",
        ["initiated_at"],
    )

    op.create_table(
        "sensitive_action_redemptions",
        sa.Column("jti", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("purpose", sa.String(64), nullable=False),
        sa.Column(
            "redeemed_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_sensitive_action_redemptions_expires_at",
        "sensitive_action_redemptions",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_sensitive_action_redemptions_expires_at",
        table_name="sensitive_action_redemptions",
    )
    op.drop_table("sensitive_action_redemptions")

    op.drop_index(
        "ix_deletion_audit_events_initiated_at",
        table_name="deletion_audit_events",
    )
    op.drop_index(
        "ix_deletion_audit_events_user_id",
        table_name="deletion_audit_events",
    )
    op.drop_table("deletion_audit_events")
