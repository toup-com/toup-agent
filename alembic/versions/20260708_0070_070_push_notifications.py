"""Push notifications: push_devices + notification_queue (Autopilot PR2).

Revision ID: 070
Revises: 069
Create Date: 2026-07-08

Backs app/db/models/notification.py — the proactive-notification
substrate the Autopilot heartbeat engine reports through:

- push_devices: Expo push tokens, one row per mobile device. Platform
  holds the tokens; agent containers never see them.
- notification_queue: durable send queue. Agents enqueue via
  POST /api/agent/notify (idempotent on (user_id, idempotency_key));
  the platform dispatcher (PR3) claims rows with a status CAS because
  platform-api runs 2 replicas with no leader election.

Idempotent + guarded: creates the tables only when absent and only on
DBs that already host `users` (platform DBs, not per-tenant agent DBs
— both tables are PLATFORM_ONLY in base.py, so agent-side init_db
create_all excludes them; this guard keeps a stray alembic run on an
agent DB from re-introducing them there).

NOTE (numbering): the unmerged feat/toup-code-claude-parity branch
carries uncommitted local migrations numbered 070/071 — those must be
renumbered on top of this one when that branch rebases onto main.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "070"
down_revision = "069"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.070")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.070] users absent; skipping (not a platform DB)")
        return

    if "push_devices" not in tables:
        op.create_table(
            "push_devices",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("expo_push_token", sa.String(200), nullable=False),
            sa.Column("platform", sa.String(16), nullable=False),
            sa.Column("device_name", sa.String(200), nullable=True),
            sa.Column("app_version", sa.String(40), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(), nullable=True),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_push_devices_user_id", "push_devices", ["user_id"])
        op.create_index("ix_push_devices_revoked_at", "push_devices", ["revoked_at"])
        op.create_index(
            "ix_push_devices_user_revoked", "push_devices",
            ["user_id", "revoked_at"],
        )
        op.create_unique_constraint(
            "uq_push_devices_expo_push_token", "push_devices",
            ["expo_push_token"],
        )
        logger.info("[alembic.070] created push_devices")
    else:
        logger.info("[alembic.070] push_devices already present; skipping")

    if "notification_queue" not in tables:
        op.create_table(
            "notification_queue",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("source", sa.String(16), nullable=False, server_default="agent"),
            sa.Column("event_kind", sa.String(32), nullable=False),
            sa.Column("title", sa.String(200), nullable=False),
            sa.Column("body", sa.Text(), nullable=True),
            sa.Column("data_json", sa.JSON().with_variant(
                postgresql.JSONB(), "postgresql"), nullable=True),
            sa.Column("priority", sa.String(12), nullable=False, server_default="default"),
            sa.Column("dedup_key", sa.String(128), nullable=True),
            sa.Column("idempotency_key", sa.String(128), nullable=True),
            sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("scheduled_for", sa.DateTime(), nullable=True),
            sa.Column("claimed_at", sa.DateTime(), nullable=True),
            sa.Column("sent_at", sa.DateTime(), nullable=True),
            sa.Column("channels_json", sa.JSON().with_variant(
                postgresql.JSONB(), "postgresql"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_notification_queue_user_id", "notification_queue", ["user_id"])
        op.create_index("ix_notification_queue_status", "notification_queue", ["status"])
        op.create_index(
            "ix_notification_queue_claim", "notification_queue",
            ["status", "scheduled_for"],
        )
        op.create_index(
            "ix_notification_queue_user_dedup", "notification_queue",
            ["user_id", "dedup_key", "created_at"],
        )
        op.create_unique_constraint(
            "uq_notification_queue_user_idem", "notification_queue",
            ["user_id", "idempotency_key"],
        )
        logger.info("[alembic.070] created notification_queue")
    else:
        logger.info("[alembic.070] notification_queue already present; skipping")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    tables = set(insp.get_table_names())
    if "notification_queue" in tables:
        op.drop_table("notification_queue")
    if "push_devices" in tables:
        op.drop_table("push_devices")
