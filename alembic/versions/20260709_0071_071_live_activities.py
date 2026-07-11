"""iOS Live Activities: live_activity_devices + live_activities.

Revision ID: 071
Revises: 070
Create Date: 2026-07-09

Backs app/db/models/live_activity.py — the Autopilot phone surface:
push-to-start tokens per device, plus one row per (device, mission)
Live Activity the platform has started via APNs. Progress ticks and
terminal transitions from the agent update the lock-screen card and
Dynamic Island even when the app is force-quit.

Idempotent + guarded like 070: platform DBs only (skips DBs without
`users`); both tables are PLATFORM_ONLY in base.py so agent-side
init_db create_all excludes them.

NOTE (numbering): the unmerged feat/toup-code-claude-parity branch
carries uncommitted local migrations formerly numbered 070/071 —
those must be renumbered past THIS one when that branch rebases.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "071"
down_revision = "070"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.071")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.071] users absent; skipping (not a platform DB)")
        return

    if "live_activity_devices" not in tables:
        op.create_table(
            "live_activity_devices",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("push_to_start_token", sa.String(200), nullable=False),
            sa.Column(
                "apns_environment", sa.String(12),
                nullable=False, server_default="development",
            ),
            sa.Column("device_name", sa.String(200), nullable=True),
            sa.Column("app_version", sa.String(40), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(), nullable=True),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
        )
        op.create_index(
            "ix_live_activity_devices_user_id",
            "live_activity_devices", ["user_id"],
        )
        op.create_index(
            "ix_live_activity_devices_revoked_at",
            "live_activity_devices", ["revoked_at"],
        )
        op.create_index(
            "ix_live_activity_devices_user_revoked",
            "live_activity_devices", ["user_id", "revoked_at"],
        )
        op.create_unique_constraint(
            "uq_live_activity_devices_push_to_start_token",
            "live_activity_devices", ["push_to_start_token"],
        )
        logger.info("[alembic.071] created live_activity_devices")
    else:
        logger.info("[alembic.071] live_activity_devices already present; skipping")

    if "live_activities" not in tables:
        op.create_table(
            "live_activities",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("mission_id", sa.String(64), nullable=False),
            sa.Column(
                "device_id", sa.String(36),
                sa.ForeignKey("live_activity_devices.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("activity_push_token", sa.String(200), nullable=True),
            sa.Column(
                "apns_environment", sa.String(12),
                nullable=False, server_default="development",
            ),
            sa.Column(
                "status", sa.String(16),
                nullable=False, server_default="started",
            ),
            sa.Column("last_progress", sa.Integer(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("ended_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_live_activities_user_id", "live_activities", ["user_id"])
        op.create_index("ix_live_activities_status", "live_activities", ["status"])
        op.create_index(
            "ix_live_activities_mission_status",
            "live_activities", ["mission_id", "status"],
        )
        op.create_unique_constraint(
            "uq_live_activities_device_mission",
            "live_activities", ["device_id", "mission_id"],
        )
        logger.info("[alembic.071] created live_activities")
    else:
        logger.info("[alembic.071] live_activities already present; skipping")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    tables = set(insp.get_table_names())
    if "live_activities" in tables:
        op.drop_table("live_activities")
    if "live_activity_devices" in tables:
        op.drop_table("live_activity_devices")
