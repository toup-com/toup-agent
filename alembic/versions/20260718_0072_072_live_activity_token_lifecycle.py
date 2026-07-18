"""Live Activity token lifecycle: install_id + one-started-per-device.

Revision ID: 072
Revises: 071
Create Date: 2026-07-18

Two hardenings from the 2026-07-16 shared-token ambiguity incident:

1. ``live_activity_devices.install_id`` (nullable String(64)): stable
   per-install identifier so a reinstall's rotated push-to-start token
   UPDATES the existing row instead of accreting a stale sibling that
   APNs happily 200s into the void forever.

2. Partial unique index ``uq_live_activities_device_started`` ON
   live_activities(device_id) WHERE status='started' — the DB-level
   backstop for the ONE-ACTIVITY-PER-DEVICE invariant. The incident
   root cause was two dispatcher replicas racing ``_preempt_device``
   (read-then-write, no constraint) and both committing a started row
   for the same device; every later tokenless send then hit the
   ambiguous_shared_token skip until the queue row failed out. With
   the index, the losing replica gets an IntegrityError, rolls back,
   and retries into the already_started dedup.

   MUST dedupe before creating the index: for each device with more
   than one started row, keep the newest ``started_at`` and end the
   rest (ended_at=now).

Idempotent + guarded like 070/071: platform DBs only (skips DBs
without ``users``); both tables are PLATFORM_ONLY in base.py so the
agent-side init_db create_all path never touches them — alembic alone
is the correct vehicle here (no init_db mirror needed for the index;
fresh DBs get install_id from create_all off the model).
"""
from __future__ import annotations

import logging
from datetime import datetime

import sqlalchemy as sa
from alembic import op


revision = "072"
down_revision = "071"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.072")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.072] users absent; skipping (not a platform DB)")
        return
    if "live_activity_devices" not in tables or "live_activities" not in tables:
        logger.info("[alembic.072] live activity tables absent; skipping")
        return

    # ── 1. install_id column + lookup index ───────────────────────
    device_cols = {c["name"] for c in insp.get_columns("live_activity_devices")}
    if "install_id" not in device_cols:
        op.add_column(
            "live_activity_devices",
            sa.Column("install_id", sa.String(64), nullable=True),
        )
        logger.info("[alembic.072] added live_activity_devices.install_id")
    else:
        logger.info("[alembic.072] install_id already present; skipping")

    device_indexes = {ix["name"] for ix in insp.get_indexes("live_activity_devices")}
    if "ix_live_activity_devices_user_install" not in device_indexes:
        op.create_index(
            "ix_live_activity_devices_user_install",
            "live_activity_devices", ["user_id", "install_id"],
        )

    # ── 2. dedupe started rows, then the partial unique index ─────
    la_indexes = {ix["name"] for ix in insp.get_indexes("live_activities")}
    if "uq_live_activities_device_started" in la_indexes:
        logger.info("[alembic.072] uq_live_activities_device_started already present; skipping")
        return

    # Dedupe: keep the newest started row per device (started_at, id
    # as tie-break), end the rest. Computed in Python for portability
    # (no DISTINCT ON / window functions — sqlite dev DBs).
    now = datetime.utcnow()
    rows = conn.execute(sa.text(
        "SELECT id, device_id FROM live_activities "
        "WHERE status = 'started' "
        "ORDER BY device_id, started_at DESC, id DESC"
    )).fetchall()
    seen_devices: set = set()
    losers: list = []
    for row_id, device_id in rows:
        if device_id in seen_devices:
            losers.append(row_id)
        else:
            seen_devices.add(device_id)
    if losers:
        conn.execute(
            sa.text(
                "UPDATE live_activities SET status = 'ended', "
                "ended_at = :now, updated_at = :now WHERE id IN :ids"
            ).bindparams(sa.bindparam("ids", expanding=True)),
            {"now": now, "ids": losers},
        )
        logger.info(
            "[alembic.072] deduped %d duplicate started rows before index", len(losers),
        )

    op.create_index(
        "uq_live_activities_device_started",
        "live_activities", ["device_id"],
        unique=True,
        postgresql_where=sa.text("status = 'started'"),
        sqlite_where=sa.text("status = 'started'"),
    )
    logger.info("[alembic.072] created uq_live_activities_device_started")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    tables = set(insp.get_table_names())

    if "live_activities" in tables:
        la_indexes = {ix["name"] for ix in insp.get_indexes("live_activities")}
        if "uq_live_activities_device_started" in la_indexes:
            op.drop_index("uq_live_activities_device_started", table_name="live_activities")

    if "live_activity_devices" in tables:
        device_indexes = {ix["name"] for ix in insp.get_indexes("live_activity_devices")}
        if "ix_live_activity_devices_user_install" in device_indexes:
            op.drop_index(
                "ix_live_activity_devices_user_install",
                table_name="live_activity_devices",
            )
        device_cols = {c["name"] for c in insp.get_columns("live_activity_devices")}
        if "install_id" in device_cols:
            op.drop_column("live_activity_devices", "install_id")
