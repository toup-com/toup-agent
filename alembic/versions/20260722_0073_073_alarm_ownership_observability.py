"""AlarmKit ownership + observability columns (2026-07-22 incident).

Revision ID: 073
Revises: 072
Create Date: 2026-07-22

Three nullable columns from the silent-T-0 reminder incident:

1. ``live_activity_devices.alarm_auth`` (String(16)) — AlarmKit
   authorization state as last reported by the app ('authorized' /
   'denied' / 'notDetermined' / 'unavailable'). The incident was
   undiagnosable because nothing anywhere recorded whether the phone
   could even ring a device alarm.

2. ``live_activity_devices.alarms_armed`` (Integer) — count of
   scheduled device alarms at last report.

3. ``live_activities.alarm_owned_at`` (DateTime) — set when the app
   reported that a device alarm (AlarmKit) owns this mission's fire;
   the fire lane then skips its loud restart + ring chain on that
   device instead of double-alerting over the ringing alarm.

Idempotent + guarded like 070-072: platform DBs only (skips DBs
without ``users``); both tables are PLATFORM_ONLY in base.py so the
agent-side init_db create_all path never touches them. The init_db
``_alter_statements`` mirror carries the same ADD COLUMN IF NOT
EXISTS statements as a belt-and-braces backstop.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "073"
down_revision = "072"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.073")

_COLUMNS = {
    "live_activity_devices": [
        ("alarm_auth", sa.String(16)),
        ("alarms_armed", sa.Integer()),
    ],
    "live_activities": [
        ("alarm_owned_at", sa.DateTime()),
    ],
}


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.073] users absent; skipping (not a platform DB)")
        return

    for table, columns in _COLUMNS.items():
        if table not in tables:
            logger.info("[alembic.073] %s absent; skipping", table)
            continue
        existing = {c["name"] for c in insp.get_columns(table)}
        for name, coltype in columns:
            if name in existing:
                logger.info("[alembic.073] %s.%s already present", table, name)
                continue
            op.add_column(table, sa.Column(name, coltype, nullable=True))
            logger.info("[alembic.073] added %s.%s", table, name)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    tables = set(insp.get_table_names())
    for table, columns in _COLUMNS.items():
        if table not in tables:
            continue
        existing = {c["name"] for c in insp.get_columns(table)}
        for name, _ in columns:
            if name in existing:
                op.drop_column(table, name)
