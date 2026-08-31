"""098 — live_activity_devices: foreground_until (the presence gate).

One column: until when this device is believed to be showing the app.
The Live Activity lane consults it before every push-to-start of an
out-of-app card, so a phone whose owner is watching the answer arrive
in the thread no longer gets a lock-screen card about it.

A TTL rather than a boolean, and it fails OPEN on expiry: the "I left"
report is the one that can be lost (a suspended or killed app sends
nothing), and a stuck `true` would silence every card on the device
forever. At worst a card the user did not need; never a reminder that
never arrived.

Platform DB only, same guard as 095/096/097 (live_activity_devices is
PLATFORM_ONLY; tenant DBs must not grow it).

Revision ID: 098
Revises: 097
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "098"
down_revision = "097"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

_DEVICES = "live_activity_devices"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.098] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.098] not a platform DB; skipping "
                    "(%s is PLATFORM_ONLY)", _DEVICES)
        return
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _DEVICES not in set(insp.get_table_names()):
        logger.info("[alembic.098] %s absent; model create_all covers it",
                    _DEVICES)
        return
    cols = {c["name"] for c in insp.get_columns(_DEVICES)}
    if "foreground_until" not in cols:
        op.add_column(
            _DEVICES,
            sa.Column("foreground_until", sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    if not _is_platform_db():
        return
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _DEVICES not in set(insp.get_table_names()):
        return
    cols = {c["name"] for c in insp.get_columns(_DEVICES)}
    if "foreground_until" in cols:
        op.drop_column(_DEVICES, "foreground_until")
