"""097 — automation_grants: mode_changed_at (Round 29).

One audit column: when the user last flipped a grant's auto/confirm
mode from the Overview (`PATCH /api/automations/{id}/mode`). The mode
itself is user consent, so the flip is user-JWT-only — this column is
the audit trail that the current mode is not the one on the original
approval card.

Platform DB only, same guard as 095/096 (automation_grants is
PLATFORM_ONLY; tenant DBs must not grow it).

Revision ID: 097
Revises: 096
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "097"
down_revision = "096"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

_GRANTS = "automation_grants"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.097] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.097] not a platform DB; skipping "
                    "(%s is PLATFORM_ONLY)", _GRANTS)
        return
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _GRANTS not in set(insp.get_table_names()):
        logger.info("[alembic.097] %s absent; model create_all covers it",
                    _GRANTS)
        return
    cols = {c["name"] for c in insp.get_columns(_GRANTS)}
    if "mode_changed_at" not in cols:
        op.add_column(
            _GRANTS,
            sa.Column("mode_changed_at", sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    if not _is_platform_db():
        return
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _GRANTS not in set(insp.get_table_names()):
        return
    cols = {c["name"] for c in insp.get_columns(_GRANTS)}
    if "mode_changed_at" in cols:
        op.drop_column(_GRANTS, "mode_changed_at")
