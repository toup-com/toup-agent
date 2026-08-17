"""088 — admin_dispatches.last_error: why a dispatch reads `failed`

``run_dispatch_fanout`` writes ``status='sending'`` at the top and only its
own ``except`` writes ``failed``. A worker that is OOM-killed, redeployed or
loses its replica runs neither, so the row stays ``sending`` forever — and it
typically dies BEFORE materialising a single target row, which is why the
per-target ``last_error`` column cannot carry the explanation. The stall
sweeper (``admin_dispatch_worker.sweep_stalled_dispatches``) needs somewhere
to say what it concluded, or the operator reads "Failed" with no reason and no
idea that Retry is the remedy.

Nullable with no default: NULL means "no dispatch-level failure", which is the
correct value for every existing row.

Gated on ``settings.run_mode`` like revisions 078-085, and here it is not
merely prudence — ``admin_dispatches`` is PLATFORM_ONLY, so on a tenant DB the
table does not exist at all and the ALTER would raise rather than no-op.

Revision ID: 086
Revises: 085
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op


revision = "088"
down_revision = "087"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.088")

_DISPATCHES = "admin_dispatches"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.088] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.088] not a platform DB; skipping (%s is PLATFORM_ONLY)",
            _DISPATCHES,
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    # Table-existence guard, matching 085. A platform DB that has not yet run
    # 085 (or a partition where these tables legitimately do not exist) must
    # skip, not raise — `alembic upgrade head` runs on boot, so a raise here is
    # a boot loop rather than a failed migration someone reads about later.
    if _DISPATCHES not in set(insp.get_table_names()):
        logger.info("[alembic.088] %s absent; skipping", _DISPATCHES)
        return

    # Inspector-guarded, NOT `ADD COLUMN IF NOT EXISTS`. 085 says why in as
    # many words — sqlite does not accept that form — and the first version of
    # this revision used it anyway, so it would have raised on the sqlite test
    # harness while looking correct against Postgres. `create_all` also builds
    # this table from the models on a fresh platform DB before alembic sees it,
    # so the column really can already be present here.
    if "last_error" not in {c["name"] for c in insp.get_columns(_DISPATCHES)}:
        op.add_column(_DISPATCHES, sa.Column("last_error", sa.Text(), nullable=True))
    else:
        logger.info("[alembic.088] %s.last_error already present", _DISPATCHES)


def downgrade() -> None:
    # No-op, same reasoning as revisions 078-085: a blue-green rollout runs the
    # OLD and NEW images against the same DB for the drain window, and the
    # migration-lint guard rejects destructive patterns anywhere in the diff.
    logger.info("[alembic.088] downgrade is a no-op; the column is left in place")
