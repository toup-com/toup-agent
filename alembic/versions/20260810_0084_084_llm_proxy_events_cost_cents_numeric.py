"""084 — llm_proxy_events.cost_cents: Integer → Numeric(12,4).

R-3 (GA run, separately approved billing-visible change): the 1¢/call
floor in ``_calc_cost_cents`` is gone — a sub-cent call now records its
true fractional cost, capped at the legacy value so no row is ever
recorded HIGHER than it would have been. An Integer column would silently
re-floor every fraction on INSERT, so the storage moves to Numeric(12,4),
the same shape ``credit_ledger.underlying_cost_cents`` has had since the
credit system shipped.

Existing rows are NOT rewritten beyond the type cast (their values are
preserved exactly: 3 → 3.0000). **Forward-only by decision**: historical
rows carry costs computed under era pricing tables (#509 corrected two
models' rates on 2026-08-07), so "recomputing" them with today's table
would replace one known inaccuracy with a second, unlabelled one — and
``credit_ledger`` is append-only by design (corrections are compensating
rows, never rewrites). The floor's historical effect stays visible and
documented instead: every 1¢ row whose true cost was lower is an
overstatement of at most 1¢, bounded and era-labelled by this
migration's timestamp.

PLATFORM_ONLY (``app/db/models/base.py``): gated on ``settings.run_mode``
like revisions 078-083 — ``alembic upgrade head`` runs on boot in BOTH
images and tenants have no such table.

Lock note: int → numeric is a table REWRITE under ACCESS EXCLUSIVE. The
table measured 12,707 rows / 6.5 MB in production on 2026-08-10 —
sub-second. Metering writes retry through their own sessions; nothing
user-facing blocks on this table.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "084"
down_revision = "083"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.084")

_TABLE = "llm_proxy_events"
_COLUMN = "cost_cents"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.084] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.084] not a platform DB; skipping (%s is PLATFORM_ONLY)", _TABLE
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        logger.info("[alembic.084] %s absent; nothing to do", _TABLE)
        return

    col = next(
        (c for c in insp.get_columns(_TABLE) if c["name"] == _COLUMN), None
    )
    if col is None:
        logger.info("[alembic.084] %s.%s absent; nothing to do", _TABLE, _COLUMN)
        return

    if isinstance(col["type"], sa.Numeric) and not isinstance(col["type"], sa.Integer):
        logger.info("[alembic.084] %s.%s already Numeric", _TABLE, _COLUMN)
        return

    op.alter_column(
        _TABLE,
        _COLUMN,
        type_=sa.Numeric(12, 4),
        existing_type=sa.Integer(),
        postgresql_using=f"{_COLUMN}::numeric(12,4)",
    )
    logger.info("[alembic.084] %s.%s is now Numeric(12,4)", _TABLE, _COLUMN)


def downgrade() -> None:
    if not _is_platform_db():
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        return

    col = next(
        (c for c in insp.get_columns(_TABLE) if c["name"] == _COLUMN), None
    )
    if col is None or isinstance(col["type"], sa.Integer):
        return

    # round(), not trunc: closest-int restoration of the legacy shape.
    op.alter_column(
        _TABLE,
        _COLUMN,
        type_=sa.Integer(),
        existing_type=sa.Numeric(12, 4),
        postgresql_using=f"round({_COLUMN})::integer",
    )
