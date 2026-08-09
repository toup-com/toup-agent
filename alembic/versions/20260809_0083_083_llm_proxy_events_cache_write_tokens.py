"""083 — llm_proxy_events.cache_write_tokens: the last unstored cache number.

``cache_write_tokens`` has been extracted on both wires
(``_extract_openai_cache_write_tokens`` / ``_extract_responses_usage``) and
priced into ``cost_cents`` and the credit charge since G1 prep — and then
DROPPED at the persistence boundary. ``cached_tokens`` (075) answers "how
much did we re-read"; without the write side, "how much did we pay to
seed those reads" was recoverable only by grepping ``[CACHE]`` platform
logs, which rotate. Part of the gpt-5.6 family bills writes at a premium
(sol/luna 1.25x input; terra's measured write rate equals list input), so
this is a real money column, not a curiosity.

PLATFORM_ONLY (``app/db/models/base.py``): ``llm_proxy_events`` lives on
the platform DB only. Gated on ``settings.run_mode`` like revisions
078-082 — ``alembic upgrade head`` runs on boot in BOTH images and tenants
have no such table.

Nullable, no backfill, no default: NULL is the honest value for every
pre-083 row (the number was computed and thrown away; inventing 0 would
claim it was measured). Aggregations COALESCE to 0, same convention as
075's ``cached_tokens``. No new index — every query this serves is
time-boxed and rides ``ix_llm_proxy_user_created`` /
``ix_llm_proxy_created_channel``.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "083"
down_revision = "082"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.083")

_TABLE = "llm_proxy_events"
_COLUMN = "cache_write_tokens"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.083] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.083] not a platform DB; skipping (%s is PLATFORM_ONLY)", _TABLE
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        logger.info("[alembic.083] %s absent; nothing to do", _TABLE)
        return

    cols = {c["name"] for c in insp.get_columns(_TABLE)}
    if _COLUMN not in cols:
        # Nullable ADD COLUMN with no default: Postgres rewrites no rows and
        # holds ACCESS EXCLUSIVE only for the catalog update.
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.Integer(), nullable=True))
        logger.info("[alembic.083] added %s.%s", _TABLE, _COLUMN)
    else:
        logger.info("[alembic.083] %s.%s already present", _TABLE, _COLUMN)


def downgrade() -> None:
    if not _is_platform_db():
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        return

    if _COLUMN in {c["name"] for c in insp.get_columns(_TABLE)}:
        op.drop_column(_TABLE, _COLUMN)
