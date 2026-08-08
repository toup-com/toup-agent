"""082 — llm_proxy_events.channel, so cache cost can be attributed to a surface.

Prompt caching is prefix-exact, and the wire TOOLS ARRAY heads the prefix. A
channel that strips a tool therefore starts a separate provider cache lineage
and re-bills the entire request — measured 2026-08-08: the ``app`` channel
diverges from web at byte 10 of a 46,517-byte tools array, and there is no
partial credit (a cold voice turn read 0 cached while a simultaneous web
control read 18,070).

That makes "which surface is burning uncached tokens" a first-order cost
question, and until this column it was unanswerable. The value exists on the
agent — ``agent_runner`` threads ``channel`` through the whole turn — and was
dropped at the agent→proxy boundary. It is not recoverable from anything else
on the row: ``operation_type`` is NULL for every user-facing chat call *by
design* (that is what makes a call count toward the user's cap), and the
``[CACHE]`` log line carries only an 8-char hash of the cache key.

PLATFORM_ONLY (``app/db/models/base.py``): ``llm_proxy_events`` lives on the
platform DB only. Gated on ``settings.run_mode`` like revisions 078-081 — for
the reason recorded there, ``alembic upgrade head`` runs on boot in BOTH
images and tenants have no such table.

Nullable with no backfill and no default. Every existing row predates the
header and NULL is the honest value for them; a default would manufacture an
attribution nobody measured. The index leads with ``created_at`` because every
query this exists for is time-boxed.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "082"
down_revision = "081"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.082")

_TABLE = "llm_proxy_events"
_COLUMN = "channel"
_INDEX = "ix_llm_proxy_created_channel"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.082] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.082] not a platform DB; skipping (%s is PLATFORM_ONLY)", _TABLE
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        logger.info("[alembic.082] %s absent; nothing to do", _TABLE)
        return

    cols = {c["name"] for c in insp.get_columns(_TABLE)}
    if _COLUMN not in cols:
        # Nullable ADD COLUMN with no default: Postgres rewrites no rows and
        # holds ACCESS EXCLUSIVE only for the catalog update.
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(20), nullable=True))
        logger.info("[alembic.082] added %s.%s", _TABLE, _COLUMN)
    else:
        logger.info("[alembic.082] %s.%s already present", _TABLE, _COLUMN)

    insp.clear_cache()
    if _INDEX not in {i["name"] for i in insp.get_indexes(_TABLE)}:
        op.create_index(_INDEX, _TABLE, ["created_at", _COLUMN])
        logger.info("[alembic.082] created %s", _INDEX)


def downgrade() -> None:
    if not _is_platform_db():
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    if _TABLE not in set(insp.get_table_names()):
        return

    if _INDEX in {i["name"] for i in insp.get_indexes(_TABLE)}:
        op.drop_index(_INDEX, table_name=_TABLE)
    if _COLUMN in {c["name"] for c in insp.get_columns(_TABLE)}:
        op.drop_column(_TABLE, _COLUMN)
