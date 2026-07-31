"""078 — search_events, the search gateway's per-user telemetry table.

Created for the platform-side search gateway, which moves the shared Brave key
off every tenant container and behind one chokepoint. The gateway is the only
thing that ever talks to Brave, so it is also the only place that can record
what was asked, how long it took, which rung answered, what was throttled, and
whose bill it lands on.

WHY A NEW TABLE, twice over:

  * Not ``credit_ledger``. That table is an immutable billing audit trail
    shared by 18 event types, indexed on ``(user_id, created_at)`` and
    ``(event_id)`` only. The search fields landed there as JSONB
    (``metadata->>'tier'``), so "searches per user per day" fleet-wide is a
    sequential scan of the entire billing history. Billing rows still get
    written — this table is telemetry, not a replacement.

  * Not ``llm_proxy_events``. Its ``provider`` and ``endpoint`` are String(20)
    and would have to carry tier semantics they were never named for, in a
    table whose name says LLM.

PLATFORM-ONLY, unlike migration 077. ``search_events`` is registered in
``app/db/models/base.py::PLATFORM_ONLY_TABLES``, so it lives only in the
platform DB and needs no mirrored ``ADD COLUMN IF NOT EXISTS`` in
``app/db/database.py``. That is a property of the gateway design: after this
change the tenant container never performs the upstream call, so it has
nothing to record.

NOTE ON REACH — ``alembic upgrade head`` runs on boot in BOTH images
(``Dockerfile`` for platform-api and ``Dockerfile.agent`` for every tenant), so
this revision is executed against agent databases too. The ``users`` guard
below is the house pattern for that (see revision 074) and must not be
removed: without it a failing revision here would halt an agent's migration
chain at 078 and silently strand every later one.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "078"
down_revision = "077"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.078")

_TABLE = "search_events"


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.078] users absent; skipping (not a platform DB)")
        return
    if _TABLE in tables:
        logger.info("[alembic.078] %s already present", _TABLE)
        return

    op.create_table(
        "search_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tier", sa.String(20), nullable=False),
        sa.Column("engine", sa.String(24), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="ok"),
        sa.Column("degraded_reason", sa.String(40), nullable=True),
        sa.Column("was_fallback", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("result_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("credits", sa.Numeric(12, 4), nullable=True),
        sa.Column("cost_cents", sa.Numeric(12, 4), nullable=True),
        sa.Column("charged", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("channel", sa.String(20), nullable=True),
        sa.Column("query_sha256", sa.String(16), nullable=True),
        sa.Column("brave_remaining", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # (user_id, created_at) answers per-user-per-day; (created_at) answers the
    # fleet-wide daily rollup without leading on user_id; (status, created_at)
    # answers "show me every throttle and fallback in the last 24h" — the one
    # the founder asked for that had no row anywhere before this table.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_search_events_user_created "
        "ON search_events (user_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_search_events_created "
        "ON search_events (created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_search_events_status_created "
        "ON search_events (status, created_at)"
    )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    if _TABLE not in set(insp.get_table_names()):
        return
    op.execute("DROP INDEX IF EXISTS ix_search_events_status_created")
    op.execute("DROP INDEX IF EXISTS ix_search_events_created")
    op.execute("DROP INDEX IF EXISTS ix_search_events_user_created")
    op.drop_table(_TABLE)
