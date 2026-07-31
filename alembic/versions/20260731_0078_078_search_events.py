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

NOTE ON REACH — this bit is load-bearing, and the first version of this
revision got it wrong in production.

``alembic upgrade head`` runs on boot in BOTH images (``Dockerfile`` for
platform-api and ``Dockerfile.agent`` for every tenant), so a revision here
executes against agent databases too. The house guard for that (revision 074)
is "skip if ``users`` is absent" — but ``users`` is a SHARED table and DOES
exist in every agent DB, so that guard does not skip anything. The first
version of this revision therefore created ``search_events`` inside the canary
tenant's agent database, and:

  * ``CREATE TABLE … REFERENCES users`` takes a lock on ``users``;
  * a blue/green upgrade runs the NEW container against the SAME database the
    OLD one is still serving from;
  * so green sat in ``green_health_wait`` for 258 s with 0 health checks
    passed, the bridge rolled it back, and CI gated the rollout
    (``aborted_canary_failed``, image ``bd1411770ad6``).

The gate below is therefore ``settings.run_mode``, the same signal
``init_db`` partitions on — not a table-presence heuristic. A PLATFORM_ONLY
table must never be created by a migration running in an agent container, for
the same reason ``init_db`` excludes it from ``create_all``.
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


def _is_platform_db() -> bool:
    """True only where a PLATFORM_ONLY table belongs.

    Reads ``settings.run_mode`` — the same signal ``init_db`` uses to decide
    which tables to create — so the migration and the ORM cannot disagree.
    "monolith" is included because legacy single-database installs hold every
    table. If the setting cannot be read at all, skip: creating this table
    where it does not belong is the failure mode that cost a rollout, and not
    creating it merely means search telemetry waits for the next boot.
    """
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.078] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.078] not a platform DB; skipping (search_events is PLATFORM_ONLY)")
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.078] users absent; skipping")
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
