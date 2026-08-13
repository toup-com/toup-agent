"""085 — admin dispatch: operator→user announcements, targets, thread.

Three tables behind the admin panel's Dispatch tab: the send itself
(``admin_dispatches``), the per-recipient fan-out ledger the worker
CAS-claims (``admin_dispatch_targets``), and the ongoing operator↔user
thread a ``persistent`` dispatch opens (``admin_thread_messages``).

PLATFORM_ONLY (``app/db/models/base.py``). The thread in particular is a
conversation between the OPERATOR and the user — not agent data. The agent
must never read it, the panel must read every user's replies without
fanning out to N containers, and a reply has to keep working while the
user's container is down. The only thing that reaches a tenant DB is the
chat card, written over HTTP by the fan-out worker.

Gated on ``settings.run_mode`` like revisions 078-084, for the same
hard-learned reason: ``alembic upgrade head`` runs on boot in BOTH images,
a ``CREATE TABLE … REFERENCES users`` takes a lock on ``users``, and taking
that lock inside a tenant DB during a blue/green rollout is what stalled
the 078 canary. Tenants never need these tables at all.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "085"
down_revision = "084"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.085")

_DISPATCHES = "admin_dispatches"
_TARGETS = "admin_dispatch_targets"
_THREAD = "admin_thread_messages"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.085] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.085] not a platform DB; skipping (%s/%s/%s are PLATFORM_ONLY)",
            _DISPATCHES, _TARGETS, _THREAD,
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.085] users absent; skipping")
        return

    if _DISPATCHES not in tables:
        op.create_table(
            _DISPATCHES,
            sa.Column("id", sa.String(36), primary_key=True),
            # SET NULL: an audit reference to the admin who sent it —
            # deleting that account must not erase the dispatch.
            sa.Column(
                "created_by_user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("mode", sa.String(16), nullable=False, server_default="once"),
            sa.Column(
                "audience", sa.String(16), nullable=False, server_default="user",
            ),
            # Set iff audience = 'user'.
            sa.Column(
                "target_user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column(
                "sender_name", sa.String(80), nullable=False, server_default="Toup",
            ),
            sa.Column("title", sa.String(200), nullable=False),
            sa.Column("body", sa.Text(), nullable=False),
            sa.Column(
                "urgent", sa.Boolean(), nullable=False, server_default="false",
            ),
            sa.Column(
                "status", sa.String(16), nullable=False, server_default="queued",
            ),
            sa.Column(
                "target_count", sa.Integer(), nullable=False, server_default="0",
            ),
            sa.Column(
                "delivered_count", sa.Integer(), nullable=False, server_default="0",
            ),
            sa.Column(
                "read_count", sa.Integer(), nullable=False, server_default="0",
            ),
            sa.Column(
                "failed_count", sa.Integer(), nullable=False, server_default="0",
            ),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
        )
    else:
        logger.info("[alembic.085] %s already present", _DISPATCHES)

    if _TARGETS not in tables:
        op.create_table(
            _TARGETS,
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "dispatch_id",
                sa.String(36),
                sa.ForeignKey("admin_dispatches.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "state", sa.String(16), nullable=False, server_default="pending",
            ),
            sa.Column(
                "chat_status", sa.String(24), nullable=False,
                server_default="pending",
            ),
            sa.Column("chat_message_id", sa.String(36), nullable=True),
            sa.Column("notification_id", sa.String(36), nullable=True),
            # The only read receipt in the system.
            sa.Column("read_at", sa.DateTime(), nullable=True),
            sa.Column(
                "attempts", sa.Integer(), nullable=False, server_default="0",
            ),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            # Replay guard for the retry sweep: re-running a dispatch must
            # never add a second row for a user it already reached.
            sa.UniqueConstraint(
                "dispatch_id", "user_id", name="uq_admin_dispatch_target",
            ),
        )
    else:
        logger.info("[alembic.085] %s already present", _TARGETS)

    if _THREAD not in tables:
        op.create_table(
            _THREAD,
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            # No FK: a provenance stamp, and the user's thread must
            # outlive the dispatch row that opened it.
            sa.Column("dispatch_id", sa.String(36), nullable=True),
            sa.Column("direction", sa.String(4), nullable=False),
            sa.Column("body", sa.Text(), nullable=False),
            sa.Column("author_admin_id", sa.String(36), nullable=True),
            # The operator's display name at write time — one operator
            # identity across the chat card and the thread. Nullable: a
            # user's `in` row has no operator name.
            sa.Column("sender_name", sa.String(80), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("read_at", sa.DateTime(), nullable=True),
            sa.Column("admin_read_at", sa.DateTime(), nullable=True),
        )
    else:
        logger.info("[alembic.085] %s already present", _THREAD)

    # `sender_name` was added to this revision after it had already run on
    # some dev DBs, and `create_all` builds these tables on a fresh platform
    # DB before alembic ever sees them — same reason the indexes below run
    # unconditionally. Inspector-guarded rather than
    # `ADD COLUMN IF NOT EXISTS`, which sqlite does not accept.
    insp.clear_cache()
    if _THREAD in set(insp.get_table_names()):
        if "sender_name" not in {c["name"] for c in insp.get_columns(_THREAD)}:
            op.add_column(_THREAD, sa.Column("sender_name", sa.String(80), nullable=True))

    # Indexes are created unconditionally with IF NOT EXISTS: a table that
    # already existed (create_all ran first on a fresh platform DB) still
    # needs them, and re-running is free.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_dispatches_status "
        f"ON {_DISPATCHES} (status)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_dispatch_targets_dispatch_id "
        f"ON {_TARGETS} (dispatch_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_dispatch_targets_user_id "
        f"ON {_TARGETS} (user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_dispatch_targets_state "
        f"ON {_TARGETS} (state)"
    )
    # The fan-out worker's claim scan: pending rows of one dispatch.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_dispatch_targets_claim "
        f"ON {_TARGETS} (state, dispatch_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_thread_messages_user_id "
        f"ON {_THREAD} (user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_thread_messages_created_at "
        f"ON {_THREAD} (created_at)"
    )
    # Both readers page the same way: one user's thread, newest last.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_admin_thread_user_created "
        f"ON {_THREAD} (user_id, created_at)"
    )


def downgrade() -> None:
    # No-op, same reasoning as revisions 078-084: a blue-green rollout runs
    # the OLD and NEW images against the same DB for the drain window, and
    # the migration-lint guard rejects destructive patterns anywhere in the
    # diff — including ones that "only" sit in downgrade(), which is exactly
    # the kind that gets run by accident during an incident.
    logger.info(
        "[alembic.085] downgrade is a no-op; the three tables are left in "
        "place (see comment)",
    )
