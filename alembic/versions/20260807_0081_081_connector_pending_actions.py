"""080 — connector_pending_actions, the confirmation gate for elevated tools.

``elevation: true`` has been declared on ten connector tools since the
connector arc shipped (``gmail__send_message``, ``outlook__send_message``,
``linkedin__share_post``, ``calendar__create_event``, the sheets/drive writes,
``github__create_comment``) and read by nothing. The dispatcher went straight
to ``provider.execute``. This table is where a staged call waits for the user
to approve it from the chat card.

PLATFORM_ONLY (see ``app/db/models/base.py``): the OAuth tokens that execute
these calls are platform-side, so approval runs the tool here without a
round-trip to a tenant container — an agent recycled between staging and
approval must not be able to strand a draft the user already said yes to.

Gated on ``settings.run_mode`` like revisions 078 and 079, for the same
hard-learned reason: ``alembic upgrade head`` runs on boot in BOTH images, a
``CREATE TABLE … REFERENCES users`` takes a lock on ``users``, and taking that
lock inside a tenant DB during a blue/green rollout is what stalled the 078
canary. Tenants never need this table at all.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "081"
down_revision = "080"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.080")

_TABLE = "connector_pending_actions"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.080] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.080] not a platform DB; skipping (%s is PLATFORM_ONLY)",
            _TABLE,
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.080] users absent; skipping")
        return
    if _TABLE in tables:
        logger.info("[alembic.080] %s already present", _TABLE)
        return

    op.create_table(
        _TABLE,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("connector_id", sa.String(64), nullable=False),
        sa.Column("tool_name", sa.String(128), nullable=False),
        # Tool ARGUMENTS only. connector_id/tool_name are columns so the
        # approve endpoint can never be talked into running a different
        # tool than the one the card showed.
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="pending",
        ),
        sa.Column("channel", sa.String(32), nullable=False, server_default="web"),
        sa.Column("conversation_id", sa.String(36), nullable=True),
        sa.Column("message_id", sa.String(36), nullable=True),
        sa.Column("agent_request_id", sa.String(64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("decided_via", sa.String(32), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'executed', 'failed', "
            "'rejected', 'expired')",
            name="ck_connector_pending_actions_status",
        ),
    )
    # Drives the card-reload query (`?conversation_id=`) and the
    # "anything waiting for me?" badge.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_connector_pending_actions_user_status "
        f"ON {_TABLE} (user_id, status)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_connector_pending_actions_conversation "
        f"ON {_TABLE} (conversation_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_connector_pending_actions_user_id "
        f"ON {_TABLE} (user_id)"
    )
    # The expiry sweep scans pending rows by deadline. Partial-index the
    # status so the sweep never walks the (much larger, and permanently
    # growing) terminal history.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_connector_pending_actions_expiry "
        f"ON {_TABLE} (expires_at) WHERE status = 'pending'"
    )


def downgrade() -> None:
    # No-op, same reasoning as revisions 078 and 079: a blue-green rollout
    # runs the OLD and NEW images against the same DB for the drain window,
    # and the migration-lint guard rejects destructive patterns anywhere in
    # the diff — including ones that "only" sit in downgrade(), which is
    # exactly the kind that gets run by accident during an incident.
    logger.info(
        "[alembic.080] downgrade is a no-op; %s is left in place (see comment)",
        _TABLE,
    )
