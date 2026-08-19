"""093 — admin_thread_attachments: a picture in the operator↔user thread.

The thread was text-only in both directions, so a user reporting something they
can SEE had to describe it, and an operator answering had to describe back. That
is the one conversation in the product where a screenshot is the whole message.

**Bytes live in the platform DB, and that is a decision with a precedent rather
than a shortcut.** ``support_attachments`` (models/support.py) already stores
mobile screenshots exactly this way, for the same trust boundary — owner or
admin — and its docstring records the reasoning: the Railway platform-api has
ephemeral disk, Message/Conversation attachments are AGENT_ONLY and therefore
unavailable to a platform-side table, and low volume + a hard size cap make DB
storage appropriate. ``file_storage.py``'s S3Backend stub is the documented
scale-up seam when that stops being true.

Two things bound the growth, because an unbounded blob column in this database
has hurt before (the audio blob cache reached 96% of it):

  * a per-file cap enforced at the route (``admin_thread_attachment_max_bytes``)
  * ON DELETE CASCADE from the message, so 092's deletion story stays complete —
    clearing a body must not leave its picture behind, which would be the most
    embarrassing possible outcome of a "delete for everyone".

PLATFORM_ONLY and run_mode-gated like 078-092: ``alembic upgrade head`` runs on
boot in BOTH images, and ``admin_thread_messages`` does not exist in a tenant DB
at all.

Chain: … → 091 (revoke audit) → 092 (thread deletion) → 093.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "093"
down_revision = "092"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.093")

_TABLE = "admin_thread_attachments"
_PARENT = "admin_thread_messages"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.093] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.093] not a platform DB; skipping (%s is PLATFORM_ONLY)", _TABLE)
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()

    tables = set(insp.get_table_names())
    if _PARENT not in tables:
        logger.info("[alembic.093] %s absent; skipping", _PARENT)
        return
    if _TABLE in tables:
        logger.info("[alembic.093] %s already present", _TABLE)
        return

    op.create_table(
        _TABLE,
        sa.Column("id", sa.String(36), primary_key=True),
        # CASCADE: 092 can clear a body, and a picture that outlived the words
        # it belonged to would be a deletion that only looked complete.
        sa.Column(
            "message_id",
            sa.String(36),
            sa.ForeignKey(f"{_PARENT}.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.Column("mime_type", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        # Lets a duplicate upload be recognised, and gives an integrity check
        # that does not require reading the blob back out through the ORM.
        sa.Column("sha256", sa.String(64), nullable=True),
        sa.Column("uploaded_by_user_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    # The only read pattern: "the attachments of these messages", after a thread
    # has already been loaded. Nothing scans this table by anything else.
    op.create_index(
        "ix_admin_thread_attachments_message_id", _TABLE, ["message_id"],
    )


def downgrade() -> None:
    # No-op, same stance as 078-092. R6 asks for reversible migrations; a
    # blue/green rollout runs the OLD and NEW images against one database for
    # the drain window, so taking this table away would break the still-live
    # old image rather than restore it.
    #
    # And the specific reason here: this table holds the only copy of the bytes.
    # A downgrade that ran cleanly would destroy user-supplied pictures with no
    # way back — the most misleading possible kind of success.
    logger.info("[alembic.093] downgrade is a no-op; the attachment table is left in place")
